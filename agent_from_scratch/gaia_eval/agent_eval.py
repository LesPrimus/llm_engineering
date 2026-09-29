"""The agent on GAIA: put the questions to the agent loop, score its final answers.

The agent gets web search and the calculator, and up to :data:`MAX_STEPS` turns
to use them. It cannot open the files some tasks attach yet, so those are told
so (:data:`prompts.FILE_NOTE`) and declining is available to them; a decline
still scores zero.

Models run one at a time, their tasks concurrently, which keeps the progress bar
readable and one provider's rate limit away from another's.

A task that runs out of steps without answering is recorded as an error, and
every attempt keeps the steps it used, so a miss that gave up can be told from
one that answered wrong.

Before the first run:

- accept GAIA's terms on the Hub (the dataset is gated), then put ``HF_TOKEN``
  in ``.env`` or run ``uv run huggingface-cli login``;
- put ``TAVILY_API_KEY`` in ``.env``, and the key for each model's provider
  (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...).

Run it from the repo root. Every flag is optional; with none, it runs
:data:`MODELS` over the whole validation split::

    # A dry run: the first 10 level 1 tasks.
    uv run python -m agent_from_scratch.gaia_eval.agent_eval --limit 10 --level 1

    # The whole split, every attempt saved as JSONL for a second look.
    uv run python -m agent_from_scratch.gaia_eval.agent_eval --out runs/agent.jsonl

    # Chosen tasks, in this order. Leave --level off: it hides the other levels.
    uv run python -m agent_from_scratch.gaia_eval.agent_eval --task-ids <id> <id>

    # Other models, by their litellm name; each gets its own scoreboard row.
    uv run python -m agent_from_scratch.gaia_eval.agent_eval \\
        --models gpt-5 anthropic/claude-haiku-4-5 --limit 10

    # Every flag, with its help.
    uv run python -m agent_from_scratch.gaia_eval.agent_eval --help

To pick out the misses from a saved run::

    jq -r 'select(.correct | not) | [.task_id, .expected, .answer] | @tsv' runs/agent.jsonl

This one spends money: up to MAX_STEPS calls per task, plus a Tavily search for
each one the model asks for.
"""

import argparse
import asyncio
import functools
import json
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from agent_from_scratch.agent import Agent
from agent_from_scratch.client import LlmClient
from agent_from_scratch.models import ExecutionContext
from agent_from_scratch.tools.base import tool
from agent_from_scratch.tools.calculator import calculate
from agent_from_scratch.tools.web_search import search

from .constants import Level, Split
from .dataset import GaiaDataset
from .models import GaiaReply, GaiaTask
from .prompts import FILE_NOTE, SYSTEM_PROMPT
from .results import Attempt, Scorecard, by_level, summarise

MODELS = ["gpt-5-mini"]

# Tasks in flight per model. Well under every provider's limit, and the run is
# long enough that a 429 storm costs more time than the concurrency saves.
CONCURRENCY = 8

# GAIA's annotators took up to a few dozen steps on level 3; a model that has
# not answered in this many turns is usually searching in circles.
MAX_STEPS = 20


@functools.wraps(search)
async def _search(*args, **kwargs) -> str:
    """``search`` off the event loop: Tavily's client blocks, and tasks run side by side."""
    return await asyncio.to_thread(search, *args, **kwargs)


def build_agent(model: str) -> Agent:
    """The agent under test, answering in GAIA's reply shape."""
    return Agent(
        model=LlmClient(model=model),
        tools=[tool(_search), tool(calculate)],
        instructions=SYSTEM_PROMPT,
        max_steps=MAX_STEPS,
        structured_output=GaiaReply,
    )


def question(task: GaiaTask) -> str:
    """The question as the agent reads it, with a note on any file it cannot open."""
    if task.file_name:
        return task.question + FILE_NOTE.format(file_name=task.file_name)
    return task.question


async def attempt(agent: Agent, task: GaiaTask, gate: asyncio.Semaphore) -> Attempt:
    """Run the agent on one question, and keep whatever came back.

    Failures are recorded rather than raised: on a run of this length something
    always times out, and losing the other results to it would be a poor trade.
    The attempt still counts as a miss.
    """
    started = time.monotonic()
    context = ExecutionContext()
    reply: GaiaReply | None = None
    error: str | None = None
    try:
        async with gate:
            result = await agent.run(question(task), context)
        if result is None:
            raise TimeoutError(f"no answer within {agent.max_steps} steps")
        if not isinstance(result, GaiaReply):
            raise TypeError(f"expected a GaiaReply, got {type(result).__name__}")
        reply = result
    except Exception as exception:  # noqa: BLE001 - one bad run is one bad row
        error = f"{type(exception).__name__}: {exception}"
    return Attempt(
        model=agent.model.model,
        task_id=task.task_id,
        level=task.level,
        question=task.question,
        expected=task.final_answer,
        file_name=task.file_name,
        reply=reply,
        error=error,
        seconds=time.monotonic() - started,
        steps=context.current_step,
    )


async def run_model(model: str, tasks: Sequence[GaiaTask]) -> list[Attempt]:
    """Put every task to one model's agent, at most :data:`CONCURRENCY` at a time."""
    agent = build_agent(model)
    gate = asyncio.Semaphore(CONCURRENCY)
    return await tqdm.gather(
        *(attempt(agent, task, gate) for task in tasks), desc=f"{model:<28}", unit="q"
    )


async def run(
    models: Sequence[str], tasks: Sequence[GaiaTask]
) -> dict[str, list[Attempt]]:
    """Run each model's agent over the whole task list, one model after another."""
    return {model: await run_model(model, tasks) for model in models}


def report(results: dict[str, list[Attempt]]) -> None:
    """Print the scoreboard: accuracy overall and per level, then the misses."""
    levels = sorted(
        {attempt.level for attempts in results.values() for attempt in attempts}
    )
    header = (
        f"{'model':<28}{'n':>5}{'all':>8}"
        + "".join(f"{f'L{level}':>8}" for level in levels)
        + f"{'declined':>10}{'errors':>8}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for model, attempts in results.items():
        overall = summarise(attempts)
        cards = by_level(attempts)
        rates = "".join(f"{_rate(cards.get(level)):>8}" for level in levels)
        print(
            f"{model:<28}{overall.tasks:>5}{overall.accuracy:>8.2f}{rates}"
            f"{overall.declined:>10}{overall.errors:>8}"
        )

    for model, attempts in results.items():
        failed = [one for one in attempts if one.error]
        if failed:
            print(f"\n{len(failed)} runs failed on {model}:")
            for one in failed[:5]:
                print(f"  {one.task_id:<38}{one.error}")


def _rate(card: Scorecard | None) -> str:
    """One accuracy cell, blank when the level was not in the run."""
    return f"{card.accuracy:.2f}" if card else "-"


def write(results: dict[str, list[Attempt]], path: Path) -> None:
    """Dump every attempt as JSONL, scored, for reading back later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as lines:
        for attempts in results.values():
            for one in attempts:
                row = one.model_dump(mode="json") | {
                    "answer": one.answer,
                    "correct": one.correct,
                    "declined": one.declined,
                }
                lines.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {sum(map(len, results.values()))} attempts to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        metavar="MODEL",
        help=f"litellm model names, run one after another (default: {' '.join(MODELS)})",
    )
    parser.add_argument(
        "--split",
        type=Split,
        choices=list(Split),
        default=Split.VALIDATION,
        help="only validation can be scored; test keeps its answers private",
    )
    parser.add_argument(
        "--level",
        type=lambda value: Level(int(value)),
        choices=list(Level),
        help="only this level's tasks (default: all three)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        metavar="TASK_ID",
        help="only these tasks, in this order",
    )
    parser.add_argument(
        "--limit", type=int, help="only the first N tasks, for a dry run"
    )
    parser.add_argument(
        "--out", type=Path, help="write every attempt to this JSONL file"
    )
    return parser.parse_args()


def main() -> None:
    """Load the split, run the agent over it, print the scoreboard."""
    args = parse_args()
    if args.split is Split.TEST:
        raise SystemExit("the test split keeps its answers private — nothing to score")

    load_dotenv()
    dataset = GaiaDataset.from_hub(args.split, args.level)
    tasks = dataset.tasks
    if args.task_ids:
        try:
            tasks = tuple(dataset.get(task_id) for task_id in args.task_ids)
        except KeyError as error:
            raise SystemExit(error.args[0]) from None
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} {args.split} tasks, {len(args.models)} models\n")

    results = asyncio.run(run(args.models, tasks))
    report(results)
    if args.out:
        write(results, args.out)


if __name__ == "__main__":
    main()
