"""A no-tools baseline on GAIA: ask a bare LLM the questions, score its final answers.

The number the agent has to beat. Every task goes to the model as plain text —
no search, no code, no file reading — so what this measures is how far the
weights alone get you, which on GAIA is not very far and is meant not to be.
Level 1 questions are answerable by a strong model that happens to remember the
fact; levels 2 and 3 chain lookups the model cannot make, and the 38 validation
tasks that ship an attached file are unanswerable here by construction. Those
are told about the file they cannot open (:data:`prompts.FILE_NOTE`) so that
declining is available to them, and a decline still scores zero — the point of
the row is what the agent will rescue.

Models run one at a time, their tasks concurrently, which keeps the progress bar
readable and one provider's rate limit away from another's. Run it with::

    uv run python -m agent_from_scratch.gaia_eval.baseline --limit 10
    uv run python -m agent_from_scratch.gaia_eval.baseline --out runs/baseline.jsonl

This one spends money: one call per task per model.
"""

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from litellm import acompletion
from tqdm.asyncio import tqdm

from .constants import Level, Split
from .dataset import GaiaDataset
from .models import GaiaReply, GaiaTask
from .prompts import FILE_NOTE, SYSTEM_PROMPT
from .results import Attempt, Scorecard, by_level, summarise

MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5",
]

# In-flight calls per model. Well under every provider's limit, and the run is
# long enough that a 429 storm costs more time than the concurrency saves.
CONCURRENCY = 8

# A reasoning model on a level 3 question thinks for minutes, so the timeout is
# generous; retries cover the transient failures that a short one would mask.
TIMEOUT = 600.0
RETRIES = 3


def messages(task: GaiaTask) -> list[dict[str, str]]:
    """The two messages one task becomes: GAIA's format rules, then the question."""
    question = task.question
    if task.file_name:
        question += FILE_NOTE.format(file_name=task.file_name)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


async def attempt(model: str, task: GaiaTask, gate: asyncio.Semaphore) -> Attempt:
    """Ask one model one question, and keep whatever came back.

    Failures are recorded rather than raised: on a run of this length something
    always times out, and losing the other 164 results to it would be a poor
    trade. The attempt still counts as a miss.
    """
    started = time.monotonic()
    reply: GaiaReply | None = None
    error: str | None = None
    try:
        async with gate:
            response = await acompletion(
                model=model,
                messages=messages(task),
                response_format=GaiaReply,
                num_retries=RETRIES,
                timeout=TIMEOUT,
            )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("model returned an empty reply")
        reply = GaiaReply.model_validate_json(content)
    except Exception as exception:  # noqa: BLE001 - one bad call is one bad row
        error = f"{type(exception).__name__}: {exception}"
    return Attempt(
        model=model,
        task_id=task.task_id,
        level=task.level,
        question=task.question,
        expected=task.final_answer,
        file_name=task.file_name,
        reply=reply,
        error=error,
        seconds=time.monotonic() - started,
    )


async def run_model(model: str, tasks: Sequence[GaiaTask]) -> list[Attempt]:
    """Put every task to one model, at most :data:`CONCURRENCY` at a time."""
    gate = asyncio.Semaphore(CONCURRENCY)
    return await tqdm.gather(
        *(attempt(model, task, gate) for task in tasks), desc=f"{model:<28}", unit="q"
    )


async def run(
    models: Sequence[str], tasks: Sequence[GaiaTask]
) -> dict[str, list[Attempt]]:
    """Run each model over the whole task list, one model after another."""
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
            print(f"\n{len(failed)} calls failed on {model}:")
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
    parser.add_argument("--models", nargs="+", default=MODELS, metavar="MODEL")
    parser.add_argument(
        "--split", type=Split, choices=list(Split), default=Split.VALIDATION
    )
    parser.add_argument(
        "--level", type=lambda value: Level(int(value)), choices=list(Level)
    )
    parser.add_argument(
        "--limit", type=int, help="only the first N tasks, for a dry run"
    )
    parser.add_argument(
        "--out", type=Path, help="write every attempt to this JSONL file"
    )
    return parser.parse_args()


def main() -> None:
    """Load the split, run the models over it, print the scoreboard."""
    args = parse_args()
    if args.split is Split.TEST:
        raise SystemExit("the test split keeps its answers private — nothing to score")

    load_dotenv()
    dataset = GaiaDataset.from_hub(args.split, args.level)
    tasks = dataset.tasks[: args.limit] if args.limit else dataset.tasks
    print(f"{len(tasks)} {args.split} tasks, {len(args.models)} models\n")

    results = asyncio.run(run(args.models, tasks))
    report(results)
    if args.out:
        write(results, args.out)


if __name__ == "__main__":
    main()
