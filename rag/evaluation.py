"""Score what the vector RAG chat retrieves, against a labelled question set.

Retrieval is the half of RAG that can be measured for free. Nothing is
generated, no model is called, and the same question returns the same chunks
every time — so the numbers here are exact rather than sampled, and a sweep
over ``k`` costs seconds instead of an API bill. This module answers one
question, *did the right documents come back?*; :mod:`rag.judge` answers the
other one, whether the model then used them.

The labelled set is ``rag/eval_cases.jsonl``: one JSON object per line, read
into :class:`Case`. Each carries the question, any prior user turns, the
documents that must be retrieved to answer it, and a short reference answer
that only the judge uses.

Labels name *documents* (``company/Setup Standard``) and not chunk ids, because
chunk ids are an artefact of ``CHUNK_SIZE`` — they change on every rebuild,
and ground truth that rots whenever a knob moves cannot be used to test that
knob.

Four numbers come out, measured at deliberately different granularities:

``hit``
    Did any labelled document come back at all? The coarsest of the four, and
    the one that separates broken retrieval from merely imprecise retrieval.
``recall``
    What fraction of the labelled documents came back. A ``cross`` case
    labelled with two documents scores 0.5 when only one arrives — which is
    exactly the failure that a hit-only view would call a success.
``mrr``
    One over the rank of the first chunk from a labelled document, counting
    from one: first is 1.0, fifth is 0.2, absent is 0.0. Rank is counted over
    chunks rather than documents because chunks are the order the model reads.
``precision``
    The fraction of retrieved *chunks* that came from a labelled document.
    Chunk-level on purpose: for one question the context window is the whole
    of what the model knows, so precision measures how much of that was noise.
    Counting deduplicated documents instead would penalise three chunks from
    the one right document, which is a good outcome and not a bad one.

``refusal`` cases carry no labels, because nothing in the knowledge base
answers them. None of the four are defined without labels, so :func:`score`
refuses to score one and :func:`evaluate` passes over them. What they retrieve
is still worth looking at — they return a full set of chunks like every other
question, confidently and irrelevantly — so :func:`retrieve` takes them
happily. Whether the bot then declines is the judge's question, not this
module's.

Score the current settings with ``uv run python -m rag.evaluation``; build the
store first if it does not exist yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag.knowledge_base import KNOWLEDGE_BASE
from rag.vector_chat import Bot

EVAL_CASES = Path(__file__).parent / "eval_cases.jsonl"

# What a case is testing, which is worth keeping separate in the results: the
# kinds are not equally hard, and averaging them together hides the failures.
# ``single`` needs one document; ``followup`` drops its subject into the
# history and can only be answered by searching the conversation; ``cross``
# needs two documents at once; ``refusal`` has no answer in the knowledge base.
Kind = Literal["single", "followup", "cross", "refusal"]


class Case(BaseModel):
    """One labelled question from ``eval_cases.jsonl``."""

    id: str
    kind: Kind
    question: str
    history: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    answer: str

    @property
    def turns(self) -> list[dict[str, str]]:
        """``history`` as the Gradio-style turns :meth:`Bot.retrieve` expects.

        User turns only, and that is not a shortcut: ``Bot.retrieve`` reads
        nothing else out of the history. So a multi-turn case can be retrieved
        for exactly as it would be in the app without generating the assistant
        replies in between — which is what keeps the whole of this module free
        to run. The judge, which does need real replies, builds its own.
        """
        return [{"role": "user", "content": question} for question in self.history]


class Retrieval(BaseModel):
    """One case's retrieved chunks, scored against its labels."""

    id: str
    kind: Kind
    # One document name per retrieved chunk, in rank order, duplicates kept.
    retrieved: list[str]
    hit: bool
    recall: float
    precision: float
    mrr: float
    # Labelled documents that did not come back — the readable half of a low
    # recall score, and where a sweep gets its next hypothesis from.
    missing: list[str]


class Summary(BaseModel):
    """The four numbers averaged over a set of scored cases."""

    cases: int
    hit_rate: float
    recall: float
    precision: float
    mrr: float


def load_cases(path: Path = EVAL_CASES, root: Path = KNOWLEDGE_BASE) -> list[Case]:
    """Read the case file, checking the labels name documents that exist.

    A mistyped label — ``company/Returns policy`` for ``company/Returns
    Policy`` — is invisible everywhere downstream: the case simply scores zero
    recall for ever, and reads as a retrieval failure rather than as the typo
    it is. Catching it here costs one glob. Ids are checked for uniqueness for
    the same reason, since every table below joins on them.
    """
    cases = [
        Case.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"no cases in {path}")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError(f"duplicate case ids in {path}")
    documents = {f"{file.parent.name}/{file.stem}" for file in root.glob("*/*.md")}
    unknown = sorted({source for case in cases for source in case.sources} - documents)
    if unknown:
        raise ValueError(
            f"labels naming no document under {root}: {', '.join(unknown)}"
        )
    return cases


def retrieve(bot: Bot, case: Case) -> list[str]:
    """Retrieve for ``case`` and name the document each chunk came from.

    Names come back one per chunk and in rank order, duplicates and all — three
    chunks from ``company/Shipping`` appear three times. :func:`score`
    collapses that where it needs to, and ``mrr`` and ``precision`` need it
    kept.

    This goes through :meth:`Bot.retrieve`, not through ``bot.retriever``. The
    difference is the conversation-wide second search, which for a ``followup``
    case is the entire reason the right document comes back; scoring the bare
    retriever would measure a system nobody runs.
    """
    documents = bot.retrieve(case.question, case.turns)
    return [document.metadata.get("name", "unknown") for document in documents]


def score(case: Case, retrieved: Sequence[str]) -> Retrieval:
    """Score one case's retrieved document names against its labels.

    ``retrieved`` is what :func:`retrieve` returned — one name per chunk, in
    rank order. An empty ``retrieved`` scores zero on all four rather than
    dividing by zero.
    """
    if not case.sources:
        raise ValueError(
            f"{case.id} is a {case.kind} case and carries no labels, so there is "
            f"nothing to score against; retrieval metrics are undefined for it"
        )
    gold = set(case.sources)
    found = gold.intersection(retrieved)
    rank = next((i for i, name in enumerate(retrieved, start=1) if name in gold), 0)
    return Retrieval(
        id=case.id,
        kind=case.kind,
        retrieved=list(retrieved),
        hit=bool(found),
        recall=len(found) / len(gold),
        precision=(
            sum(name in gold for name in retrieved) / len(retrieved)
            if retrieved
            else 0.0
        ),
        mrr=1 / rank if rank else 0.0,
        missing=sorted(gold - found),
    )


def evaluate(bot: Bot, cases: Iterable[Case]) -> list[Retrieval]:
    """Retrieve and score every labelled case, passing over the unlabelled ones.

    ``bot`` carries the settings under test: ``Bot(retriever=open_retriever(8),
    k_conversation=6)`` scores that pair, and sweeping is a loop over these.
    """
    return [score(case, retrieve(bot, case)) for case in cases if case.sources]


def summarise(scores: Sequence[Retrieval]) -> Summary:
    """Average the four numbers over ``scores``.

    Every case weighs the same, whatever its kind and however many documents it
    is labelled with. That is a choice, and it is why :func:`by_kind` exists:
    the set is deliberately not balanced, so the overall mean is a headline
    rather than a verdict.
    """
    if not scores:
        raise ValueError("no scored cases to summarise")
    total = len(scores)
    return Summary(
        cases=total,
        hit_rate=sum(result.hit for result in scores) / total,
        recall=sum(result.recall for result in scores) / total,
        precision=sum(result.precision for result in scores) / total,
        mrr=sum(result.mrr for result in scores) / total,
    )


def by_kind(scores: Sequence[Retrieval]) -> dict[str, Summary]:
    """Summarise per kind, in the order the kinds first appear.

    This is where the useful reading is. ``single`` cases are the easy ones and
    there are more of them than anything else, so a respectable overall mean
    sits happily on top of ``followup`` cases that fail outright — and it is
    the follow-ups that the two-search merge in :meth:`Bot.retrieve` exists to
    fix, so they are the ones that say whether it works.
    """
    kinds = dict.fromkeys(result.kind for result in scores)
    return {
        kind: summarise([result for result in scores if result.kind == kind])
        for kind in kinds
    }


def main() -> None:
    """Score the default settings over the whole case set and print the tables.

    Constructing ``Bot()`` reads ``OPENROUTER_API_KEY`` because it builds the
    LLM as it goes, but nothing below ever calls it: this is retrieval only,
    and it costs nothing to run.
    """
    cases = load_cases()
    scores = evaluate(Bot(), cases)
    unlabelled = len(cases) - len(scores)
    print(f"{len(scores)} labelled cases scored, {unlabelled} refusal cases skipped\n")

    columns = f"{'kind':<10}{'n':>4}{'hit':>8}{'recall':>8}{'prec':>8}{'mrr':>8}"
    print(columns)
    print("-" * len(columns))
    for kind, summary in by_kind(scores).items():
        print(
            f"{kind:<10}{summary.cases:>4}{summary.hit_rate:>8.2f}"
            f"{summary.recall:>8.2f}{summary.precision:>8.2f}{summary.mrr:>8.2f}"
        )
    overall = summarise(scores)
    print("-" * len(columns))
    print(
        f"{'all':<10}{overall.cases:>4}{overall.hit_rate:>8.2f}"
        f"{overall.recall:>8.2f}{overall.precision:>8.2f}{overall.mrr:>8.2f}"
    )

    incomplete = [result for result in scores if result.recall < 1]
    if incomplete:
        print(f"\n{len(incomplete)} cases did not retrieve every labelled document:")
        for result in incomplete:
            print(f"  {result.id:<34}missing {', '.join(result.missing)}")


if __name__ == "__main__":
    main()
