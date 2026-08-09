"""Grade the chat's answers with a second model, against the labelled set.

:mod:`rag.evaluation` measures whether the right chunks came back. This module
measures the other half — whether the model then used them — and it cannot be
done with set arithmetic, because there is no string comparison that separates
a right answer from a right answer worded differently. So a model reads the
question, the reference answer, the retrieved context, and the reply, and
returns four booleans.

Three things make that grading worth trusting rather than merely automatic:

*The judge is a different model from the answerer.* A model marking its own
homework grades generously, and the failure is invisible — the scores just come
out high. ``JUDGE_MODEL`` is deliberately from another family than
:data:`rag.vector_chat.MODEL`.

*Correct and grounded are separate questions.* ``correct`` compares the reply
against the reference answer; ``grounded`` asks whether the retrieved context
actually supports what the reply claims. The pair is the interesting output,
because the two disagree in both directions and each disagreement means
something specific:

    correct + grounded      working as intended.
    correct + ungrounded    the model knew it from pretraining and answered
                            around the context. The retrieval is not doing the
                            work, and the day the knowledge base contradicts
                            the model's memory, this case turns into a
                            confident wrong answer.
    incorrect + grounded    retrieval fetched the wrong passage and the model
                            faithfully read it back.
    incorrect + ungrounded  invention.

Measuring only correctness collapses all four into one number and quietly
passes the second row.

*Refusal cases invert the test.* A case with no labels has no answer in the
knowledge base, so declining is the pass and answering is the failure —
:attr:`Graded.passed` reads ``refused`` for those and ``correct and grounded``
for the rest.

Multi-turn cases are run forward rather than faked: each prior question goes
through the bot for real, so the history holds replies the model actually gave,
footers and all, exactly as ``gr.ChatInterface`` would have stored them.

Unlike :mod:`rag.evaluation`, this costs tokens — roughly two calls per case
plus one per prior turn, and it is not deterministic, so two runs of the same
set will not agree perfectly. Run it with ``uv run python -m rag.judge``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag.evaluation import Case, Kind, load_cases
from rag.vector_chat import SOURCES_PREFIX, Bot, open_llm

# Gemini grading Claude. Any OpenRouter model works here; what matters is that
# it is not the model under test.
JUDGE_MODEL = "google/gemini-2.5-pro"

JUDGE_SYSTEM = (
    "You are grading a retrieval-augmented chatbot that answers questions about "
    "Fretwork, an online guitar store, strictly from context retrieved for each "
    "question. You will be shown the conversation so far, the question, a "
    "reference answer written by hand from the source documents, the context the "
    "bot retrieved, and the bot's reply. Grade the reply on four counts.\n\n"
    "correct — does the reply agree with the reference answer on the facts that "
    "matter? Wording, length, and extra detail that is also true do not count "
    "against it; a wrong or invented figure, name, or date does. Some reference "
    "answers describe an expected refusal rather than a fact ('Should decline. "
    "...'); for those, correct means the reply did decline and did not invent "
    "the missing fact.\n\n"
    "grounded — is every factual claim in the reply supported by the context "
    "shown? Judge this against the context alone, never against what you happen "
    "to know about guitars or retailers: a claim that is true in the world but "
    "absent from the context is not grounded. A reply that makes no factual "
    "claims, such as a refusal, is grounded.\n\n"
    "refused — did the reply decline to answer, or say the context does not "
    "contain what was asked? A reply that answers and then notes a caveat has "
    "not refused.\n\n"
    "reason — one sentence, naming the specific claim or figure that decided it. "
    "Write it so someone reading only this line knows what to go and look at."
)

JUDGE_TEMPLATE = (
    "## Conversation so far\n\n{history}\n\n"
    "## Question\n\n{question}\n\n"
    "## Reference answer\n\n{reference}\n\n"
    "## Context the bot retrieved\n\n{context}\n\n"
    "## The bot's reply\n\n{reply}"
)


class Judgement(BaseModel):
    """One judge's verdict on one reply."""

    correct: bool = Field(description="Agrees with the reference answer on the facts.")
    grounded: bool = Field(
        description="Every factual claim is supported by the retrieved context."
    )
    refused: bool = Field(
        description="Declined to answer, or said the context does not cover it."
    )
    reason: str = Field(description="One sentence naming what decided the verdict.")


class Answered(BaseModel):
    """What the bot actually said for one case, and what it was reading."""

    id: str
    kind: Kind
    question: str
    history: list[str]
    # The reply with its sources footer stripped: the footer is an artefact of
    # the UI, and the model never sees its own, so it is not graded either.
    reply: str
    # The retrieved chunks, labelled and laid out as the bot's system prompt
    # laid them out — the judge has to grade groundedness against the same
    # text the bot was reading, not a tidier version of it.
    context: list[str]
    sources: list[str]


class Graded(BaseModel):
    """One case, answered and judged, flattened for tabulating."""

    id: str
    kind: Kind
    question: str
    reply: str
    sources: list[str]
    correct: bool
    grounded: bool
    refused: bool
    reason: str

    @property
    def passed(self) -> bool:
        """Did this case come out the way it should have?

        Refusal cases pass by declining. Everything else has to be both right
        and grounded — correct alone would pass a model answering from its own
        memory, which is the one thing a RAG system is not supposed to do.
        """
        if self.kind == "refusal":
            return self.refused
        return self.correct and self.grounded


class Scorecard(BaseModel):
    """Rates over a set of graded cases."""

    cases: int
    passed: float
    correct: float
    grounded: float
    refused: float


def _last(stream: Iterator[str]) -> str:
    """Drain a ``Bot.chat`` stream and return the final, complete reply.

    ``chat`` yields the reply accumulated so far on every token, so the last
    thing it yields is the whole of it.
    """
    reply = ""
    for reply in stream:
        pass
    return reply


def converse(bot: Bot, case: Case) -> tuple[list[dict[str, Any]], str]:
    """Replay the case's prior questions through the bot, then answer it.

    Returns the history that built up and the reply to ``case.question``.

    The prior turns are answered for real rather than stubbed, because a
    ``followup`` case is testing what happens when the subject is two turns
    back — and a hand-written stand-in for the assistant's reply would be
    testing a conversation that never happened. Replies go into the history
    with their sources footers still attached, which is what
    ``gr.ChatInterface`` stores and therefore what ``Bot`` has to cope with.
    """
    history: list[dict[str, Any]] = []
    for question in case.history:
        reply = _last(bot.chat(question, history))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply})
    return history, _last(bot.chat(case.question, history))


def answer(bot: Bot, case: Case) -> Answered:
    """Run one case through the bot and keep what the judge will need.

    Retrieval runs a second time here to recover the chunks ``chat`` used
    internally. That is free and returns the same chunks — no model is
    involved and the store does not change between the two calls — and it
    keeps this module to ``Bot``'s public surface.
    """
    history, reply = converse(bot, case)
    documents = bot.retrieve(case.question, history)
    return Answered(
        id=case.id,
        kind=case.kind,
        question=case.question,
        history=case.history,
        reply=reply.split(SOURCES_PREFIX)[0],
        context=[_block(document) for document in documents],
        sources=list(dict.fromkeys(_name(document) for document in documents)),
    )


def _name(document: Document) -> str:
    """The document a chunk came from, as the knowledge base labels it."""
    return str(document.metadata.get("name", "unknown"))


def _block(document: Document) -> str:
    """One chunk, headed by its document name, as the system prompt shows it."""
    return f"# {_name(document)}\n\n{document.page_content.strip()}"


def judge(llm: ChatOpenAI, case: Case, answered: Answered) -> Judgement:
    """Grade one answered case, returning the verdict as a validated model.

    ``with_structured_output`` binds :class:`Judgement` as a tool schema, so
    the four fields come back typed and the model retries rather than
    returning prose that would have to be parsed.
    """
    prompt = JUDGE_TEMPLATE.format(
        history="\n".join(answered.history) or "(nothing — this is the first turn)",
        question=case.question,
        reference=case.answer,
        context="\n\n".join(answered.context) or "(nothing was retrieved)",
        reply=answered.reply,
    )
    verdict = llm.with_structured_output(Judgement).invoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
    )
    if not isinstance(verdict, Judgement):
        raise TypeError(f"judge returned {type(verdict).__name__}, not a Judgement")
    return verdict


def grade(bot: Bot, llm: ChatOpenAI, case: Case) -> Graded:
    """Answer one case and judge the reply."""
    answered = answer(bot, case)
    verdict = judge(llm, case, answered)
    return Graded(
        id=case.id,
        kind=case.kind,
        question=case.question,
        reply=answered.reply,
        sources=answered.sources,
        **verdict.model_dump(),
    )


def run(bot: Bot, llm: ChatOpenAI, cases: Sequence[Case]) -> list[Graded]:
    """Answer and judge every case, in order.

    Sequential on purpose: the set is small, the whole run is a few minutes,
    and one case failing is easier to read when it fails on its own line.
    """
    return [grade(bot, llm, case) for case in cases]


def summarise(graded: Sequence[Graded]) -> Scorecard:
    """Rates over ``graded`` — every case weighing the same."""
    if not graded:
        raise ValueError("no graded cases to summarise")
    total = len(graded)
    return Scorecard(
        cases=total,
        passed=sum(result.passed for result in graded) / total,
        correct=sum(result.correct for result in graded) / total,
        grounded=sum(result.grounded for result in graded) / total,
        refused=sum(result.refused for result in graded) / total,
    )


def by_kind(graded: Sequence[Graded]) -> dict[str, Scorecard]:
    """Summarise per kind, in the order the kinds first appear.

    ``refused`` reads in opposite directions down this table: near 1.0 is the
    goal on the ``refusal`` row and a failure on every other one.
    """
    kinds = dict.fromkeys(result.kind for result in graded)
    return {
        kind: summarise([result for result in graded if result.kind == kind])
        for kind in kinds
    }


def main() -> None:
    """Answer and judge the whole case set, then print the scorecard.

    This one spends money: two model calls per case, plus one for each prior
    turn a multi-turn case replays.
    """
    cases = load_cases()
    graded = run(Bot(), open_llm(JUDGE_MODEL), cases)
    print(f"{len(graded)} cases answered by Bot and judged by {JUDGE_MODEL}\n")

    columns = (
        f"{'kind':<10}{'n':>4}{'passed':>9}{'correct':>9}{'grounded':>10}{'refused':>9}"
    )
    print(columns)
    print("-" * len(columns))
    for kind, card in by_kind(graded).items():
        print(
            f"{kind:<10}{card.cases:>4}{card.passed:>9.2f}{card.correct:>9.2f}"
            f"{card.grounded:>10.2f}{card.refused:>9.2f}"
        )
    overall = summarise(graded)
    print("-" * len(columns))
    print(
        f"{'all':<10}{overall.cases:>4}{overall.passed:>9.2f}{overall.correct:>9.2f}"
        f"{overall.grounded:>10.2f}{overall.refused:>9.2f}"
    )

    ungrounded = [result for result in graded if result.correct and not result.grounded]
    if ungrounded:
        print(
            f"\n{len(ungrounded)} right but not grounded — answered around the context:"
        )
        for result in ungrounded:
            print(f"  {result.id:<34}{result.reason}")

    failures = [result for result in graded if not result.passed]
    if failures:
        print(f"\n{len(failures)} failed:")
        for result in failures:
            print(f"  {result.id:<34}{result.reason}")


if __name__ == "__main__":
    main()
