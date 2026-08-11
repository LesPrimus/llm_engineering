"""A dashboard over the two RAG evaluations, in Gradio.

:mod:`rag.evaluation` and :mod:`rag.judge` both print a table and exit. That is
the right shape for a CLI and the wrong one for the question they exist to
answer, which is never "what does this number say?" but "what happens to it when
I turn this knob?". Comparing two settings from a terminal means running twice
and diffing by eye; the interesting comparison — the whole curve of a metric
against ``k`` — means running ten times and building the picture in your head.

So this module puts the two of them behind three tabs:

**Retrieval**, which is free. Nothing is generated and the same question returns
the same chunks every time, so ``k`` and ``k_conversation`` are sliders, the
button can be pressed as often as you like, and a sweep over every ``k`` in a
range costs seconds rather than an API bill. The sweep is the tab's reason for
existing: recall climbing to a plateau while precision falls away is the trade
:mod:`rag.evaluation` describes in prose, drawn.

**Answers**, which is not free. Every case costs two model calls plus one for
each prior turn it replays, so the kinds to grade are chosen up front and the
button says what the run will cost before it runs. Rows land one case at a time
rather than all at the end, because a run is minutes long and the failures are
worth reading as they arrive.

**Cases**, which is the labelled set itself. Neither scoreboard means much
without the questions behind it to hand.

The expensive things are built once, in :class:`Evaluation`, and shared. That
matters more than it looks: ``open_retriever(k)`` reopens the store on every
call, and the store loads a sentence-transformers model as it opens — fine once
from a CLI, twelve times over for a sweep. Opening the store once and deriving a
retriever per ``k`` from it is the difference between a sweep that takes seconds
and one that takes a minute.

Build the store first, then run the dashboard::

    uv run python -m rag.vector_store
    uv run python -m rag.eval_app
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import plotly.graph_objects as go  # type: ignore[import-untyped]
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from rag.evaluation import Case, Retrieval, Summary, evaluate, load_cases
from rag.evaluation import by_kind as retrieval_by_kind
from rag.evaluation import summarise as retrieval_summarise
from rag.judge import JUDGE_MODEL, Graded, Scorecard, grade
from rag.judge import by_kind as answer_by_kind
from rag.judge import summarise as answer_summarise
from rag.vector_chat import K, K_CONVERSATION, MODEL, Bot, open_llm
from rag.vector_store import open_store

# How far the sliders and the sweep go. Twelve is past the point where more
# chunks stop adding documents and only add noise, which is the part of the
# curve worth being able to see.
K_MAX = 12

# The shallowest conversation-wide search the slider offers, and not zero:
# Chroma refuses a search for zero results, and ``Bot.retrieve`` runs the second
# search unconditionally once a case has any history. So turning the second
# search off is not a setting this dashboard can reach — 1 is as close as it
# gets, and it still merges in whatever the conversation's nearest chunk is.
K_CONVERSATION_MIN = 1

# Metric display name to the :class:`Summary` field behind it, in the order the
# categorical colour slots below are assigned.
METRICS: dict[str, str] = {
    "hit": "hit_rate",
    "recall": "recall",
    "precision": "precision",
    "mrr": "mrr",
    "ndcg": "ndcg",
}

# Slots 1-5 of a categorical palette, assigned in fixed order — a metric keeps
# its colour whatever else is on the chart. Validated on the light chart surface:
# every adjacent pair clears the colour-blind separation floor (worst 9.1) and
# the normal-vision floor (worst 19.6). Three of the five sit under 3:1 contrast
# there, which is why the sweep ships its numbers as a table underneath the chart
# and not only as lines. A Gradio app follows the viewer's system theme and a
# figure carries one palette, not two, so these are the light steps; on a dark
# surface they keep their separation and their contrast, and only drift out of
# the lightness band the dark steps would be chosen for.
METRIC_COLORS: dict[str, str] = dict(
    zip(METRICS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], strict=True)
)

# Chart chrome, picked to work on either surface rather than to match one: this
# grey is the palette's muted ink, which is the same value in both modes, and the
# gridlines are it again at low alpha. The plot itself is transparent, so the
# Gradio card behind it supplies the background and the chart follows the app's
# theme without carrying two palettes.
AXIS_INK = "#898781"
GRID_INK = "rgba(137, 135, 129, 0.25)"

RETRIEVAL_SUMMARY_HEADERS = ["kind", "cases", *METRICS]
RETRIEVAL_CASE_HEADERS = ["id", "kind", *METRICS, "missing"]
ANSWER_SUMMARY_HEADERS = ["kind", "cases", "passed", "correct", "grounded", "refused"]
ANSWER_CASE_HEADERS = [
    "id",
    "kind",
    "passed",
    "correct",
    "grounded",
    "refused",
    "reason",
]
CASE_HEADERS = ["id", "kind", "question", "history", "sources", "reference answer"]

TICK = "✓"
CROSS = "✗"


@dataclass(frozen=True)
class Evaluation:
    """The store, the cases, and the two models — built once, scored many times.

    Every field is expensive to construct and none of them change between runs:
    opening the store loads the embedding model, and each ``ChatOpenAI`` reads
    ``.env`` as it is built. Holding them here is what lets :meth:`sweep` score
    a dozen settings without paying for any of it twice.

    ``llm`` is the model under test and ``judge_llm`` grades it. They are
    deliberately different models — a model marking its own homework grades
    generously, and the failure is invisible in the scores.
    """

    store: Chroma = field(default_factory=open_store)
    cases: list[Case] = field(default_factory=load_cases)
    llm: ChatOpenAI = field(default_factory=open_llm)
    judge_llm: ChatOpenAI = field(default_factory=lambda: open_llm(JUDGE_MODEL))

    def bot(self, k: int, k_conversation: int) -> Bot:
        """A bot at these settings, over the already-open store.

        The retriever is derived from ``self.store`` rather than built by
        ``open_retriever``, which would reopen the store — and reload the
        embedding model — on every call.
        """
        return Bot(
            retriever=self.store.as_retriever(search_kwargs={"k": k}),
            llm=self.llm,
            k_conversation=k_conversation,
        )

    def score(self, k: int, k_conversation: int) -> list[Retrieval]:
        """Score retrieval at these settings over every labelled case."""
        return evaluate(self.bot(k, k_conversation), self.cases)

    def sweep(self, k_max: int, k_conversation: int) -> dict[int, Summary]:
        """Score every ``k`` from 1 to ``k_max``, holding ``k_conversation``.

        One knob moves at a time on purpose: sweeping both together produces a
        surface that is harder to read than the two curves it contains.
        """
        return {
            k: retrieval_summarise(self.score(k, k_conversation))
            for k in range(1, k_max + 1)
        }

    def selected(self, kinds: Sequence[str]) -> list[Case]:
        """The cases of the chosen kinds, in file order."""
        return [case for case in self.cases if case.kind in kinds]

    def graded(
        self, kinds: Sequence[str], k: int, k_conversation: int
    ) -> Iterator[Graded]:
        """Answer and judge the chosen cases, yielding each verdict as it lands.

        A case that raises comes back as a failed row naming the error rather
        than as an exception. Losing one case to a judge timeout is a bad
        outcome; losing the twenty already paid for behind it is a worse one,
        and a run this expensive should not be all-or-nothing.
        """
        bot = self.bot(k, k_conversation)
        for case in self.selected(kinds):
            try:
                yield grade(bot, self.judge_llm, case)
            except Exception as exc:
                yield Graded(
                    id=case.id,
                    kind=case.kind,
                    question=case.question,
                    reply="",
                    sources=[],
                    correct=False,
                    grounded=False,
                    refused=False,
                    reason=f"grading failed: {type(exc).__name__}: {exc}",
                )


def _mark(value: bool) -> str:
    """A boolean as a table cell."""
    return TICK if value else CROSS


def _retrieval_summary_rows(scores: Sequence[Retrieval]) -> list[list[Any]]:
    """The per-kind table, with the overall mean underneath it.

    Both halves are here because neither is the answer on its own: the kinds are
    not equally hard and the set is not balanced between them, so a respectable
    ``all`` row sits happily on top of a ``followup`` row that fails outright.
    """
    rows = [
        [
            kind,
            summary.cases,
            *(round(getattr(summary, f), 2) for f in METRICS.values()),
        ]
        for kind, summary in retrieval_by_kind(scores).items()
    ]
    overall = retrieval_summarise(scores)
    rows.append(
        [
            "all",
            overall.cases,
            *(round(getattr(overall, f), 2) for f in METRICS.values()),
        ]
    )
    return rows


def _retrieval_case_rows(scores: Sequence[Retrieval]) -> list[list[Any]]:
    """Every scored case, worst first — the failures are what this table is for."""
    ordered = sorted(scores, key=lambda result: (result.recall, result.ndcg, result.id))
    return [
        [
            result.id,
            result.kind,
            _mark(result.hit),
            round(result.recall, 2),
            round(result.precision, 2),
            round(result.mrr, 2),
            round(result.ndcg, 2),
            ", ".join(result.missing),
        ]
        for result in ordered
    ]


def _sweep_figure(sweep: dict[int, Summary], k_conversation: int) -> go.Figure:
    """Draw the five metrics against ``k``, one line each.

    Plotly rather than ``gr.LinePlot`` for one reason: ``k`` is a count, and a
    chart of it must tick at 1, 2, 3. Gradio's native plot picks its own tick
    positions from a continuous scale and lands on 1.4 and 1.8 — offering the
    reader settings that do not exist. ``dtick=1`` says otherwise.

    One y-axis, fixed to 0-1, because all five metrics already share that scale;
    a second axis would let two of them be drawn at incomparable heights.
    """
    figure = go.Figure()
    for metric, field_name in METRICS.items():
        figure.add_trace(
            go.Scatter(
                x=list(sweep),
                y=[getattr(summary, field_name) for summary in sweep.values()],
                name=metric,
                mode="lines+markers",
                line={"color": METRIC_COLORS[metric], "width": 2},
                marker={"size": 8},
                hovertemplate=f"{metric} %{{y:.2f}}<extra></extra>",
            )
        )
    figure.update_layout(
        # Transparent, so the Gradio card behind supplies the surface and the
        # chart sits correctly in either theme.
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": AXIS_INK, "size": 12},
        # Every line read at once, which is the comparison this chart is for.
        hovermode="x unified",
        # Below the plot and horizontal: five series need the width, and a
        # legend inside the plot area would sit on top of the lines it labels.
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
        },
        margin={"l": 50, "r": 20, "t": 20, "b": 70},
        height=420,
    )
    figure.update_xaxes(
        # The setting the sweep held fixed belongs on the chart, not only in the
        # label beside it: a screenshot of this outlives the widget above it.
        title={"text": f"k  (k_conversation held at {k_conversation})"},
        dtick=1,
        gridcolor=GRID_INK,
        linecolor=GRID_INK,
        zeroline=False,
    )
    figure.update_yaxes(
        title={"text": "score"},
        range=[0, 1.02],
        dtick=0.2,
        gridcolor=GRID_INK,
        linecolor=GRID_INK,
        zeroline=False,
    )
    return figure


def _sweep_rows(sweep: dict[int, Summary]) -> list[list[Any]]:
    """The same sweep as a table — the readable form, and the accessible one."""
    return [
        [
            k,
            *(
                round(getattr(summary, field_name), 2)
                for field_name in METRICS.values()
            ),
        ]
        for k, summary in sweep.items()
    ]


def _answer_summary_rows(graded: Sequence[Graded]) -> list[list[Any]]:
    """The judge's scorecard per kind, with the overall rates underneath.

    ``refused`` reads in opposite directions down this table: near 1.0 is the
    goal on the ``refusal`` row and a failure on every other one.

    Empty in, empty out — a run yields its first frame before it has graded
    anything, and rates over no cases are undefined rather than zero.
    """
    if not graded:
        return []

    def row(label: str, card: Scorecard) -> list[Any]:
        return [
            label,
            card.cases,
            round(card.passed, 2),
            round(card.correct, 2),
            round(card.grounded, 2),
            round(card.refused, 2),
        ]

    rows = [row(kind, card) for kind, card in answer_by_kind(graded).items()]
    rows.append(row("all", answer_summarise(graded)))
    return rows


def _answer_case_rows(graded: Sequence[Graded]) -> list[list[Any]]:
    """Every graded case, failures first, each with the line that decided it."""
    ordered = sorted(graded, key=lambda result: (result.passed, result.id))
    return [
        [
            result.id,
            result.kind,
            _mark(result.passed),
            _mark(result.correct),
            _mark(result.grounded),
            _mark(result.refused),
            result.reason,
        ]
        for result in ordered
    ]


def _answer_notes(graded: Sequence[Graded]) -> str:
    """The two lists worth acting on: answered around the context, and failed.

    Correct-but-ungrounded is the one that a pass rate hides. The model knew the
    answer from pretraining and did not need the context, so the retrieval is
    not doing the work — and the day the knowledge base contradicts the model's
    memory, that case turns into a confident wrong answer.
    """
    ungrounded = [result for result in graded if result.correct and not result.grounded]
    failed = [result for result in graded if not result.passed]
    sections = []
    if ungrounded:
        lines = "\n".join(f"- `{r.id}` — {r.reason}" for r in ungrounded)
        sections.append(
            f"**{len(ungrounded)} right but not grounded** — answered around the "
            f"context:\n{lines}"
        )
    if failed:
        lines = "\n".join(f"- `{r.id}` — {r.reason}" for r in failed)
        sections.append(f"**{len(failed)} failed:**\n{lines}")
    return "\n\n".join(sections) if sections else "_Nothing to flag._"


def _settings(k: int, k_conversation: int) -> str:
    """The settings a result was scored at, for stamping onto its label."""
    return f"k={k}, k_conversation={k_conversation}"


def _table(headers: Sequence[str], rows: list[list[Any]], label: str) -> gr.Dataframe:
    """A dataframe update that carries its own headers and label.

    Results outlive the slider positions that produced them — move a slider and
    the numbers on screen are suddenly from settings nobody can see. So every
    table is replaced wholesale, label and all, and a result always says what it
    was scored at. ``headers`` has to be passed again: an update only applies the
    properties it is given, and a value without headers would arrive as a
    dataframe of unnamed columns.
    """
    return gr.Dataframe(value=rows, headers=list(headers), label=label, wrap=True)


def _case_rows(cases: Sequence[Case]) -> list[list[Any]]:
    """The labelled set itself, as it reads in ``eval_cases.jsonl``."""
    return [
        [
            case.id,
            case.kind,
            case.question,
            " → ".join(case.history),
            ", ".join(case.sources),
            case.answer,
        ]
        for case in cases
    ]


def _judge_label(evaluation: Evaluation, kinds: Sequence[str]) -> str:
    """A button that says what pressing it will cost.

    Two calls per case — one to answer, one to grade — plus one more for each
    prior turn a multi-turn case replays, because those are answered for real
    rather than stubbed.
    """
    cases = evaluation.selected(kinds)
    if not cases:
        return "Run judge — no kinds selected"
    calls = sum(2 + len(case.history) for case in cases)
    return f"Run judge — {len(cases)} cases, ~{calls} model calls"


def build_demo(evaluation: Evaluation | None = None) -> gr.Blocks:
    """Build the dashboard: retrieval, answers, and the labelled set behind them.

    ``evaluation`` is built here rather than at import, so importing this module
    neither opens the store nor reads an API key. Every handler below is a
    closure over it, which is what keeps the store open across the whole
    session.
    """
    evaluation = evaluation if evaluation is not None else Evaluation()
    labelled = [case for case in evaluation.cases if case.sources]
    skipped = len(evaluation.cases) - len(labelled)

    def score(k: int, k_conversation: int) -> tuple[gr.Dataframe, gr.Dataframe]:
        k, k_conversation = int(k), int(k_conversation)
        scores = evaluation.score(k, k_conversation)
        stamp = _settings(k, k_conversation)
        return (
            _table(
                RETRIEVAL_SUMMARY_HEADERS,
                _retrieval_summary_rows(scores),
                f"By kind — {stamp}",
            ),
            _table(
                RETRIEVAL_CASE_HEADERS,
                _retrieval_case_rows(scores),
                f"By case, worst recall first — {stamp}",
            ),
        )

    def sweep(k_conversation: int) -> tuple[gr.Plot, gr.Dataframe]:
        k_conversation = int(k_conversation)
        results = evaluation.sweep(K_MAX, k_conversation)
        return (
            gr.Plot(
                value=_sweep_figure(results, k_conversation),
                label=f"Retrieval metrics against k — k_conversation={k_conversation}",
            ),
            _table(
                ["k", *METRICS],
                _sweep_rows(results),
                f"The same numbers — k_conversation={k_conversation}",
            ),
        )

    def judge(
        kinds: list[str], k: int, k_conversation: int
    ) -> Iterator[tuple[str, gr.Dataframe, gr.Dataframe, str]]:
        k, k_conversation = int(k), int(k_conversation)
        total = len(evaluation.selected(kinds))
        stamp = f"{', '.join(kinds)} at {_settings(k, k_conversation)}"

        def tables(results: Sequence[Graded]) -> tuple[gr.Dataframe, gr.Dataframe]:
            return (
                _table(
                    ANSWER_SUMMARY_HEADERS,
                    _answer_summary_rows(results),
                    f"By kind — {stamp}",
                ),
                _table(
                    ANSWER_CASE_HEADERS,
                    _answer_case_rows(results),
                    f"By case, failures first — {stamp}",
                ),
            )

        if not total:
            empty = (
                _table(ANSWER_SUMMARY_HEADERS, [], "By kind"),
                _table(ANSWER_CASE_HEADERS, [], "By case, failures first"),
            )
            yield "Pick at least one kind to grade.", *empty, ""
            return
        graded: list[Graded] = []
        yield f"grading {total} cases — {stamp}…", *tables(graded), ""
        for result in evaluation.graded(kinds, k, k_conversation):
            graded.append(result)
            passed = sum(one.passed for one in graded) / len(graded)
            yield (
                f"graded {len(graded)}/{total} · passed {passed:.2f} — {stamp}",
                *tables(graded),
                "",
            )
        yield (
            f"graded {len(graded)}/{total} · "
            f"passed {answer_summarise(graded).passed:.2f} — {stamp}",
            *tables(graded),
            _answer_notes(graded),
        )

    with gr.Blocks(title="llm-engineering — RAG evaluation") as demo:
        gr.Markdown(
            "# RAG evaluation — the Fretwork knowledge base\n"
            f"{len(evaluation.cases)} labelled cases against the Chroma store. "
            "**Retrieval** asks whether the right documents came back, and is free "
            f"to run; **Answers** asks whether `{MODEL}` then used them, and is "
            f"graded by `{JUDGE_MODEL}` at two model calls a case."
        )

        with gr.Tab("Retrieval"):
            gr.Markdown(
                f"Scored over the {len(labelled)} labelled cases — the {skipped} "
                "`refusal` cases carry no labels, because nothing in the knowledge "
                "base answers them, so retrieval metrics are undefined for them. "
                "No model is called here: press as often as you like."
            )
            with gr.Row():
                k_slider = gr.Slider(
                    1, K_MAX, value=K, step=1, label="k — chunks for the new message"
                )
                k_conversation_slider = gr.Slider(
                    K_CONVERSATION_MIN,
                    10,
                    value=K_CONVERSATION,
                    step=1,
                    label="k_conversation — chunks for the conversation-wide search",
                )
            score_button = gr.Button("Score retrieval", variant="primary")
            summary_table = gr.Dataframe(
                headers=RETRIEVAL_SUMMARY_HEADERS, label="By kind", wrap=True
            )
            case_table = gr.Dataframe(
                headers=RETRIEVAL_CASE_HEADERS,
                label="By case — worst recall first",
                wrap=True,
            )

            gr.Markdown(
                f"### Sweeping k\nScores every `k` from 1 to {K_MAX} at the "
                "`k_conversation` above. Recall should climb to a plateau while "
                "precision falls away — where those two cross is the setting "
                "argument, and the whole sweep costs nothing but seconds."
            )
            sweep_button = gr.Button(f"Sweep k 1–{K_MAX}")
            sweep_plot = gr.Plot(label="Retrieval metrics against k")
            sweep_table = gr.Dataframe(
                headers=["k", *METRICS], label="The same numbers", wrap=True
            )

            score_button.click(
                fn=score,
                inputs=[k_slider, k_conversation_slider],
                outputs=[summary_table, case_table],
            )
            sweep_button.click(
                fn=sweep,
                inputs=[k_conversation_slider],
                outputs=[sweep_plot, sweep_table],
            )

        with gr.Tab("Answers"):
            gr.Markdown(
                "Each case is answered by the bot and graded by a second model on "
                "four counts. `correct` and `grounded` are separate questions, and "
                "the pair is the interesting output: correct but ungrounded means "
                "the model answered from pretraining and the retrieval is not doing "
                "the work. `refusal` cases invert the test — declining is the pass.\n\n"
                "**This one spends money.** Pick fewer kinds to spend less; rows "
                "land as each case is graded."
            )
            with gr.Row():
                kinds_group = gr.CheckboxGroup(
                    choices=list(dict.fromkeys(case.kind for case in evaluation.cases)),
                    value=list(dict.fromkeys(case.kind for case in evaluation.cases)),
                    label="Kinds to grade",
                )
            with gr.Row():
                judge_k = gr.Slider(1, K_MAX, value=K, step=1, label="k")
                judge_k_conversation = gr.Slider(
                    K_CONVERSATION_MIN,
                    10,
                    value=K_CONVERSATION,
                    step=1,
                    label="k_conversation",
                )
            judge_button = gr.Button(
                _judge_label(evaluation, [case.kind for case in evaluation.cases]),
                variant="primary",
            )
            judge_status = gr.Markdown()
            judge_summary = gr.Dataframe(
                headers=ANSWER_SUMMARY_HEADERS, label="By kind", wrap=True
            )
            judge_cases = gr.Dataframe(
                headers=ANSWER_CASE_HEADERS,
                label="By case — failures first",
                wrap=True,
            )
            judge_notes = gr.Markdown()

            kinds_group.change(
                fn=lambda kinds: gr.Button(value=_judge_label(evaluation, kinds)),
                inputs=[kinds_group],
                outputs=[judge_button],
            )
            judge_button.click(
                fn=judge,
                inputs=[kinds_group, judge_k, judge_k_conversation],
                outputs=[judge_status, judge_summary, judge_cases, judge_notes],
            )

        with gr.Tab("Cases"):
            gr.Markdown(
                "`rag/eval_cases.jsonl`, hand-written. `sources` names the documents "
                "that must *all* come back — documents rather than chunk ids, so the "
                "labels survive a change to `CHUNK_SIZE`. The reference answer is "
                "read by the judge and by nothing else."
            )
            gr.Dataframe(
                value=_case_rows(evaluation.cases),
                headers=CASE_HEADERS,
                label=f"{len(evaluation.cases)} cases",
                wrap=True,
            )

    return demo


def launch(**kwargs: Any) -> None:
    """Build the dashboard and serve it.

    Everything expensive happens as :class:`Evaluation` is built: the store is
    opened (and raises if it was never built), and both models are constructed,
    which reads ``OPENROUTER_API_KEY``. The key is needed even to sit on the
    retrieval tab, because a ``Bot`` carries a model whether or not it is asked
    to answer anything.
    """
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
