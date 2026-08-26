"""Score a price predictor against held-out Items.

A predictor is any callable that takes an `Item` and returns a price -- a
constant, a fitted sklearn model, or an LLM whose reply still has a dollar
sign in it. `evaluate` runs one over a slice of the test split and reports
the average absolute error next to two charts: predicted-vs-actual, and the
running error with a confidence band that says whether the sample was big
enough to trust the number.

    from notebooks.helpers.evaluator import evaluate

    evaluate(constant_pricer, test)

Chart text carries at most one "$" per string. Plotly reads a `$...$` pair
as LaTeX and hands it to MathJax; where no MathJax bundle is loaded -- a
PyCharm notebook, for one -- the typeset fails and the figure draws nothing
at all. Second amounts are written bare, with the unit left to the axis.
"""

import math
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import accumulate

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.notebook import tqdm

from notebooks.models.items import Item

Predictor = Callable[[Item], float | str | None]

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"green": GREEN, "orange": YELLOW, "red": RED}

DEFAULT_SIZE = 200
WORKERS = 5

NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


@dataclass(frozen=True, slots=True)
class Result:
    """One scored datapoint."""

    title: str
    guess: float
    truth: float
    error: float
    color: str


class Tester:
    """Run a predictor over `size` items and chart how far off it was."""

    def __init__(
        self,
        predictor: Predictor,
        data: Sequence[Item],
        title: str | None = None,
        size: int = DEFAULT_SIZE,
        workers: int = WORKERS,
    ) -> None:
        self.predictor = predictor
        self.data = list(data[:size])
        self.title = title or self.make_title(predictor)
        self.workers = workers
        self.results: list[Result] = []

    @staticmethod
    def make_title(predictor: Predictor) -> str:
        """Turn `xg_boost_pricer` into `Xg Boost Pricer`, `gpt_4o` into `GPT 4O`."""
        name = getattr(predictor, "__name__", "Predictor")
        return name.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def post_process(value: float | str | None) -> float:
        """Pull a price out of whatever the predictor returned.

        Numbers pass through. Strings get the first number in them, so an LLM
        answering "around $1,249.99" scores the same as a model returning a
        float. Anything unreadable counts as a guess of zero rather than
        killing the run.
        """
        if value is None:
            return 0.0
        if isinstance(value, str):
            match = NUMBER.search(value.replace("$", "").replace(",", ""))
            return float(match.group()) if match else 0.0
        return float(value)

    @staticmethod
    def color_for(error: float, truth: float) -> str:
        """Green for a good guess, red for a bad one, judged absolutely or relatively.

        Whichever measure is kinder wins: $30 off is fine on a cheap item, and
        20% off is fine on an expensive one.
        """
        relative = error / truth if truth else math.inf
        if error < 40 or relative < 0.2:
            return "green"
        if error < 80 or relative < 0.4:
            return "orange"
        return "red"

    def run_datapoint(self, item: Item) -> Result:
        guess = self.post_process(self.predictor(item))
        error = abs(guess - item.price)
        title = item.title if len(item.title) <= 40 else f"{item.title[:40]}…"
        return Result(
            title=title,
            guess=guess,
            truth=item.price,
            error=error,
            color=self.color_for(error, item.price),
        )

    def run(self) -> None:
        """Score every datapoint, printing errors as they land, then report.

        Threads, not processes: the work here is waiting on an API, not
        burning CPU. `map` preserves input order, so results line up with
        `data` even though they finish out of order.
        """
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            scored = pool.map(self.run_datapoint, self.data)
            for result in tqdm(scored, total=len(self.data)):
                self.results.append(result)
                print(f"{COLOR_MAP[result.color]}${result.error:.0f}{RESET} ", end="")
        print()
        self.report()

    def report(self) -> None:
        truths = [result.truth for result in self.results]
        guesses = [result.guess for result in self.results]
        average_error = sum(result.error for result in self.results) / len(self.results)
        mse = mean_squared_error(truths, guesses)
        r2 = r2_score(truths, guesses) * 100
        self.error_trend_chart()
        self.scatter_chart(
            f"{self.title} results<br>"
            f"<b>Error:</b> ${average_error:,.2f} "
            f"<b>MSE:</b> {mse:,.0f} "
            f"<b>r²:</b> {r2:.1f}%"
        )

    def error_trend_chart(self) -> None:
        """Plot the running average error with its 95% confidence band.

        The band is the point of this chart. If it is still wide at the last
        datapoint, the gap between two models may be sampling noise rather
        than a real difference, and the run needs a bigger `size`.
        """
        errors = [result.error for result in self.results]
        counts = list(range(1, len(errors) + 1))

        running_means = [
            total / n for total, n in zip(accumulate(errors), counts, strict=True)
        ]
        running_squares = accumulate(error * error for error in errors)
        running_stds = [
            math.sqrt(max(squares / n - mean**2, 0.0)) if n > 1 else 0.0
            for n, squares, mean in zip(
                counts, running_squares, running_means, strict=True
            )
        ]
        ci = [
            1.96 * std / math.sqrt(n) if n > 1 else 0.0
            for n, std in zip(counts, running_stds, strict=True)
        ]
        upper = [mean + margin for mean, margin in zip(running_means, ci, strict=True)]
        lower = [mean - margin for mean, margin in zip(running_means, ci, strict=True)]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=counts + counts[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=counts,
                y=running_means,
                mode="lines",
                line={"width": 3, "color": "firebrick"},
                customdata=ci,
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=$%{y:,.2f}<br>"
                    "±95% CI=±%{customdata:,.2f}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            title=f"{self.title} Error: ${running_means[-1]:,.2f} ± {ci[-1]:,.2f}",
            xaxis_title="Number of Datapoints",
            yaxis_title="Average Absolute Error ($)",
            width=1000,
            height=360,
            showlegend=False,
        )
        figure.show()

    def scatter_chart(self, title: str) -> None:
        """Plot every guess against the truth, with y = x for reference.

        Dots hugging the dashed line are good guesses. A flat horizontal band
        means the predictor answers the same thing whatever it is shown.
        """
        frame = pd.DataFrame(
            {
                "truth": [result.truth for result in self.results],
                "guess": [result.guess for result in self.results],
                "color": [result.color for result in self.results],
                "hover": [
                    f"{result.title}<br>"
                    f"Guess {result.guess:,.2f} / Actual {result.truth:,.2f}"
                    for result in self.results
                ],
            }
        )
        limit = float(max(frame["truth"].max(), frame["guess"].max()))

        figure = px.scatter(
            frame,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual Price ($)", "guess": "Predicted Price ($)"},
            width=1000,
            height=800,
        )
        # Plotly Express splits one trace per color, so hover text is attached
        # per trace rather than per row.
        for trace in figure.data:
            trace.customdata = frame.loc[
                frame["color"] == trace.name, ["hover"]
            ].to_numpy()
            trace.hovertemplate = "%{customdata[0]}<extra></extra>"
            trace.marker.update(size=6)

        figure.add_trace(
            go.Scatter(
                x=[0, limit],
                y=[0, limit],
                mode="lines",
                line={"width": 2, "dash": "dash", "color": "deepskyblue"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.update_xaxes(range=[0, limit])
        figure.update_yaxes(range=[0, limit])
        figure.update_layout(showlegend=False)
        figure.show()


def evaluate(
    predictor: Predictor,
    data: Sequence[Item],
    title: str | None = None,
    size: int = DEFAULT_SIZE,
    workers: int = WORKERS,
) -> Tester:
    """Score `predictor` over the first `size` items and show the charts.

    Returns the finished `Tester` so its `results` can be inspected -- useful
    for picking through the worst misses after a run.
    """
    tester = Tester(predictor, data, title=title, size=size, workers=workers)
    tester.run()
    return tester
