"""What a run produced: one record per question asked, and the rates over them.

An :class:`Attempt` is kept whether or not the model answered — a refusal and a
failed call are results too, and a run that quietly dropped them would report a
flattering accuracy over whatever happened to succeed. So every attempt counts
against the denominator, and :class:`Scorecard` breaks out *why* the misses
missed: wrong, declined, or never came back.

The records are pydantic models rather than tuples so a run can be written to
JSONL and read back for a second look without a parser.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from .constants import Level
from .models import GaiaReply
from .scoring import is_correct


class Attempt(BaseModel):
    """One model's shot at one task."""

    model_config = ConfigDict(frozen=True)

    model: str
    task_id: str
    level: Level
    question: str
    # GAIA's own answer, ``None`` on the test split where they stay private.
    expected: str | None
    # The name of the attached file, when the task has one. A no-tools baseline
    # cannot open it, so these are the questions it is expected to decline.
    file_name: str | None
    # ``None`` when the call failed or its reply would not validate.
    reply: GaiaReply | None
    error: str | None
    seconds: float

    @property
    def answer(self) -> str | None:
        """The final answer, if the model gave one."""
        return self.reply.final_answer if self.reply else None

    @property
    def declined(self) -> bool:
        """Did the model say outright that it could not solve this one?"""
        return self.reply is not None and not self.reply.is_solvable

    @property
    def correct(self) -> bool:
        """Scored against GAIA's published answer; always false without one."""
        return self.expected is not None and is_correct(self.answer, self.expected)


class Scorecard(BaseModel):
    """Counts over a set of attempts, all of them weighing the same."""

    model_config = ConfigDict(frozen=True)

    tasks: int
    correct: int
    declined: int
    errors: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.tasks if self.tasks else 0.0


def summarise(attempts: Sequence[Attempt]) -> Scorecard:
    """Tally one set of attempts."""
    return Scorecard(
        tasks=len(attempts),
        correct=sum(attempt.correct for attempt in attempts),
        declined=sum(attempt.declined for attempt in attempts),
        errors=sum(attempt.error is not None for attempt in attempts),
    )


def by_level(attempts: Sequence[Attempt]) -> dict[Level, Scorecard]:
    """Tally per level, hardest last — GAIA's levels are the interesting cut.

    Level 1 is the row a tool-less model can plausibly win; a score that holds
    up on levels 2 and 3 without tools is usually a memorised question rather
    than a solved one.
    """
    return {
        level: summarise([attempt for attempt in attempts if attempt.level == level])
        for level in sorted({attempt.level for attempt in attempts})
    }
