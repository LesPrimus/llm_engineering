"""Load the GAIA benchmark from the Hugging Face Hub as a ``GaiaDataset`` of tasks.

The dataset is gated: accept its terms on the Hub, and the loader picks up the
token from ``HF_TOKEN`` or ``huggingface-cli login``.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Self

from datasets import load_dataset

from .constants import REPO_ID, Level, Split
from .models import GaiaTask


@dataclass(frozen=True)
class GaiaDataset:
    """One split of GAIA 2023, optionally narrowed to a single level."""

    split: Split
    level: Level | None
    tasks: tuple[GaiaTask, ...]

    @classmethod
    def from_hub(
        cls, split: Split = Split.VALIDATION, level: Level | None = None
    ) -> Self:
        config = "2023_all" if level is None else f"2023_level{level}"
        rows = load_dataset(REPO_ID, config, split=split)
        return cls(split, level, tuple(GaiaTask.model_validate(row) for row in rows))

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[GaiaTask]:
        return iter(self.tasks)

    def __getitem__(self, index: int) -> GaiaTask:
        return self.tasks[index]

    def get(self, task_id: str) -> GaiaTask:
        """The task with this id, or ``KeyError`` if the split and level lack it."""
        try:
            return self._by_id[task_id]
        except KeyError:
            level = "" if self.level is None else f" level {self.level}"
            raise KeyError(f"no task {task_id!r} in {self.split}{level}") from None

    @cached_property
    def _by_id(self) -> dict[str, GaiaTask]:
        return {task.task_id: task for task in self.tasks}
