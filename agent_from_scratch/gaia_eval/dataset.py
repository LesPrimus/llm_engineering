"""Load the GAIA benchmark from the Hugging Face Hub as a ``GaiaDataset`` of tasks.

The dataset is gated: accept its terms on the Hub, and the loader picks up the
token from ``HF_TOKEN`` or ``huggingface-cli login``.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from datasets import load_dataset

from .models import GaiaTask, Level

REPO_ID = "gaia-benchmark/GAIA"


class Split(StrEnum):
    """The dataset's splits. Only validation publishes its answers."""

    VALIDATION = "validation"
    TEST = "test"


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
