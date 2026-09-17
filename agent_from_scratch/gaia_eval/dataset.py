"""Load the GAIA benchmark from the Hugging Face Hub as typed ``GaiaTask`` records.

The dataset is gated: accept its terms on the Hub, and the loader picks up the
token from ``HF_TOKEN`` or ``huggingface-cli login``.

The raw rows are shaped for humans, not code: column names carry spaces and a
question mark, numbers arrive as strings, and "nothing here" is an empty
string. The models below rename, cast and blank those out in one
``model_validate`` call, and raise naming the field if the upstream format
drifts again.
"""

from typing import Annotated, Literal, Self

from datasets import load_dataset
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

REPO_ID = "gaia-benchmark/GAIA"

Split = Literal["validation", "test"]
Level = Literal[1, 2, 3]


def _blank_to_none(value: str | None) -> str | None:
    return value or None


def _hidden_to_none(value: str | None) -> str | None:
    """The test split's answers are private; every one of them reads ``?``."""
    return None if value == "?" else _blank_to_none(value)


def _count(value: str | int) -> int | None:
    """Annotators mostly wrote a number, but one left it blank and one wrote prose."""
    text = str(value)
    return int(text) if text.isdigit() else None


class AnnotatorMetadata(BaseModel):
    """How the human annotator solved the task. Only the validation split has it."""

    model_config = ConfigDict(frozen=True)

    steps: str = Field(alias="Steps")
    number_of_steps: Annotated[int | None, BeforeValidator(_count)] = Field(
        alias="Number of steps"
    )
    time_taken: str = Field(alias="How long did this take?")
    tools: str = Field(alias="Tools")
    number_of_tools: Annotated[int | None, BeforeValidator(_count)] = Field(
        alias="Number of tools"
    )


class GaiaTask(BaseModel):
    """One GAIA question, with its answer when the split publishes it."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    question: str = Field(alias="Question")
    level: Annotated[Level, BeforeValidator(int)] = Field(alias="Level")
    final_answer: Annotated[str | None, BeforeValidator(_hidden_to_none)] = Field(
        alias="Final answer"
    )
    file_name: Annotated[str | None, BeforeValidator(_blank_to_none)]
    # Relative to the dataset repo, e.g. ``2023/validation/<task_id>.xlsx``.
    file_path: Annotated[str | None, BeforeValidator(_blank_to_none)]
    annotator_metadata: AnnotatorMetadata | None = Field(alias="Annotator Metadata")

    @field_validator("annotator_metadata", mode="before")
    @classmethod
    def _empty_metadata_to_none(
        cls, value: dict[str, str] | None
    ) -> dict[str, str] | None:
        """The test split ships the struct with every field an empty string."""
        if value is None or not any(value.values()):
            return None
        return value

    @classmethod
    def from_hub(
        cls, split: Split = "validation", level: Level | None = None
    ) -> list[Self]:
        """Load one split of GAIA 2023, optionally narrowed to a single level."""
        config = "2023_all" if level is None else f"2023_level{level}"
        rows = load_dataset(REPO_ID, config, split=split)
        return [cls.model_validate(row) for row in rows]
