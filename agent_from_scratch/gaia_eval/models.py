"""The records GAIA runs on: tasks as the dataset ships them, replies as a model gives them.

The raw dataset rows are shaped for humans, not code: column names carry spaces
and a question mark, numbers arrive as strings, and "nothing here" is an empty
string. ``GaiaTask`` renames, casts and blanks those out in one
``model_validate`` call, and raises naming the field if the upstream format
drifts again.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

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


# The docstring below is sent to the model as the schema's description, so notes
# for us live here instead.
#
# The field names are the keys ``prompts.SYSTEM_PROMPT`` asks for: rename them in
# both places or not at all. Both text fields are nullable rather than defaulted,
# because strict structured outputs require every key to be present.
class GaiaReply(BaseModel):
    """Either an answer, or the reason there isn't one."""

    model_config = ConfigDict(frozen=True)

    is_solvable: bool
    final_answer: str | None
    unsolvable_reason: str | None
