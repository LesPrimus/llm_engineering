"""The fixed choices GAIA offers: which split, and which difficulty level."""

from enum import IntEnum, StrEnum


class Split(StrEnum):
    """The dataset's splits. Only validation publishes its answers."""

    VALIDATION = "validation"
    TEST = "test"


class Level(IntEnum):
    """How hard a task is, from 1 (within reach of a very good LLM) to 3."""

    ONE = 1
    TWO = 2
    THREE = 3
