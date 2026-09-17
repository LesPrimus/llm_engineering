"""Clean-ups for raw GAIA values, run before pydantic checks their type."""


def blank_to_none(value: str | None) -> str | None:
    return value or None


def hidden_to_none(value: str | None) -> str | None:
    """The test split's answers are private; every one of them reads ``?``."""
    return None if value == "?" else blank_to_none(value)


def count(value: str | int) -> int | None:
    """Annotators mostly wrote a number, but one left it blank and one wrote prose."""
    text = str(value)
    return int(text) if text.isdigit() else None
