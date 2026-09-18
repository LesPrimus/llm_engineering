"""GAIA's scorer: quasi exact match between the answer given and the one published.

A GAIA answer is a number, a short string, or a comma-separated list of those, so
grading needs no model — but it does need GAIA's own leniency, or scores come out
below what the leaderboard would give. ``42.0`` matches ``42``, ``$1,000`` matches
``1000``, ``Saint Petersburg`` matches ``st. petersburg`` only after punctuation
and case go away, and a list matches element by element with each element judged
as a number or as a string depending on what the reference holds.

This is a port of the scorer the benchmark ships, kept deliberately faithful:
where it is lenient we are lenient, and where it is strict — element counts,
ordering — so are we. The one thing it does not do is repair a badly formatted
answer, which is why :mod:`~agent_from_scratch.gaia_eval.prompts` spends most of
its words on the format.
"""

import re
import string

# The reference answers use both, sometimes in the same list.
SEPARATORS = ",;"


def is_correct(answer: str | None, reference: str) -> bool:
    """Does ``answer`` match the published ``reference`` under GAIA's rules?

    A missing answer — the model declined, or the call failed — is wrong rather
    than an error, so a run scores even when some of it did not finish.
    """
    if answer is None:
        return False
    if _as_number(reference) is not None:
        return _as_number(answer) == float(reference)
    if any(separator in reference for separator in SEPARATORS):
        return _list_matches(answer, reference)
    return _normalise(answer) == _normalise(reference)


def _list_matches(answer: str, reference: str) -> bool:
    """Compare two comma-separated lists element by element, in order.

    Punctuation is kept inside string elements here — the separator has already
    done the splitting, so a remaining period or hyphen is part of the value.
    """
    expected = _split(reference)
    given = _split(answer)
    if len(expected) != len(given):
        return False
    return all(
        _as_number(one) == float(other)
        if _as_number(other) is not None
        else _normalise(one, remove_punctuation=False)
        == _normalise(other, remove_punctuation=False)
        for one, other in zip(given, expected)
    )


def _split(text: str) -> list[str]:
    return re.split(f"[{SEPARATORS}]", text)


def _as_number(text: str) -> float | None:
    """Read ``text`` as a number, first dropping the units GAIA's rules ban.

    The model is told not to write ``$`` or ``%`` or thousands separators, and
    is not always obedient; stripping them costs nothing and recovers answers
    that are right about the quantity. Returns ``None`` when what is left is not
    a number at all, which is also how the caller asks "is this a number?".
    """
    stripped = text.strip().translate(str.maketrans("", "", "$%,"))
    try:
        return float(stripped)
    except ValueError:
        return None


def _normalise(text: str, *, remove_punctuation: bool = True) -> str:
    """Lowercase, and throw away whitespace — and usually punctuation with it."""
    squeezed = re.sub(r"\s", "", text).lower()
    if not remove_punctuation:
        return squeezed
    return squeezed.translate(str.maketrans("", "", string.punctuation))
