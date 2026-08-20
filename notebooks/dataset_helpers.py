"""Helpers shared by the notebooks in this directory."""

import json
import re
from collections.abc import Mapping
from typing import Any

from datasets import Dataset
from tqdm.notebook import tqdm

# A leading number, then the rest of the string as the unit — so multi-word units
# ("100 Hundredths Pounds") parse, and odd values ("1.5-2 pounds") miss the unit
# table and come back as None rather than raising.
_WEIGHT = re.compile(r"\s*([\d.,]+)\s*(.*?)\s*$")

# Pounds per unit; the dataset states weights in any of these.
_POUNDS_PER_UNIT: dict[str, float] = {
    "pounds": 1.0,
    "ounces": 1 / 16,
    "grams": 1 / 453.592,
    "milligrams": 1 / 453_592,
    "kilograms": 1 / 0.453592,
    "hundredths pounds": 1 / 100,
}


def to_price(value: str | None) -> float | None:
    """Parse a price, which the loading script stores as a string, using "None" for missing."""
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def rank_by_price(dataset: Dataset) -> list[tuple[float, int]]:
    """(price, index) pairs for the items that have a price, most expensive first."""
    prices = dataset[
        "price"
    ]  # materializes the column; the slow part, before tqdm sees anything

    return sorted(
        (
            (price, i)
            for i, value in tqdm(enumerate(prices), total=len(prices))
            if (price := to_price(value)) is not None
        ),
        reverse=True,
    )


def report_top_prices(
    dataset: Dataset, ranked: list[tuple[float, int]], top: int = 10
) -> None:
    print(f"{len(ranked):,} of {len(dataset):,} items have a price\n")
    for price, i in ranked[:top]:
        print(f"${price:>10,.2f}  {dataset[i]['title'][:60]}")

    most_expensive = dataset[ranked[0][1]]
    print(f"\nMost expensive: {most_expensive['title']}")
    print(f"Price:  ${float(most_expensive['price']):,.2f}")
    print(f"Store:  {most_expensive['store']}")
    print(
        f"Rating: {most_expensive['average_rating']} from {most_expensive['rating_number']:,} ratings"
    )


def to_weight(details: Mapping[str, str]) -> float | None:
    """Parse an ``Item Weight`` detail into pounds, or ``None`` if unusable.

    ``None`` covers every case the data doesn't pin down — the key is missing,
    the value is blank or unparsable, or the unit isn't one we know — so callers
    can tell "no weight" apart from a genuinely light item, and can count the
    units that fall through instead of silently reading them as zero.
    """
    weight = details.get("Item Weight")
    if not isinstance(weight, str):
        return None

    match = _WEIGHT.match(weight)
    if match is None:
        return None

    amount, unit = match.groups()
    try:
        pounds = float(amount.replace(",", ""))
    except ValueError:  # e.g. "1.2.3", a bare "."
        return None

    unit = unit.lower()
    # Singular is rarer but real ("1 Pound"); fall back to the plural key.
    per_unit = _POUNDS_PER_UNIT.get(unit) or _POUNDS_PER_UNIT.get(f"{unit}s")
    if per_unit is None:
        return None
    return pounds * per_unit


def to_details(datapoint: Mapping[str, Any]) -> dict[str, Any]:
    """The ``details`` column, which the loading script stores as a JSON string.

    Returns an empty dict when the column is absent, isn't valid JSON, or holds
    something other than an object, so callers can just ``.get`` a key.
    """
    raw = datapoint.get("details")
    if not isinstance(raw, str):
        return {}
    try:
        details = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return details if isinstance(details, dict) else {}
