"""Helpers shared by the notebooks in this directory."""

from datasets import Dataset
from tqdm.notebook import tqdm


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
