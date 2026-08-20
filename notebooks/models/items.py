from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Self

from pydantic import BaseModel

from notebooks.dataset_helpers import to_details, to_price, to_weight

# Stand-in for the rows that name no category at all, so an otherwise usable
# item is never dropped over a missing label.
UNCATEGORIZED = "Uncategorized"


class Item(BaseModel):
    title: str
    category: str
    price: Decimal
    weight: float | None = None

    @classmethod
    def from_datapoint(cls, datapoint: Mapping[str, Any]) -> Self | None:
        """Build an ``Item`` from one raw dataset row, or ``None`` if it has no price.

        Most rows in the dataset carry no usable price, so a missing one is a
        skip rather than an error — filter the ``None``s out at the call site.
        The price string is handed to pydantic untouched, keeping the exact
        base-10 value instead of routing it through a float.
        """
        price = datapoint.get("price")
        if to_price(price) is None:
            return None

        # ~5% of rows have no main_category; their categories breadcrumb starts
        # with the same top-level label, so fall back to that.
        categories = datapoint.get("categories") or []
        category = datapoint["main_category"] or (categories[0] if categories else "")

        return cls(
            title=datapoint["title"],
            category=category or UNCATEGORIZED,
            price=price,
            weight=to_weight(to_details(datapoint)),
        )
