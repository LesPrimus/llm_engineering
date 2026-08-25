from collections.abc import Sequence
from typing import Self

from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from pydantic import BaseModel

FEATURES = Features(
    {
        "title": Value("string"),
        "category": Value("string"),
        "price": Value("float64"),
        "full": Value("string"),
        "weight": Value("float64"),
        "summary": Value("string"),
        "prompt": Value("string"),
        "id": Value("string"),
    }
)


class Item(BaseModel):
    """
    An Item is a data-point of a Product with a Price
    """

    title: str
    category: str
    price: float
    full: str | None = None
    weight: float | None = None
    summary: str | None = None
    prompt: str | None = None
    id: str | None = None

    def __repr__(self) -> str:
        title = self.title if len(self.title) <= 50 else f"{self.title[:49]}…"
        return f"<{title} | {self.category} | ${self.price:.2f}>"

    __str__ = __repr__

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load from HuggingFace Hub and reconstruct Items"""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )

    @classmethod
    def to_hub(
        cls,
        dataset_name: str,
        train: Sequence[Self],
        validation: Sequence[Self],
        test: Sequence[Self],
        private: bool = True,
    ) -> None:
        """Push Items to the Hub as the three splits `from_hub` expects.

        Every split is written with the same explicit schema. Inferring it
        instead would type a column as null wherever no split has values --
        `summary` is a string in train but empty in the other two.
        """
        splits = DatasetDict(
            {
                "train": cls.as_dataset(train),
                "validation": cls.as_dataset(validation),
                "test": cls.as_dataset(test),
            }
        )
        splits.push_to_hub(dataset_name, private=private)

    @classmethod
    def as_dataset(cls, items: Sequence[Self]) -> Dataset:
        """Build one split, typed by FEATURES rather than by inference."""
        return Dataset.from_list(
            [item.model_dump() for item in items], features=FEATURES
        )
