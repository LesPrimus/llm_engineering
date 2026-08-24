from typing import Self

from datasets import load_dataset
from pydantic import BaseModel


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
