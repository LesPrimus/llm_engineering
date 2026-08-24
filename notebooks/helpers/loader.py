from dataclasses import dataclass
from itertools import batched
from typing import ClassVar, Self

from notebooks.models.items import Item


@dataclass
class Batch:
    items: list[Item]
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.items)

    @property
    def name(self) -> str:
        return f"{self.start}_{self.end}"

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class BatchLoader:
    BATCH_SIZE: ClassVar[int] = 1_000

    batches: list[Batch]

    @classmethod
    def create(cls, items: list[Item]) -> Self:
        batches = [
            Batch(items=list(chunk), start=index * cls.BATCH_SIZE)
            for index, chunk in enumerate(batched(items, cls.BATCH_SIZE))
        ]
        return cls(batches=batches)
