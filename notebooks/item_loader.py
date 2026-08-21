"""Load one Amazon Reviews category into ``Item`` objects across a process pool.

Safe to call straight from a notebook. From a *script* the ``load`` call needs
the usual ``if __name__ == "__main__":`` guard — Python 3.14 starts workers with
``forkserver`` on Linux, so each one re-imports the main module by path.
"""

import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from math import ceil

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from notebooks.models.items import Item

DATASET = "McAuley-Lab/Amazon-Reviews-2023"

# The only columns Item reads. The ten we drop — images, videos, features and
# description chief among them — are ~74% of the file's bytes, and the formatter
# would decode every one of them into Python objects per row just to discard them.
ITEM_COLUMNS = ["title", "price", "main_category", "categories", "details"]

# Measured sweet spot on this data: bigger chunks leave workers idle at the tail,
# smaller ones spend more on shipping results back than they save. Fixed size
# rather than a fixed count, so per-worker memory stays flat whether the category
# has 90k rows or 2M.
CHUNK_SIZE = 10_000


@dataclass(slots=True)
class ChunkedItemLoader:
    """Turns one category's raw rows into ``Item``s, chunk by chunk, in parallel.

    The expensive step is not the loading — the Arrow file is memory-mapped, so
    that part is nearly free. It is the *formatting*: walking each column's
    buffers to build a Python dict per row, which costs about six times what the
    ``Item`` parsing does. Both happen inside the worker, which is the whole
    point of shipping a slice rather than a list of rows.
    """

    category: str
    chunk_size: int = CHUNK_SIZE
    workers: int | None = None
    # Skips the Hub round trip when the caller already holds the rows.
    source: Dataset | None = None
    dataset: Dataset = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = self.source
        if source is None:
            source = load_dataset(
                DATASET,
                f"raw_meta_{self.category}",
                split="full",
                trust_remote_code=True,
            )
        # Project the columns away up front, before any row is decoded.
        self.dataset = source.select_columns(ITEM_COLUMNS)

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def chunk_count(self) -> int:
        # ceil, not //: a floor would silently drop the final partial chunk.
        return ceil(len(self.dataset) / self.chunk_size)

    def _chunks(self) -> Iterator[Dataset]:
        """Contiguous slices of the dataset.

        ``select`` over a ``range`` is the zero-copy path: it shares the parent's
        Arrow buffers and builds no index, so a slice pickles to a few KiB of
        file path and offsets, and every worker re-maps the same cache file.

        A contiguous *sequence* of row numbers — ``itertools.batched``, say —
        lands on that same path, but only after ``select`` walks it element by
        element to rediscover the contiguity a ``range`` states in O(1). Only
        genuinely scattered indices are worth avoiding: those build a lookup
        table and read through it.
        """
        total = len(self.dataset)
        for start in range(0, total, self.chunk_size):
            yield self.dataset.select(range(start, min(start + self.chunk_size, total)))

    def _items_from_chunk(self, chunk: Dataset) -> list[Item]:
        """Decode one slice into ``Item``s. Runs in a worker process.

        Most rows carry no usable price, so ``from_datapoint`` returning ``None``
        is the common case rather than an error.
        """
        return [
            item
            for datapoint in chunk
            if (item := Item.from_datapoint(datapoint)) is not None
        ]

    def load(self) -> list[Item]:
        """Every priced item in the category, in dataset order."""
        # One chunk isn't worth a pool; spawning workers would cost more than the
        # work. Small categories land here.
        if self.chunk_count == 1:
            return self._items_from_chunk(self.dataset)

        workers = min(self.workers or os.process_cpu_count() or 1, self.chunk_count)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map yields in submission order, so the result stays dataset-ordered.
            batches = tqdm(
                pool.map(self._items_from_chunk, self._chunks()),
                total=self.chunk_count,
                desc=self.category,
            )
            return [item for batch in batches for item in batch]
