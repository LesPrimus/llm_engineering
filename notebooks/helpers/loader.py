import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any, ClassVar, Self

from notebooks.models.items import Item

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


@dataclass(frozen=True)
class RequestConfig:
    model: str = "openai/gpt-oss-20b"
    system_prompt: str = SYSTEM_PROMPT
    reasoning_effort: str = "low"
    endpoint: str = "/v1/chat/completions"


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

    def to_requests(self, config: RequestConfig) -> Iterator[dict[str, Any]]:
        for offset, item in enumerate(self.items):
            if item.id is None or item.full is None:
                raise ValueError(
                    f"Item at position {self.start + offset} is missing id or full text: {item!r}"
                )
            yield {
                "custom_id": item.id,
                "method": "POST",
                "url": config.endpoint,
                "body": {
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": item.full},
                    ],
                    "reasoning_effort": config.reasoning_effort,
                },
            }

    def write(self, folder: Path, config: RequestConfig) -> Path:
        lines = [f"{json.dumps(request)}\n" for request in self.to_requests(config)]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.name}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            file.writelines(lines)
        return path


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
