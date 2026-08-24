import json
from collections.abc import Iterator
from dataclasses import dataclass, field
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
            missing = [name for name in ("id", "full") if getattr(item, name) is None]
            if missing:
                raise ValueError(
                    f"Item at position {self.start + offset} is missing "
                    f"{' and '.join(missing)}: {item!r}"
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
    FOLDER: ClassVar[Path] = Path("batches")

    batches: list[Batch]
    folder: Path = FOLDER
    config: RequestConfig = field(default_factory=RequestConfig)
    written: list[Path] = field(default_factory=list, init=False)

    @classmethod
    def create(
        cls,
        items: list[Item],
        folder: Path = FOLDER,
        config: RequestConfig | None = None,
    ) -> Self:
        batches = [
            Batch(items=list(chunk), start=index * cls.BATCH_SIZE)
            for index, chunk in enumerate(batched(items, cls.BATCH_SIZE))
        ]
        return cls(batches=batches, folder=folder, config=config or RequestConfig())

    def __enter__(self) -> Self:
        self.folder.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            for path in self.written:
                path.unlink(missing_ok=True)
            self.written = []
        return False

    def __len__(self) -> int:
        return len(self.batches)

    def make_files(self) -> list[Path]:
        self.written = []
        for batch in self.batches:
            self.written.append(batch.write(self.folder, self.config))
        return self.written
