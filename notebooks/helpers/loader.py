"""Chunk items into batches and write them as JSONL request files.

Currently covers the local half of the pipeline: `BatchLoader.create` splits
items into `Batch` windows, and `make_files` writes one `.jsonl` request file
per batch, named `{start}_{end}.jsonl`.

Remaining work, in order:

1. Client. Point the OpenAI SDK at Groq -- `base_url="https://api.groq.com/openai/v1"`
   with `GROQ_API_KEY` passed explicitly, or the SDK silently falls back to
   `OPENAI_API_KEY`. Smoke-test a two-line file first: Groq documents the batch
   routes under that base path but does not officially bless the OpenAI SDK.
2. Upload. `client.files.create(file=..., purpose="batch")` per written file,
   keeping the returned `file_id` on the Batch.
3. Submit. `client.batches.create(completion_window=..., endpoint=config.endpoint,
   input_file_id=...)` -> `batch_id`. Groq allows 24h to 7d; longer windows
   complete more reliably under load.
4. Poll. `client.batches.retrieve(batch_id)` until status is "completed", then
   capture `output_file_id`. Jobs can also fail or expire -- handle both.
5. Download. `client.files.content(output_file_id).write_to_file(...)` into an
   `output/` folder, and add `notebooks/**/output/` to .gitignore.
6. Apply. Map each result line's `custom_id` back to its Item and set `summary`.
   Results arrive unordered, so look up by id rather than by position, and
   account for lines that carry an error instead of a response.
7. Persist. A run holds one `batch_id` per batch against a 24h+ window, so a
   kernel restart must not lose them. Decide what to save (job ids only, with
   items re-attached on load) and in what format.

Open questions:

- `BATCH_SIZE` is a ClassVar, so it cannot be varied per run without mutating
  the class. Make it a `create()` parameter if a run needs a different size.
- Items must arrive with `id` and `full` populated; `ed-donner/items_raw_lite`
  leaves `id` null, so the caller assigns ids before batching.
"""

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
