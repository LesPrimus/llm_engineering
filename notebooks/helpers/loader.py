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
7. Persist. Done -- `save_jobs` writes the ids to `jobs.json` after every
   successful submission, and `restore` re-attaches them to freshly created
   batches by name.

Open questions:

- Groq accepts up to 50,000 lines and 200MB per file, so `batch_size` trades
  fewer job ids to track against coarser retries when one job fails.
- Items must arrive with `id` and `full` populated; `ed-donner/items_raw_lite`
  leaves `id` null, so the caller assigns ids before batching.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import Any, ClassVar, Self

from tqdm.auto import tqdm

from notebooks.helpers.batch_client import BatchClient
from notebooks.models.items import Item

JOBS_FILE = "jobs.json"

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
    file_id: str | None = None
    batch_id: str | None = None
    output_file_id: str | None = None

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

    def path(self, folder: Path) -> Path:
        return folder / f"{self.name}.jsonl"

    def write(self, folder: Path, config: RequestConfig) -> Path:
        lines = [f"{json.dumps(request)}\n" for request in self.to_requests(config)]
        folder.mkdir(parents=True, exist_ok=True)
        path = self.path(folder)
        with path.open("w", encoding="utf-8") as file:
            file.writelines(lines)
        return path


@dataclass
class SubmitReport:
    """Outcome of a `submit_all` run: nothing raises, everything is reported."""

    submitted: list[Batch] = field(default_factory=list)
    skipped: list[Batch] = field(default_factory=list)
    failed: list[tuple[Batch, str]] = field(default_factory=list)

    def __str__(self) -> str:
        summary = (
            f"{len(self.submitted)} submitted, "
            f"{len(self.skipped)} already submitted, "
            f"{len(self.failed)} failed"
        )
        details = "".join(f"\n  {batch.name}: {error}" for batch, error in self.failed)
        return summary + details


@dataclass
class BatchLoader:
    BATCH_SIZE: ClassVar[int] = 1_000  # default only; override per run via create()
    FOLDER: ClassVar[Path] = Path("batches")

    batches: list[Batch]
    folder: Path = FOLDER
    config: RequestConfig = field(default_factory=RequestConfig)
    client: BatchClient | None = None
    written: list[Path] = field(default_factory=list, init=False)

    @classmethod
    def create(
        cls,
        items: list[Item],
        batch_size: int = BATCH_SIZE,
        folder: Path = FOLDER,
        config: RequestConfig | None = None,
        client: BatchClient | None = None,
    ) -> Self:
        batches = [
            Batch(items=list(chunk), start=index * batch_size)
            for index, chunk in enumerate(batched(items, batch_size))
        ]
        return cls(
            batches=batches,
            folder=folder,
            config=config or RequestConfig(),
            client=client,
        )

    def __enter__(self) -> Self:
        self.folder.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None and not self.submitted:
            for path in self.written:
                path.unlink(missing_ok=True)
            self.written = []
        return False

    @property
    def submitted(self) -> list[Batch]:
        return [batch for batch in self.batches if batch.batch_id is not None]

    def __len__(self) -> int:
        return len(self.batches)

    def make_files(self) -> list[Path]:
        self.written = []
        for batch in tqdm(self.batches, desc="Writing batches", unit="file"):
            self.written.append(batch.write(self.folder, self.config))
        return self.written

    def submit_all(self) -> SubmitReport:
        """Upload and submit every batch, carrying on past individual failures.

        Batches that already hold a `batch_id` are skipped, so a re-run retries
        only what failed rather than duplicating live jobs.
        """
        if self.client is None:
            raise ValueError("no BatchClient: pass client=... to BatchLoader.create")

        report = SubmitReport()
        for batch in tqdm(self.batches, desc="Submitting batches", unit="job"):
            if batch.batch_id is not None:
                report.skipped.append(batch)
                continue
            try:
                batch.file_id = self.client.upload(batch.path(self.folder))
                batch.batch_id = self.client.submit(batch.file_id, self.config.endpoint)
            except Exception as error:
                report.failed.append((batch, f"{type(error).__name__}: {error}"))
            else:
                report.submitted.append(batch)
                self.save_jobs()
        return report

    def jobs_path(self, path: Path | None = None) -> Path:
        return path or self.folder / JOBS_FILE

    def save_jobs(self, path: Path | None = None) -> Path:
        """Write the job ids to disk so a restart cannot strand live batches."""
        path = self.jobs_path(path)
        jobs = {
            batch.name: {
                "file_id": batch.file_id,
                "batch_id": batch.batch_id,
                "output_file_id": batch.output_file_id,
            }
            for batch in self.batches
            if batch.file_id is not None or batch.batch_id is not None
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        return path

    def restore(self, path: Path | None = None) -> int:
        """Re-attach saved job ids to freshly created batches, by batch name."""
        path = self.jobs_path(path)
        jobs = json.loads(path.read_text(encoding="utf-8"))
        by_name = {batch.name: batch for batch in self.batches}

        unknown = sorted(set(jobs) - set(by_name))
        if unknown:
            raise ValueError(
                f"{path} holds batches this loader does not have: "
                f"{', '.join(unknown)}. The chunk boundaries differ -- check that "
                f"batch_size and the item list match the run that saved it."
            )

        for name, job in jobs.items():
            batch = by_name[name]
            batch.file_id = job["file_id"]
            batch.batch_id = job["batch_id"]
            batch.output_file_id = job["output_file_id"]
        return len(jobs)
