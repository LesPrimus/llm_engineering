"""Summarize items through a batch API, one JSONL request file per chunk.

`BatchLoader.create` splits items into `Batch` windows; `make_files` writes one
request file per batch; `submit_all` uploads and submits them, recording the job
ids in `jobs.json`; `fetch` polls, downloads results, and fills in
`Item.summary`. Entering the loader restores whatever run its folder already
holds, so re-running a cell resumes rather than duplicating live jobs.

Summaries live on the items themselves, so nothing here stores them.
`Item.to_hub` publishes the finished splits; the result files under `output/`
are a local safety net that `fetch` re-applies without calling the API.

Notes:

- Groq accepts up to 50,000 lines and 200MB per file, so `batch_size` trades
  fewer job ids to track against coarser retries when one job fails.
- Items must arrive with `id` and `full` populated; `ed-donner/items_raw_lite`
  leaves `id` null, so the caller assigns ids before batching.
- Results come back unordered and may contain error lines, so `Batch.apply`
  matches on `custom_id` and counts what it could not use.
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
OUTPUT_FOLDER = "output"
COMPLETED = "completed"
TERMINAL_FAILURES = frozenset({"failed", "expired", "cancelled"})

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
    error_file_id: str | None = None

    @property
    def end(self) -> int:
        return self.start + len(self.items)

    @property
    def name(self) -> str:
        return f"{self.start}_{self.end}"

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        job = self.batch_id or "not submitted"
        return f"<Batch {self.name} | {len(self.items)} items | {job}>"

    __str__ = __repr__

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

    def apply(self, path: Path) -> tuple[int, int]:
        """Fill in item summaries from a result file.

        Results arrive in arbitrary order, so every line is matched on its
        `custom_id`. Returns (summaries applied, lines that carried no usable
        response -- an error line, or an id this batch does not own).
        """
        by_id = {item.id: item for item in self.items}
        applied = unusable = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            item = by_id.get(result.get("custom_id"))
            choices = ((result.get("response") or {}).get("body") or {}).get("choices")
            if item is None or not choices:
                unusable += 1
                continue
            item.summary = choices[0]["message"]["content"].strip()
            applied += 1
        return applied, unusable

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

    __repr__ = __str__


@dataclass
class FetchReport:
    """Outcome of a `fetch` run. Terminal failures are reported, not raised."""

    applied: list[Batch] = field(default_factory=list)
    pending: list[tuple[Batch, str]] = field(default_factory=list)
    failed: list[tuple[Batch, str]] = field(default_factory=list)
    summaries: int = 0
    unusable: int = 0

    def __str__(self) -> str:
        summary = (
            f"{len(self.applied)} applied ({self.summaries} summaries), "
            f"{len(self.pending)} pending, {len(self.failed)} failed"
        )
        if self.unusable:
            summary += f", {self.unusable} results unusable"
        details = "".join(
            f"\n  {batch.name}: {status}"
            for batch, status in self.pending + self.failed
        )
        return summary + details

    __repr__ = __str__


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
        """Prepare the folder and pick up any run already recorded there.

        Restoring on entry makes re-running a submission cell a no-op instead of
        a second set of live jobs: batches that already hold an id are skipped.
        To start over deliberately, delete the jobs file or use a new folder.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.jobs_path().exists():
            self.restore()
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

    @property
    def output_folder(self) -> Path:
        return self.folder.parent / OUTPUT_FOLDER

    def poll(self) -> dict[str, str]:
        """Refresh the status of every live job, recording result file ids."""
        if self.client is None:
            raise ValueError("no BatchClient: pass client=... to BatchLoader.create")

        statuses = {
            batch.name: COMPLETED
            for batch in self.batches
            if batch.output_file_id is not None
        }
        live = [
            batch
            for batch in self.batches
            if batch.batch_id is not None and batch.output_file_id is None
        ]
        for batch in tqdm(live, desc="Polling batches", unit="job"):
            job = self.client.status(batch.batch_id)
            statuses[batch.name] = job.status
            batch.error_file_id = getattr(job, "error_file_id", None)
            if job.status == COMPLETED:
                batch.output_file_id = job.output_file_id
        if live:
            self.save_jobs()
        return statuses

    def fetch(self) -> FetchReport:
        """Poll, download whatever finished, and apply summaries to the items."""
        statuses = self.poll()
        report = FetchReport()
        ready = []
        for batch in self.batches:
            if batch.output_file_id is not None:
                ready.append(batch)
            elif batch.batch_id is None:
                report.pending.append((batch, "not submitted"))
            elif statuses.get(batch.name, "unknown") in TERMINAL_FAILURES:
                report.failed.append((batch, statuses[batch.name]))
            else:
                report.pending.append((batch, statuses.get(batch.name, "unknown")))

        for batch in tqdm(ready, desc="Fetching results", unit="file"):
            path = batch.path(self.output_folder)
            if not path.exists():
                self.client.download(batch.output_file_id, path)
            applied, unusable = batch.apply(path)
            report.applied.append(batch)
            report.summaries += applied
            report.unusable += unusable
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
                "error_file_id": batch.error_file_id,
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
            batch.error_file_id = job.get("error_file_id")
        return len(jobs)
