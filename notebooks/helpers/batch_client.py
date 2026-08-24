"""Thin wrapper over the Groq batch endpoints, reached through the OpenAI SDK.

Groq exposes `/files` and `/batches` under an OpenAI-compatible base path, so the
`openai` package talks to them directly -- no extra dependency. The API key must
be passed explicitly: left to itself the SDK falls back to `OPENAI_API_KEY`.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
COMPLETION_WINDOW = "24h"


@dataclass
class BatchClient:
    """Owns the API client and the job-level settings shared by every batch."""

    client: OpenAI
    completion_window: str = COMPLETION_WINDOW

    @classmethod
    def from_env(cls, completion_window: str = COMPLETION_WINDOW) -> Self:
        return cls(
            client=OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"]),
            completion_window=completion_window,
        )

    def upload(self, path: Path) -> str:
        """Upload a request file and return its file id."""
        with path.open("rb") as file:
            return self.client.files.create(file=file, purpose="batch").id

    def submit(self, file_id: str, endpoint: str) -> str:
        """Create a batch job from an uploaded file and return its batch id."""
        job = self.client.batches.create(
            completion_window=self.completion_window,
            endpoint=endpoint,
            input_file_id=file_id,
        )
        return job.id
