"""The state of a run, kept for its whole length."""

import uuid
from typing import Any

from pydantic import BaseModel, Field

from .events import Event


class ExecutionContext(BaseModel):
    """Central storage for all execution state."""

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    events: list[Event] = Field(default_factory=list)
    current_step: int = 0
    state: dict[str, Any] = Field(default_factory=dict)
    final_result: str | BaseModel | None = None

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1
