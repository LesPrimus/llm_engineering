import uuid
from dataclasses import dataclass, field
from datetime import datetime as dt
from enum import StrEnum, auto
from typing import Literal, Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    MESSAGE = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()


class Event(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: dt.now().timestamp())
    author: str  # "user" or agent name
    content: list[ContentItem] = Field(default_factory=list)


class Message(BaseModel):
    """A text message in the conversation."""

    type: EventType = EventType.MESSAGE
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool."""

    type: EventType = EventType.TOOL_CALL
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result from tool execution."""

    type: EventType = EventType.TOOL_RESULT
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


type ContentItem = Message | ToolCall | ToolResult


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[Event] = field(default_factory=list)
    current_step: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    final_result: str | BaseModel | None = None

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1
