import uuid
from datetime import datetime as dt
from enum import StrEnum, auto
from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, Field

from agent_from_scratch.tools.base import BaseTool


class EventType(StrEnum):
    MESSAGE = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()


class Role(StrEnum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()


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
    role: Role
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


class LlmRequest(BaseModel):
    """Request object for LLM calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: list[str] = Field(default_factory=list)
    contents: list[ContentItem] = Field(default_factory=list)
    tools: list[BaseTool] = Field(default_factory=list)
    tool_choice: str | None = None


class LlmResponse(BaseModel):
    """Response object from LLM calls."""

    content: list[ContentItem] = Field(default_factory=list)
    error_message: str | None = None
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
