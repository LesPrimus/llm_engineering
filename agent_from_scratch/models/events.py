"""What happens during a run: messages, tool calls and their results."""

import uuid
from datetime import datetime as dt
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: dt.now().timestamp())
    # "user" or agent name. The LLM client can't know the agent, so the agent
    # stamps its own events when it records them.
    author: str | None = None


class Message(BaseEvent):
    """A text message in the conversation."""

    type: Literal["message"] = "message"
    role: Literal["user", "assistant"]
    content: str


class ToolCall(BaseEvent):
    """LLM's request to execute a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseEvent):
    """Result from tool execution."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


type Event = Annotated[Message | ToolCall | ToolResult, Field(discriminator="type")]
