import uuid
from datetime import datetime as dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_from_scratch.tools.base import BaseTool


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


class LlmRequest(BaseModel):
    """Request object for LLM calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: list[str] = Field(default_factory=list)
    contents: list[Event] = Field(default_factory=list)
    tools: list[BaseTool] = Field(default_factory=list)
    tool_choice: str | None = None


class LlmResponse(BaseModel):
    """Response object from LLM calls."""

    content: list[Event] = Field(default_factory=list)
    error_message: str | None = None
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
