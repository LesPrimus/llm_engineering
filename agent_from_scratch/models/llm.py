"""What goes to the model and what comes back."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_from_scratch.tools.base import BaseTool

from .events import Event


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
