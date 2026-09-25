import json
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion

from agent_from_scratch.models import (
    Event,
    LlmRequest,
    LlmResponse,
    Message,
    ToolCall,
    ToolResult,
)


@dataclass(frozen=True)
class LlmClient:
    """Calls a model through litellm, in OpenAI's chat format whatever the provider."""

    model: str = "gpt-5-mini"
    num_retries: int = 3
    timeout: float = 600.0
    # Anything else acompletion takes, such as temperature or reasoning_effort.
    options: dict[str, Any] = field(default_factory=dict)

    async def generate(self, request: LlmRequest) -> LlmResponse:
        kwargs: dict[str, Any] = {}
        # Some providers reject an empty tool list, so none is sent at all.
        if request.tools:
            kwargs["tools"] = [tool.tool_definition for tool in request.tools]
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice

        response = await acompletion(
            model=self.model,
            messages=self.to_messages(request),
            num_retries=self.num_retries,
            timeout=self.timeout,
            **kwargs,
            **self.options,
        )
        return self.from_response(response)

    @classmethod
    def to_messages(cls, request: LlmRequest) -> list[dict[str, Any]]:
        """The request as OpenAI chat messages: instructions first, then the events.

        Events are flat, but the API wants the tool calls of one turn on the
        assistant message that makes them, so each call joins the assistant message
        before it, or starts one if the model called without saying anything.
        """
        messages: list[dict[str, Any]] = []
        if request.instructions:
            messages.append(
                {"role": "system", "content": "\n\n".join(request.instructions)}
            )
        for event in request.contents:
            match event:
                case Message(role=role, content=content):
                    messages.append({"role": role, "content": content})
                case ToolCall():
                    if not messages or messages[-1]["role"] != "assistant":
                        messages.append({"role": "assistant", "content": None})
                    messages[-1].setdefault("tool_calls", []).append(
                        {
                            "id": event.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": event.name,
                                "arguments": json.dumps(event.arguments),
                            },
                        }
                    )
                case ToolResult():
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": event.tool_call_id,
                            "content": cls.tool_result_text(event),
                        }
                    )
        return messages

    @staticmethod
    def tool_result_text(result: ToolResult) -> str:
        """A tool's output as the text the model reads, flagged when the call failed."""
        text = "\n".join(
            item if isinstance(item, str) else json.dumps(item, default=str)
            for item in result.content
        )
        return f"Error: {text}" if result.status == "error" else text

    @staticmethod
    def from_response(response: Any) -> LlmResponse:
        """The model's reply as events: its text, if any, then its tool calls."""
        message = response.choices[0].message
        content: list[Event] = []
        if message.content:
            content.append(Message(role="assistant", content=message.content))
        for call in message.tool_calls or []:
            content.append(
                ToolCall(
                    tool_call_id=call.id,
                    name=call.function.name,
                    arguments=json.loads(call.function.arguments or "{}"),
                )
            )
        usage = getattr(response, "usage", None)
        return LlmResponse(
            content=content, usage_metadata=usage.model_dump() if usage else {}
        )
