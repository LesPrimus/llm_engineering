import json
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion, get_llm_provider, get_model_info

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
        if request.response_format:
            kwargs["response_format"] = request.response_format

        response = await acompletion(
            model=self.model,
            messages=self.to_messages(request),
            num_retries=self.num_retries,
            timeout=self.timeout,
            **kwargs,
            **self.options,
        )
        return self.from_response(response)

    def supports_native_structured_output(self) -> bool:
        """Whether the provider itself holds the answer to a ``response_format``.

        Where it cannot, litellm stands in with a tool the model is forced to
        call on every turn, which leaves the model no way to call any other —
        or, for a model with no support at all, the format may be dropped. So
        only a provider known to enforce it counts, and anything uncertain is
        left to the answer tool, which works wherever tool calling does.
        """
        try:
            _, provider, _, _ = get_llm_provider(self.model)
            info = get_model_info(self.model)
        except Exception:
            # A model litellm does not know is one it cannot vouch for.
            return False
        # OpenAI's API enforces response_format itself; litellm passes it straight through.
        if provider in {"openai", "azure"}:
            return bool(info.get("supports_response_schema"))
        # Elsewhere litellm may fake it with a forced tool, so only its native flag counts.
        return bool(info.get("supports_native_structured_output"))

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
