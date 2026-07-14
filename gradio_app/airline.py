"""A multi-turn Gradio chat app: an airline ticket assistant with tool use.

Chat with a concise airline-ticket assistant and ask what a flight costs — the
bot calls a ``get_airline_price`` tool that returns a price in euros. Built on
``gr.ChatInterface`` (which owns the transcript and history) and backed by the
native Anthropic SDK, so replies stream straight from Claude.

This is the sibling of ``gradio_app/chatinterface.py`` — same self-contained
``Bot`` dataclass that owns its own ``Anthropic`` client, on ``gr.ChatInterface``
— but it declares a tool. ``chat`` runs a streaming manual tool loop: it streams a
turn; if Claude stops with ``stop_reason == "tool_use"`` it runs the tool, appends
the ``tool_result``, and streams the next turn. Text accumulates across turns into
one growing string so the UI never resets. The persona (``SYSTEM``) and tool list
(``TOOLS``) live on ``Bot`` as ``ClassVar``s. The bot is built inside
``build_demo`` (not at import), so importing this module has no side effects.

``get_airline_price`` returns dummy data (a small known-city table plus a
deterministic fallback) — there is no real pricing backend. Run it with
``uv run python -m gradio_app.airline``. It reads ``ANTHROPIC_API_KEY`` from
``.env`` (native Claude, like ``gradio_app.chatinterface``).
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import gradio as gr
from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv

# Dummy EUR ticket prices for a handful of destinations. Anything not listed
# gets a deterministic fallback (see get_airline_price) so unknown cities still
# vary and the demo stays reproducible.
_PRICES: dict[str, int] = {
    "london": 120,
    "paris": 95,
    "rome": 140,
    "berlin": 110,
    "new york": 480,
    "tokyo": 720,
    "sydney": 910,
}


def get_airline_price(location: str) -> str:
    """Return a dummy ticket price to ``location``, formatted as ``"<n> EUR"``.

    Known cities come from ``_PRICES`` (case-insensitive). Unknown locations get
    a deterministic fallback derived from the name, so repeated calls are stable
    and the demo needs no real pricing backend. Never raises.
    """
    key = location.strip().lower()
    if key in _PRICES:
        price = _PRICES[key]
    else:
        # Deterministic pseudo-price in the 100–999 range for unknown cities.
        price = 100 + sum(ord(c) for c in key) % 900
    return f"{price} EUR"


TOOL: ToolParam = {
    "name": "get_airline_price",
    "description": (
        "Get the price of an airline ticket to a destination city, in euros "
        "(EUR). Call this whenever the user asks the price or cost of a flight "
        "to a place, or asks how much it is to fly somewhere."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The destination city, e.g. 'London' or 'Tokyo'.",
            },
        },
        "required": ["location"],
    },
}


@dataclass(frozen=True)
class Bot:
    """An airline-ticket chatbot backed by Claude and a price-lookup tool."""

    SYSTEM: ClassVar[str] = (
        "You are a helpful airline ticket assistant. Help customers with flights "
        "and ticket prices. Keep every reply short and concise — a sentence or "
        "two. When a customer asks the price or cost of a flight to a place, call "
        "the get_airline_price tool to look it up; never guess or invent a price. "
        "Prices are in euros (EUR)."
    )
    TOOLS: ClassVar[list[ToolParam]] = [TOOL]

    model: str = "claude-sonnet-5"
    system: str = SYSTEM
    max_tokens: int = 1024
    client: Anthropic = field(default_factory=Anthropic)

    @staticmethod
    def _to_messages(message: str, history: list[dict[str, Any]]) -> list[MessageParam]:
        """Map a Gradio messages-format history plus the new user ``message`` to
        an Anthropic ``messages`` list.

        Gradio history entries are ``{"role": "user"|"assistant", "content": str}``
        dicts — the same shape as ``MessageParam`` — so each maps directly. Only
        ``user``/``assistant`` turns with non-empty content are kept (any
        metadata/tool entries are skipped). The system prompt is *not* included
        here; it is passed separately via ``system=``.
        """
        messages: list[MessageParam] = [
            MessageParam(
                role=cast(Literal["user", "assistant"], turn["role"]),
                content=turn["content"],
            )
            for turn in history
            if turn.get("role") in ("user", "assistant") and turn.get("content")
        ]
        messages.append(MessageParam(role="user", content=message))
        return messages

    def chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]:
        """Stream Claude's reply to ``message``, running the price tool as needed.

        Streams a turn; if Claude stops to call ``get_airline_price``, runs it,
        appends the ``tool_result``, and streams the next turn. ``reply``
        accumulates across every turn so ``gr.ChatInterface`` sees one growing
        string and never resets the displayed text.
        """
        messages = self._to_messages(message, history)
        reply = ""
        while True:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                messages=messages,
                tools=self.TOOLS,
            ) as stream:
                for text in stream.text_stream:
                    reply += text
                    yield reply
                final = stream.get_final_message()

            messages.append(MessageParam(role="assistant", content=final.content))
            if final.stop_reason != "tool_use":
                return

            tool_results: list[ToolResultBlockParam] = []
            for block in final.content:
                if block.type == "tool_use" and block.name == "get_airline_price":
                    args = cast(dict[str, Any], block.input)
                    result = get_airline_price(str(args["location"]))
                    tool_results.append(
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=block.id,
                            content=result,
                        )
                    )
            messages.append(MessageParam(role="user", content=tool_results))


def build_demo() -> gr.Blocks:
    """Build the UI: a multi-turn airline chat streamed from Claude with tool use."""
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        title="llm-engineering — airline tickets",
    )


def launch(**kwargs: Any) -> None:
    """Load environment variables, build the demo, and serve it.

    ``load_dotenv`` runs here (not at import) so ``ANTHROPIC_API_KEY`` is
    available before ``Bot`` builds its ``Anthropic`` client, matching
    ``gradio_app/chatinterface.py``.
    """
    load_dotenv()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
