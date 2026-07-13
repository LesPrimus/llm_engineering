"""A multi-turn Gradio chat app for an online guitar store.

Chat with a friendly guitar-store sales assistant: ask for recommendations by
budget, skill, and style, and follow up — the bot sees the whole conversation.
Built on ``gr.ChatInterface`` (which owns the transcript and history) and backed
by the native Anthropic SDK, so replies stream straight from Claude.

This is the sibling of ``gradio_app/app.py`` — same self-contained ``Bot``
dataclass that owns its own ``Anthropic`` client — but on ``gr.ChatInterface``
(multi-turn) rather than the stateless ``gr.Interface``. ``gr.ChatInterface``
calls ``fn(message, history)``, and the history is a list of ``{"role",
"content"}`` dicts, which map 1:1 to Anthropic ``MessageParam``s. The
guitar-store persona lives on ``Bot`` as a ``ClassVar`` and the ``system`` field
defaults to it. The bot is built inside ``build_demo`` (not at import), so
importing this module has no side effects.

Run it with ``uv run python -m gradio_app.chatinterface``. It reads
``ANTHROPIC_API_KEY`` from ``.env`` (native Claude, like ``gradio_app.app``).
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import gradio as gr
from anthropic import Anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv


@dataclass(frozen=True)
class Bot:
    """A guitar-store chatbot backed by a single Claude model and a persona."""

    SYSTEM: ClassVar[str] = (
        "You are a friendly, knowledgeable assistant for an online guitar store. "
        "Help customers choose acoustic, electric, and bass guitars and related "
        "gear (amps, pedals, strings, accessories) based on their budget, skill "
        "level, and musical style. Explain differences — tonewoods, body shapes, "
        "pickups, amp types — in approachable, encouraging terms, and ask a "
        "clarifying question when it would help you recommend better. Keep replies "
        "concise. You don't have live access to prices or stock, so don't invent "
        "them — say so and suggest what specs or options to look for instead."
    )

    model: str = "claude-sonnet-5"
    system: str = SYSTEM
    max_tokens: int = 1024
    client: Anthropic = field(default_factory=Anthropic)

    @staticmethod
    def _to_messages(message: str, history: list[dict[str, Any]]) -> list[MessageParam]:
        """Map a Gradio messages-format history plus the new user ``message``
        to an Anthropic ``messages`` list.

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
        """Stream Claude's reply to ``message`` given the prior ``history``.

        Yields the reply accumulated so far on each text chunk, so
        ``gr.ChatInterface`` can render the response as it arrives.
        """
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self._to_messages(message, history),
        ) as stream:
            reply = ""
            for text in stream.text_stream:
                reply += text
                yield reply


def build_demo() -> gr.Blocks:
    """Build the UI: a multi-turn guitar-store chat streamed from Claude."""
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        title="llm-engineering — guitar store",
    )


def launch(**kwargs: Any) -> None:
    """Load environment variables, build the demo, and serve it.

    ``load_dotenv`` runs here (not at import) so ``ANTHROPIC_API_KEY`` is
    available before ``Bot`` builds its ``Anthropic`` client, matching
    ``gradio_app/app.py``.
    """
    load_dotenv()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
