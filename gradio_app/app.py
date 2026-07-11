"""A minimal Gradio chat app for the llm-engineering project.

A ``Bot`` pairs a Claude model with a system prompt and its own Anthropic
client; ``chat`` takes a message and streams Claude's reply, wired to an input
textbox and an output textbox. ``build_demo`` constructs the bot inside a
factory (not at import) so ``load_dotenv`` has run before the client reads
``ANTHROPIC_API_KEY``.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
from anthropic import Anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv


@dataclass(frozen=True)
class Bot:
    """A chatbot backed by a single Claude model and a system prompt."""

    model: str = "claude-sonnet-5"
    system: str = "You are a helpful assistant."
    max_tokens: int = 1024
    client: Anthropic = field(default_factory=Anthropic)

    def chat(self, message: str) -> Iterator[str]:
        """Stream Claude's reply to ``message`` under this bot's system prompt.

        Yields the reply accumulated so far on each text chunk, so Gradio can
        update the output textbox as the response arrives.
        """
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=[MessageParam(role="user", content=message)],
        ) as stream:
            reply = ""
            for text in stream.text_stream:
                reply += text
                yield reply


def build_demo() -> gr.Blocks:
    """Build the UI: a message in, Claude's reply out."""
    bot = Bot()
    return gr.Interface(
        fn=bot.chat,
        inputs=gr.Textbox(label="You"),
        outputs=gr.Textbox(label="Claude"),
        title="llm-engineering",
        flagging_mode="never",
    )


def launch(**kwargs: Any) -> None:
    """Load environment variables, build the demo, and serve it.

    ``load_dotenv`` runs here (not at import) so ``ANTHROPIC_API_KEY`` is
    available before ``Bot`` builds its Anthropic client, matching how
    ``main.py`` bootstraps the project.
    """
    load_dotenv()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
