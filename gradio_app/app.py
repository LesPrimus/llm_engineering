"""A minimal Gradio chat app for the llm-engineering project.

A ``Bot`` wraps a Claude model together with its Anthropic client; ``chat``
takes a message and returns Claude's reply, wired to an input textbox and an
output textbox. ``build_demo`` constructs the bot inside a factory (not at
import) so ``load_dotenv`` has run before the client reads ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from claude import ClaudeClient


@dataclass(frozen=True)
class Bot:
    """A chatbot backed by a single Claude model."""

    model: str = "claude-sonnet-5"
    client: ClaudeClient = field(default_factory=ClaudeClient)

    def chat(self, message: str) -> str:
        """Send ``message`` to Claude and return its reply."""
        return self.client.ask(self.model, message)


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
    available before ``Bot`` builds its client, matching how ``main.py``
    bootstraps the project.
    """
    load_dotenv()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
