"""A multi-model Gradio chat app for the llm-engineering project.

Pick a bot — GPT, Claude, or DeepSeek — from a dropdown and its reply streams
back. Every bot is reached through an OpenAI client pointed at OpenRouter, so
switching bots is just a different model ID. A ``Model`` enum lists the choices
(member name is the UI label, value is the OpenRouter model ID); a ``Bot`` pairs
a chosen ``Model`` with its own OpenRouter client and a system prompt. The client
factory calls ``load_dotenv`` before reading ``OPENROUTER_API_KEY``, and the bots
are built inside ``build_demo`` (not at import), so importing this module has no
side effects.

This is a separate entry point from ``gradio_app.app`` (the single-Claude app);
run it with ``uv run python -m gradio_app.multibot``.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
)

SYSTEM = "You are a helpful assistant."


class Model(Enum):
    """Selectable bots. Member name is the UI label; value is the OpenRouter ID."""

    GPT = "openai/gpt-4o-mini"
    Claude = "anthropic/claude-sonnet-4.5"
    DeepSeek = "deepseek/deepseek-chat"


@dataclass(frozen=True)
class Bot:
    """A chatbot: one model reached through its own OpenRouter client."""

    @staticmethod
    def _client() -> OpenAI:
        # Load .env here so OPENROUTER_API_KEY is present. The OpenAI SDK's
        # default env var is OPENAI_API_KEY, so point the client at OpenRouter
        # and resolve the key explicitly rather than relying on the SDK fallback.
        load_dotenv()
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    model: Model
    client: OpenAI = field(default_factory=_client)
    system: str = SYSTEM

    def chat(self, message: str) -> Iterator[str]:
        """Stream the model's reply to ``message`` under this bot's system prompt.

        Yields the reply accumulated so far on each text chunk, so Gradio can
        update the output textbox as the response arrives.
        """
        messages: list[Message] = [
            SystemMessage(role="system", content=self.system),
            UserMessage(role="user", content=message),
        ]
        stream = self.client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                reply += delta
                yield reply


def build_demo() -> gr.Blocks:
    """Build the UI: pick a model, send a message, stream the reply back."""
    bots = {model.name: Bot(model) for model in Model}

    def respond(name: str, message: str) -> Iterator[str]:
        yield from bots[name].chat(message)

    return gr.Interface(
        fn=respond,
        inputs=[
            gr.Dropdown(
                choices=[model.name for model in Model],
                value=Model.Claude.name,
                label="Model",
            ),
            gr.Textbox(label="You"),
        ],
        outputs=gr.Textbox(label="Reply"),
        title="llm-engineering",
        flagging_mode="never",
    )


def launch(**kwargs: Any) -> None:
    """Build the demo and serve it.

    Each bot's client factory calls ``load_dotenv`` as it builds, so
    ``OPENROUTER_API_KEY`` is read from ``.env`` without ``launch`` having to
    manage the environment.
    """
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
