"""A Gradio app that summarizes a company's website in Markdown.

Paste a company URL and pick a model — GPT, Claude, or DeepSeek — from a
dropdown. The page's readable text is fetched by ``helpers.Website`` (main page
only, no crawling) and a short Markdown summary streams back into a live
``gr.Markdown`` output. Every model is reached through an OpenAI client pointed
at OpenRouter, so switching models is just a different model ID.

This mirrors ``gradio_app/multibot.py``: a ``Model`` enum lists the choices
(member name is the UI label, value is the OpenRouter model ID) and a
``Summarizer`` dataclass pairs a chosen ``Model`` with its own OpenRouter client
and a summarization system prompt. A ``SummarizerApp`` owns one ``Summarizer``
per model and serves the UI; its summarizers (and their clients) are built when
the app is instantiated, not at import, so importing this module has no side
effects.

Run it with ``uv run python -m gradio_app.website_summarizer``.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import gradio as gr
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
)

from helpers import Website

SYSTEM = (
    "You analyze the landing page of a company website and write a short "
    "summary in Markdown. Ignore navigation, cookie banners, and other "
    "boilerplate. Cover what the company does, its products or services, and "
    "any notable news if present. Respond in Markdown."
)


class Model(Enum):
    """Selectable models. Member name is the UI label; value is the OpenRouter ID."""

    GPT = "openai/gpt-4o-mini"
    Claude = "anthropic/claude-sonnet-4.5"
    DeepSeek = "deepseek/deepseek-chat"


@dataclass(frozen=True)
class Summarizer:
    """One model, reached through its own OpenRouter client, that summarizes a page."""

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

    def summarize(self, website: Website) -> Iterator[str]:
        """Stream a Markdown summary of ``website``.

        Yields the summary accumulated so far on each text chunk, so Gradio can
        render the Markdown output as the response arrives.
        """
        user = (
            f"Company website: {website.url}\n"
            f"Page title: {website.title}\n\n"
            f"Page contents:\n{website.text}\n\n"
            "Summarize this company in Markdown."
        )
        messages: list[Message] = [
            SystemMessage(role="system", content=self.system),
            UserMessage(role="user", content=user),
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


class SummarizerApp:
    """The website-summarizer Gradio app: one ``Summarizer`` per model, plus the UI.

    Summarizers (and their OpenRouter clients) are built in ``__init__``, so
    constructing an app reads ``OPENROUTER_API_KEY`` from ``.env`` — but only
    when instantiated (e.g. under ``if __name__ == "__main__"``), never at import.
    """

    def __init__(self) -> None:
        self.summarizers = {model.name: Summarizer(model) for model in Model}

    def respond(self, name: str, url: str) -> Iterator[str]:
        """Fetch ``url`` (main page only) and stream the chosen model's summary."""
        url = url.strip()
        if not url:
            yield "Enter a company website URL to summarize."
            return
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        try:
            website = Website.fetch(url)
        except httpx.HTTPError as error:
            yield f"**Couldn't fetch {url}:** {error}"
            return
        yield from self.summarizers[name].summarize(website)

    def build_demo(self) -> gr.Blocks:
        """Build the UI: pick a model, enter a company URL, stream a Markdown summary."""
        return gr.Interface(
            fn=self.respond,
            inputs=[
                gr.Dropdown(
                    choices=[model.name for model in Model],
                    value=Model.Claude.name,
                    label="Model",
                ),
                gr.Textbox(label="Company website URL"),
            ],
            outputs=gr.Markdown(label="Summary"),
            title="llm-engineering — website summarizer",
            flagging_mode="never",
        )

    def launch(self, **kwargs: Any) -> None:
        """Build the demo and serve it.

        Each summarizer's client factory calls ``load_dotenv`` as it builds, so
        ``OPENROUTER_API_KEY`` is read from ``.env`` without ``launch`` having to
        manage the environment.
        """
        self.build_demo().launch(**kwargs)


if __name__ == "__main__":
    SummarizerApp().launch()
