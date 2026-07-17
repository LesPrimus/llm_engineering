"""A multimodal Gradio airline-ticket assistant: chat + destination image + voice.

Three boxes — a chat transcript, a generated image of the destination city, and a
spoken (TTS) rendering of every reply. Everything runs through **one** provider: the
OpenAI SDK pointed at **OpenRouter** with a single ``OPENROUTER_API_KEY`` (like
``gradio_app.multibot``). OpenRouter covers all three modalities, so no second client or
key is needed — chat + function calling via ``chat.completions``, voice via
``audio.speech`` (OpenAI-SDK compatible), and images via a raw ``POST /api/v1/images``
(the SDK's ``images.generate`` targets a different path, so ``ImageStudio`` uses
``httpx``).

Four injected ``@dataclass(frozen=True)`` units keep each concern isolated: ``PriceStore``
(SQLite dummy prices — a self-contained copy; this module imports nothing from
``gradio_app.airline``), ``ImageStudio`` (returns a PIL image), ``VoiceStudio`` (streams an
MP3 file and returns its path), and ``Bot``. ``Bot`` owns its chat client and holds the
other three; its ``get_airline_price`` bound method reads the store, and ``respond``
hand-rolls the tool loop — generating a city image when the tool fires and voice for every
reply — yielding one ``(text, image, audio)`` tuple into a ``gr.ChatInterface`` with
``additional_outputs``. ``gr.Image`` takes the PIL image directly; ``gr.Audio`` takes the
``.mp3`` file path (``launch`` allows the temp dir so Gradio can serve it).

Prices are dummy SQLite data (seed cities + a deterministic fallback); there is no real
backend. Importing this module has no side effects. Run it with
``uv run python -m gradio_app.airline_multimodal``; it reads ``OPENROUTER_API_KEY`` from
``.env``.
"""

import base64
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, cast

import gradio as gr
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image

CHAT_MODEL = "openai/gpt-4o-mini"
IMAGE_MODEL = "openai/gpt-image-1"
# OpenRouter serves no OpenAI TTS model (its docs' `openai/gpt-4o-mini-tts-*` returns 400
# "does not exist"), so use Deepgram Aura-2 — fast, and the same voice `gradio_app.chat_voice`
# uses. List/choose others via GET /api/v1/models?output_modalities=speech.
TTS_MODEL = "deepgram/aura-2"
TTS_VOICE = "aura-2-thalia-en"


def _client() -> OpenAI:
    """Build an OpenAI client pointed at OpenRouter.

    ``load_dotenv`` runs here (not at import) so ``OPENROUTER_API_KEY`` is available; the
    OpenAI SDK's default env var is ``OPENAI_API_KEY``, so the key is resolved explicitly
    and the base URL is repointed at OpenRouter (like ``multibot``).
    """
    load_dotenv()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


@dataclass(frozen=True)
class PriceStore:
    """SQLite-backed store of dummy EUR ticket prices, injected into ``Bot``.

    A self-contained copy of the design in ``gradio_app.airline`` (this module imports
    nothing from it). Owns every database concern — path, connection, seeding, lookup.
    ``price`` is a pure reader; ``ensure_seeded`` (called by ``launch``) is the only
    writer.
    """

    SEED: ClassVar[dict[str, int]] = {
        "london": 120,
        "paris": 95,
        "rome": 140,
        "berlin": 110,
        "new york": 480,
        "tokyo": 720,
        "sydney": 910,
    }
    DEFAULT_PATH: ClassVar[Path] = Path(__file__).with_name("airline_prices.db")

    path: Path = DEFAULT_PATH

    def _connect(self) -> sqlite3.Connection:
        """Open one connection per call (SQLite's same-thread rule; Gradio threads)."""
        return sqlite3.connect(self.path)

    def ensure_seeded(self) -> None:
        """Create and seed the ``prices`` table if missing or empty (idempotent)."""
        with closing(self._connect()) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price INTEGER)"
            )
            if conn.execute("SELECT 1 FROM prices LIMIT 1").fetchone() is None:
                conn.executemany(
                    "INSERT OR IGNORE INTO prices (city, price) VALUES (?, ?)",
                    list(self.SEED.items()),
                )
                conn.commit()

    def price(self, location: str) -> str:
        """Return a dummy ticket price to ``location`` as ``"<n> EUR"`` (pure reader).

        Looks the lowercased/stripped city up in the seeded ``prices`` table; unknown
        cities get a deterministic char-sum fallback so repeats are stable. Assumes the
        table exists (``launch`` seeds first).
        """
        key = location.strip().lower()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT price FROM prices WHERE city = ?", (key,)
            ).fetchone()
        price = row[0] if row is not None else 100 + sum(ord(c) for c in key) % 900
        return f"{price} EUR"


@dataclass(frozen=True)
class ImageStudio:
    """Generates a destination image via OpenRouter's Image API (``POST /images``).

    Owns its own OpenRouter client (for auth + base URL). The OpenAI SDK's
    ``images.generate`` posts to ``/images/generations``, which is *not* OpenRouter's
    ``/api/v1/images`` endpoint, so ``image`` makes the raw POST with ``httpx`` and reuses
    the client's key and base URL. Returns a PIL image — ``gr.Image`` accepts one
    directly, so no temp file.
    """

    client: OpenAI = field(default_factory=_client)
    model: str = IMAGE_MODEL

    @staticmethod
    def _decode(b64_json: str) -> Image.Image:
        """Decode a base64 image payload into a PIL image."""
        return Image.open(BytesIO(base64.b64decode(b64_json)))

    def image(self, city: str) -> Image.Image:
        """Generate a scenic travel image of ``city`` and return it as a PIL image."""
        response = httpx.post(
            f"{str(self.client.base_url).rstrip('/')}/images",
            headers={"Authorization": f"Bearer {self.client.api_key}"},
            json={
                "model": self.model,
                "prompt": f"A scenic travel photograph of {city}.",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return self._decode(response.json()["data"][0]["b64_json"])


@dataclass(frozen=True)
class VoiceStudio:
    """Renders text to speech via OpenRouter's OpenAI-compatible ``audio.speech`` API.

    Owns its own OpenRouter client. ``speech`` streams the MP3 to a temp ``.mp3`` file
    (via ``with_streaming_response`` / ``stream_to_file``, per OpenRouter's TTS guide) and
    returns the file path. A real file with a ``.mp3`` suffix plays reliably in
    ``gr.Audio``; raw bytes rely on Gradio sniffing the container, which fails for MP3
    frames that carry no ID3 header.
    """

    client: OpenAI = field(default_factory=_client)
    model: str = TTS_MODEL
    voice: str = TTS_VOICE

    def talker(self, message):
        response = self.client.audio.speech.create(
            model=self.model, voice=self.voice, input=message
        )
        return response.content

    # def speech(self, text: str) -> str:
    #     """Synthesize ``text`` to an MP3 file and return its path."""
    #     fd, path = tempfile.mkstemp(suffix=".mp3")
    #     os.close(fd)
    #     with self.client.audio.speech.with_streaming_response.create(
    #         model=self.model,
    #         voice=self.voice,
    #         input=text,
    #         response_format="mp3",
    #     ) as response:
    #         response.stream_to_file(path)
    #     return path


@dataclass(frozen=True)
class Bot:
    """Airline-ticket assistant: chat + a price tool, plus image and voice output."""

    SYSTEM: ClassVar[str] = (
        "You are a helpful airline ticket assistant. On the customer's first message, "
        "briefly introduce yourself and ask: 'How can I help you today for your trip?' "
        "Keep every reply short and concise — a sentence or two. Whenever the customer "
        "names a destination they want to fly to (for example 'I would like to go to "
        "London' or 'how much is a flight to Tokyo?'), call the get_airline_price tool "
        "for that city and tell them the price. Never guess or invent a price. Prices "
        "are in euros (EUR)."
    )

    TOOL: ClassVar[ChatCompletionFunctionToolParam] = {
        "type": "function",
        "function": {
            "name": "get_airline_price",
            "description": (
                "Get the price of an airline ticket to a destination city, in euros "
                "(EUR). Call this whenever the user asks the price or cost of a flight "
                "to a place, or asks how much it is to fly somewhere."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The destination city, e.g. 'London' or 'Tokyo'.",
                    }
                },
                "required": ["location"],
            },
        },
    }

    model: str = CHAT_MODEL
    system: str = SYSTEM
    max_tokens: int = 1024
    client: OpenAI = field(default_factory=_client)
    prices: PriceStore = field(default_factory=PriceStore)
    images: ImageStudio = field(default_factory=ImageStudio)
    voice: VoiceStudio = field(default_factory=VoiceStudio)

    def get_airline_price(self, location: str) -> str:
        """Return the ticket price to ``location`` (bound tool → injected store)."""
        return self.prices.price(location)

    def _to_messages(
        self, message: str, history: list[dict[str, Any]]
    ) -> list[ChatCompletionMessageParam]:
        """Map the system prompt + Gradio ``messages``-format history + new user turn to
        an OpenAI ``messages`` list. Only ``user``/``assistant`` turns with content kept.
        """
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=self.system)
        ]
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role == "user" and content:
                messages.append(
                    ChatCompletionUserMessageParam(role="user", content=content)
                )
            elif role == "assistant" and content:
                messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {"role": "assistant", "content": content},
                    )
                )
        messages.append(ChatCompletionUserMessageParam(role="user", content=message))
        return messages

    def _safe_image(self, city: str) -> Any:
        """Generate a city image; on any failure log it and return ``gr.skip()``.

        A failed image must never break the chat, but the error is printed to stderr
        (not silently swallowed) so a misconfigured model surfaces during a run.
        """
        try:
            return self.images.image(city)
        except Exception as exc:
            print(
                f"[airline_multimodal] image generation failed: {exc}", file=sys.stderr
            )
            return gr.skip()

    def _safe_speech(self, text: str) -> Any:
        """Synthesize voice; on any failure log it and return ``gr.skip()``.

        Same contract as ``_safe_image``: the chat continues, but the error is printed to
        stderr so a bad TTS model/voice is visible instead of a silently empty box.
        """
        try:
            return self.voice.talker(text)
        except Exception as exc:
            print(
                f"[airline_multimodal] speech synthesis failed: {exc}", file=sys.stderr
            )
            return gr.skip()

    def respond(
        self, message: str, history: list[dict[str, Any]]
    ) -> Iterator[tuple[str, Any, Any]]:
        """Answer ``message``, running the price tool as needed; show a city image and speak.

        Yields one ``(chat_text, image_value, audio_value)`` tuple for ``gr.ChatInterface`` +
        its image and audio ``additional_outputs``. **Every** reply is spoken. The **image is
        generated only when the price tool fires** (a destination city is then known); on the
        plain-text path ``image_value`` is ``gr.skip()``, leaving the image box unchanged.
        Both media calls are wrapped (``_safe_image``/``_safe_speech``) so a failure yields
        ``gr.skip()`` and never breaks the chat.
        """
        messages = self._to_messages(message, history)
        first = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=[self.TOOL],
        )
        choice = first.choices[0].message

        image_value: Any = gr.skip()
        if choice.tool_calls:
            messages.append(
                cast(ChatCompletionMessageParam, choice.model_dump(exclude_none=True))
            )
            location = ""
            for call in choice.tool_calls:
                if call.type != "function":
                    continue
                location = json.loads(call.function.arguments).get("location", "")
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=call.id,
                        content=self.get_airline_price(location),
                    )
                )
            # A destination city is now known — generate an image of it.
            if location:
                image_value = self._safe_image(location)
            followup = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
            )
            final_text = followup.choices[0].message.content or ""
        else:
            final_text = choice.content or ""

        # One tuple: reply text, the destination image (or skip), and the spoken reply.
        yield final_text, image_value, self._safe_speech(final_text)


def build_demo(bot: Bot | None = None) -> gr.Blocks:
    """Build the chat + image + voice UI: chat (left), destination image + voice (right).

    The ``image`` and ``audio`` components are ``additional_outputs`` of the
    ``gr.ChatInterface``, so they must exist in the ``Blocks`` context — each is defined with
    ``render=False`` and ``.render()``-ed into the right column, stacked. Their order matches
    the trailing elements of ``respond``'s yielded tuple (``image`` then ``audio``). Accepts an
    optional pre-built ``Bot`` (so ``launch`` can seed its store); defaults to a fresh ``Bot``.
    """
    bot = bot if bot is not None else Bot()
    with gr.Blocks(title="llm-engineering — airline (voice + image)") as demo:
        gr.Markdown("# Airline ticket assistant — chat, image & voice")
        image = gr.Image(label="Destination", render=False)
        audio = gr.Audio(label="Voice", autoplay=True, render=False)
        with gr.Row():
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=bot.respond,
                    additional_outputs=[image, audio],
                    title="Chat",
                )
            with gr.Column(scale=1):
                image.render()
                audio.render()
    return demo


def launch(**kwargs: Any) -> None:
    """Load env, seed the price DB, build the demo, and serve.

    ``load_dotenv`` runs (also inside ``_client``); the bot's ``PriceStore`` is seeded
    before serving so the first request reads a ready table. ``allowed_paths`` lets Gradio
    serve the temp ``.mp3`` files ``VoiceStudio`` writes (a caller may override it).
    """
    load_dotenv()
    bot = Bot()
    bot.prices.ensure_seeded()
    kwargs.setdefault("allowed_paths", [tempfile.gettempdir()])
    build_demo(bot).launch(**kwargs)


if __name__ == "__main__":
    launch()
