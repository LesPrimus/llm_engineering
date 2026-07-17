# Airline Multimodal (Image + Voice) Chat Implementation Plan

> **Status (2026-07-17): implemented directly, not via the task/test flow below.** The module
> `gradio_app/airline_multimodal.py` was written in one pass (no smoke-check tasks) at the user's
> request, verified with `ruff` + `mypy` + live API runs. Two design points changed during
> implementation and bug-fixing, so the delivered code is authoritative over the task code blocks:
> (1) **TTS model** — OpenRouter exposes no OpenAI TTS model, so `TTS_MODEL = "hexgrad/kokoro-82m"`,
> `TTS_VOICE = "af_heart"` (mp3-capable). (2) **`respond` yields a single** `(text, image, audio)`
> tuple with audio present on **every** reply (not the staged two-yield form the tasks show). The
> tool trigger was also broadened to fire on any named destination (e.g. "I would like to go to
> London"), not only explicit price questions. See the design spec for the reconciled description.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained Gradio module `gradio_app/airline_multimodal.py` — an airline-ticket assistant with a three-box UI (chat, destination image, voice) where every reply is spoken and a city image is generated whenever the price tool fires.

**Architecture:** One `OpenAI` SDK client pointed at OpenRouter (single `OPENROUTER_API_KEY`) serves all three modalities: chat + function calling (`chat.completions`), TTS (`audio.speech`), and image generation (raw `POST /api/v1/images` via `httpx`, because the SDK's `images.generate()` targets a different path). Four injected `@dataclass(frozen=True)` units — `PriceStore` (SQLite prices), `ImageStudio`, `VoiceStudio`, `Bot` — keep each concern isolated and fake-injectable. `Bot.respond` hand-rolls the tool loop and yields staged `(text, image, audio)` tuples into a `gr.ChatInterface` with `additional_outputs`.

**Tech Stack:** Python 3.14, `openai` SDK (pointed at OpenRouter), `httpx`, `gradio` 6, `Pillow`, `python-dotenv`, stdlib `sqlite3`/`base64`/`json`.

## Global Constraints

- **Provider:** OpenAI SDK with `base_url="https://openrouter.ai/api/v1"`; key from `os.environ["OPENROUTER_API_KEY"]` (via `load_dotenv()`). One provider, one key.
- **Models (fixed constants, no dropdown):** `CHAT_MODEL = "openai/gpt-4o-mini"`, `IMAGE_MODEL = "openai/gpt-image-1"`, `TTS_MODEL = "hexgrad/kokoro-82m"`, `TTS_VOICE = "af_heart"` (OpenRouter exposes no OpenAI TTS model; Kokoro is cheap and mp3-capable).
- **Self-contained:** no imports from `gradio_app.airline` or any sibling module. `PriceStore` is re-implemented locally.
- **No side effects at import:** `_client()` (reads the env var) and all network/DB/file work happen only on instantiation or in `launch()` — never at module top level.
- **No test framework in this repo.** Verification is `uv run python` import/run **smoke checks** (with injected fakes), plus `ruff` + `mypy`, plus a manual run. Mirrors `gradio_app/airline.py`.
- **Run command:** `uv run python -m gradio_app.airline_multimodal`.
- **Behavior:** voice is generated for **every** bot reply; the image is generated **only** when the `get_airline_price` tool fires. A media (image/TTS) failure must never break the chat.
- **Git:** commit to `master`; **no** `Co-Authored-By` trailer (repo convention).
- **DB file:** `gradio_app/airline_prices.db` (already gitignored — do not commit it). Same schema/seed as `airline.py`; sharing the file is safe because seeding is idempotent.

---

### Task 1: Module scaffold, constants, `_client()`, and `PriceStore`

**Files:**
- Create: `gradio_app/airline_multimodal.py`

**Interfaces:**
- Produces:
  - `_client() -> OpenAI` — OpenAI client pointed at OpenRouter.
  - Constants `CHAT_MODEL`, `IMAGE_MODEL`, `TTS_MODEL`, `TTS_VOICE` (all `str`).
  - `PriceStore` dataclass: `ensure_seeded() -> None`, `price(location: str) -> str`, `path: Path` field, `SEED`/`DEFAULT_PATH` ClassVars.

- [ ] **Step 1: Write the module with the header, constants, `_client()`, and `PriceStore`.**

Create `gradio_app/airline_multimodal.py`:

```python
"""A multimodal Gradio airline-ticket assistant: chat + destination image + voice.

Three boxes — a chat transcript, a generated image of the destination city, and a
spoken (TTS) rendering of every reply. Everything runs through **one** provider: the
OpenAI SDK pointed at **OpenRouter** with a single ``OPENROUTER_API_KEY`` (like
``gradio_app.multibot``). OpenRouter now covers all three modalities, so no second
client or key is needed — chat + function calling via ``chat.completions``, voice via
``audio.speech`` (OpenAI-SDK compatible), and images via a raw ``POST /api/v1/images``
(the SDK's ``images.generate`` targets a different path, so ``ImageStudio`` uses
``httpx``).

Four injected ``@dataclass(frozen=True)`` units keep each concern isolated and
fake-injectable: ``PriceStore`` (SQLite dummy prices — a self-contained copy, this
module imports nothing from ``gradio_app.airline``), ``ImageStudio``, ``VoiceStudio``,
and ``Bot``. ``Bot`` owns its chat client and holds the other three; its
``get_airline_price`` bound method reads the store, and ``respond`` hand-rolls the tool
loop, generating a city image when the tool fires and voice for every reply, yielding
staged ``(text, image, audio)`` tuples into a ``gr.ChatInterface`` with
``additional_outputs``.

Prices are dummy SQLite data (seed cities + a deterministic fallback); there is no real
backend. Importing this module has no side effects. Run it with
``uv run python -m gradio_app.airline_multimodal``; it reads ``OPENROUTER_API_KEY`` from
``.env``.
"""

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from openai import OpenAI

CHAT_MODEL = "openai/gpt-4o-mini"
IMAGE_MODEL = "openai/gpt-image-1"
TTS_MODEL = "hexgrad/kokoro-82m"
TTS_VOICE = "af_heart"


def _client() -> OpenAI:
    """Build an OpenAI client pointed at OpenRouter.

    ``load_dotenv`` runs here (not at import) so ``OPENROUTER_API_KEY`` is available;
    the OpenAI SDK's default env var is ``OPENAI_API_KEY``, so the key is resolved
    explicitly and the base URL is repointed at OpenRouter (like ``multibot``).
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
        table exists (``launch`` seeds first) — a read against an unseeded DB raises
        ``sqlite3.OperationalError``.
        """
        key = location.strip().lower()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT price FROM prices WHERE city = ?", (key,)
            ).fetchone()
        price = row[0] if row is not None else 100 + sum(ord(c) for c in key) % 900
        return f"{price} EUR"
```

- [ ] **Step 2: Write the `PriceStore` smoke check and run it to confirm it FAILS first.**

Before implementing (if doing strict TDD, temporarily comment out the `PriceStore` body), run against a temp DB. Save this as the check you re-run in Step 3:

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run python - <<'PY'
import tempfile, sqlite3
from pathlib import Path
from gradio_app.airline_multimodal import PriceStore

tmp = Path(tempfile.mkdtemp()) / "t.db"
store = PriceStore(path=tmp)

# unseeded read raises (documents launch-seeds-first)
try:
    store.price("london")
    raise SystemExit("FAIL: expected OperationalError on unseeded DB")
except sqlite3.OperationalError:
    pass

store.ensure_seeded()
store.ensure_seeded()  # idempotent
assert store.price("London") == "120 EUR", store.price("London")
assert store.price("  tokyo ") == "720 EUR"
fb1 = store.price("Atlantis")
fb2 = store.price("Atlantis")
assert fb1 == fb2 and fb1.endswith(" EUR"), fb1  # deterministic fallback
print("PriceStore OK", store.price("London"), fb1)
PY
```
Expected before implementation: `ImportError`/`AttributeError` (FAIL). After Step 1: PASS printing e.g. `PriceStore OK 120 EUR 693 EUR`.

- [ ] **Step 3: Run the smoke check to verify it PASSES.**

Run the Step 2 command. Expected: `PriceStore OK 120 EUR <n> EUR`.

- [ ] **Step 4: Confirm the module imports with no side effects and passes lint/type checks.**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run python -c "import gradio_app.airline_multimodal; print('import OK (no key needed)')"
uv run ruff format gradio_app/airline_multimodal.py
uv run ruff check gradio_app/airline_multimodal.py
uv run mypy gradio_app/airline_multimodal.py
```
Expected: `import OK (no key needed)` (no `OPENROUTER_API_KEY` required at import); `ruff check` prints `All checks passed!`; `mypy` prints `Success`.

> **Incremental imports:** each task adds only the imports it needs (Tasks 2–4 begin by extending the import block), so every task commits lint-clean. Do **not** paste the full import set up front — `ruff` would flag the still-unused names as `F401`.

- [ ] **Step 5: Commit.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
git add gradio_app/airline_multimodal.py
git commit -m "Add airline_multimodal scaffold with PriceStore (SQLite prices)"
```

---

### Task 2: `ImageStudio` — destination image via OpenRouter Image API

**Files:**
- Modify: `gradio_app/airline_multimodal.py` (add `ImageStudio` after `PriceStore`)

**Interfaces:**
- Consumes: `_client`, `IMAGE_MODEL`.
- Produces: `ImageStudio` dataclass with `client: OpenAI` field, `_decode(b64_json: str) -> Image.Image` (staticmethod, pure), `image(city: str) -> Image.Image`.

- [ ] **Step 1: Extend imports, then add `ImageStudio`.**

Add these imports to the existing import block (alphabetical within their groups):

```python
import base64
from io import BytesIO

import httpx
from PIL import Image
```

Insert after `PriceStore`:

```python
@dataclass(frozen=True)
class ImageStudio:
    """Generates a destination image via OpenRouter's Image API (``POST /images``).

    Owns its own OpenRouter client (for auth + base URL). The OpenAI SDK's
    ``images.generate`` posts to ``/images/generations``, which is *not* OpenRouter's
    ``/api/v1/images`` endpoint, so ``image`` makes the raw POST with ``httpx`` and
    reuses the client's key and base URL. Returns a PIL image (no temp file needed —
    ``gr.Image`` accepts a PIL image directly).
    """

    client: OpenAI = field(default_factory=_client)
    model: str = IMAGE_MODEL

    @staticmethod
    def _decode(b64_json: str) -> Image.Image:
        """Decode a base64 image payload into a PIL image (pure, offline-testable)."""
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
```

- [ ] **Step 2: Write the `_decode` smoke check and run it (expect FAIL before Step 1).**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run python - <<'PY'
import base64
from io import BytesIO
from PIL import Image
from gradio_app.airline_multimodal import ImageStudio

# make a tiny 2x2 PNG, base64-encode it, then decode via ImageStudio._decode
buf = BytesIO()
Image.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

img = ImageStudio._decode(b64)
assert isinstance(img, Image.Image), type(img)
assert img.size == (2, 2), img.size
print("ImageStudio._decode OK", img.size, img.mode)
PY
```
Expected after Step 1: `ImageStudio._decode OK (2, 2) RGB`. (The networked `image()` path is exercised in the final manual run.)

- [ ] **Step 3: Lint + type check.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run ruff format gradio_app/airline_multimodal.py
uv run ruff check gradio_app/airline_multimodal.py
uv run mypy gradio_app/airline_multimodal.py
```
Expected: all pass.

- [ ] **Step 4: Commit.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
git add gradio_app/airline_multimodal.py
git commit -m "Add ImageStudio (OpenRouter image generation via httpx)"
```

---

### Task 3: `VoiceStudio` — text-to-speech via OpenRouter

**Files:**
- Modify: `gradio_app/airline_multimodal.py` (add `VoiceStudio` after `ImageStudio`)

**Interfaces:**
- Consumes: `_client`, `TTS_MODEL`, `TTS_VOICE`.
- Produces: `VoiceStudio` dataclass with `client: OpenAI` field and `speech(text: str) -> bytes` (returns raw MP3 bytes — `gr.Audio` accepts bytes directly).

- [ ] **Step 1: Add `VoiceStudio` (no new imports needed).**

Insert after `ImageStudio`:

```python
@dataclass(frozen=True)
class VoiceStudio:
    """Renders text to speech via OpenRouter's OpenAI-compatible ``audio.speech`` API.

    Owns its own OpenRouter client. ``speech`` returns the raw MP3 ``bytes``:
    ``gr.Audio`` accepts bytes directly and caches them itself (no temp file), matching
    the reference notebook's ``talker`` pattern (``return response.content``).
    """

    client: OpenAI = field(default_factory=_client)
    model: str = TTS_MODEL
    voice: str = TTS_VOICE

    def speech(self, text: str) -> bytes:
        """Synthesize ``text`` to MP3 and return the raw audio bytes."""
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="mp3",
        )
        return response.content
```

- [ ] **Step 2: Write the `speech` smoke check (fake client, no network) and run it.**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run python - <<'PY'
from types import SimpleNamespace as NS
from gradio_app.airline_multimodal import VoiceStudio

# fake client: client.audio.speech.create(...) -> object with .content bytes
fake = NS(audio=NS(speech=NS(create=lambda **kw: NS(content=b"ID3fake-mp3-bytes"))))
studio = VoiceStudio(client=fake)  # type: ignore[arg-type]
out = studio.speech("hello there")
assert out == b"ID3fake-mp3-bytes", out
print("VoiceStudio.speech OK", len(out), "bytes")
PY
```
Expected: `VoiceStudio.speech OK 17 bytes`.

- [ ] **Step 3: Lint + type check.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run ruff format gradio_app/airline_multimodal.py
uv run ruff check gradio_app/airline_multimodal.py
uv run mypy gradio_app/airline_multimodal.py
```
Expected: all pass.

- [ ] **Step 4: Commit.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
git add gradio_app/airline_multimodal.py
git commit -m "Add VoiceStudio (OpenRouter TTS returning mp3 bytes)"
```

---

### Task 4: `Bot` — persona, tool, message mapping, and the `respond` loop

**Files:**
- Modify: `gradio_app/airline_multimodal.py` (add `Bot` after `VoiceStudio`)

**Interfaces:**
- Consumes: `_client`, `PriceStore`, `ImageStudio`, `VoiceStudio`, `CHAT_MODEL`, the message/tool param types.
- Produces: `Bot` dataclass with fields `model/system/max_tokens/client/prices/images/voice`, `SYSTEM` and `TOOL` ClassVars, `get_airline_price(location: str) -> str`, `_to_messages(message, history) -> list[ChatCompletionMessageParam]` (instance method, uses `self.system`), and `respond(message: str, history: list[dict[str, Any]]) -> Iterator[tuple[str, Any, Any]]`.

- [ ] **Step 1: Extend imports, then add `Bot`.**

Add these imports to the existing import block:

```python
import json
from collections.abc import Iterator
from typing import Any, cast

import gradio as gr
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
```

(`Any` and `cast` join the existing `from typing import ClassVar` line → `from typing import Any, ClassVar, cast`.)

Insert after `VoiceStudio`:

```python
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
        """Map the system prompt + Gradio ``messages``-format history + new user turn
        to an OpenAI ``messages`` list. Only ``user``/``assistant`` turns with content
        are kept."""
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
        """Generate a city image; on any failure return ``gr.skip()`` (box unchanged)."""
        try:
            return self.images.image(city)
        except Exception:
            return gr.skip()

    def _safe_speech(self, text: str) -> Any:
        """Synthesize voice; on any failure return ``gr.skip()`` (box unchanged)."""
        try:
            return self.voice.speech(text)
        except Exception:
            return gr.skip()

    def respond(
        self, message: str, history: list[dict[str, Any]]
    ) -> Iterator[tuple[str, Any, Any]]:
        """Answer ``message``, running the price tool as needed, and drive three boxes.

        Yields ``(chat_text, image_value, audio_value)`` tuples for ``gr.ChatInterface``
        + its ``additional_outputs`` (image, audio). The chat slot always carries the
        reply text; ``gr.skip()`` leaves an image/audio box unchanged. On a tool call the
        destination image is generated (once) and a second completion produces the final
        text; every reply is then spoken. Media failures are swallowed by the
        ``_safe_*`` helpers so a modality never breaks the chat.
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
            city = ""
            for call in choice.tool_calls:
                if call.type != "function":
                    continue
                city = json.loads(call.function.arguments).get("location", "")
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=call.id,
                        content=self.get_airline_price(city),
                    )
                )
            image_value = self._safe_image(city)
            followup = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
            )
            final_text = followup.choices[0].message.content or ""
        else:
            final_text = choice.content or ""

        # Stage 1: show text (and image, if any) immediately; audio still rendering.
        yield final_text, image_value, gr.skip()
        # Stage 2: attach the spoken reply.
        yield final_text, image_value, self._safe_speech(final_text)
```

- [ ] **Step 2: Write the `respond` smoke check (fakes, no network) and run it (expect FAIL first).**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run python - <<'PY'
from types import SimpleNamespace as NS
import gradio as gr
from gradio_app.airline_multimodal import Bot

# In Gradio 6, gr.skip() is a no-op update dict ({'__type__': 'update'}); compare by ==.
SKIP = gr.skip()

def completion(content=None, tool_calls=None):
    msg = NS(content=content, tool_calls=tool_calls,
             model_dump=lambda exclude_none=True: {"role": "assistant", "content": content or ""})
    return NS(choices=[NS(message=msg)])

def tool_call(location):
    return NS(type="function", id="call_1",
              function=NS(name="get_airline_price", arguments='{"location": "%s"}' % location))

def fake_client(scripted):
    it = iter(scripted)
    return NS(chat=NS(completions=NS(create=lambda **kw: next(it))))

fake_prices = NS(price=lambda loc: "120 EUR")
fake_images = NS(image=lambda city: "IMG:" + city)
fake_voice = NS(speech=lambda text: "AUDIO:" + text)

# --- tool path: tool call, then final text ---
bot = Bot(
    client=fake_client([
        completion(tool_calls=[tool_call("London")]),
        completion(content="A ticket to London is 120 EUR."),
    ]),
    prices=fake_prices, images=fake_images, voice=fake_voice,
)
out = list(bot.respond("I want to go to London", []))
assert len(out) == 2, out
assert out[0][0] == "A ticket to London is 120 EUR."     # reply text
assert out[0][1] == "IMG:London"                         # image from the tool's city
assert out[0][2] == SKIP                                 # audio skipped in stage 1
assert out[1][1] == "IMG:London"                         # image persists into stage 2
assert out[1][2] == "AUDIO:A ticket to London is 120 EUR."  # audio in stage 2
print("respond tool-path OK")

# --- plain path: no tool call -> no image, audio still generated ---
bot2 = Bot(
    client=fake_client([completion(content="Hi! How can I help you today for your trip?")]),
    prices=fake_prices, images=fake_images, voice=fake_voice,
)
out2 = list(bot2.respond("Hi", []))
assert out2[0][0] == "Hi! How can I help you today for your trip?"
assert out2[0][1] == SKIP                                # image unchanged (no tool)
assert out2[1][2] == "AUDIO:Hi! How can I help you today for your trip?"
print("respond plain-path OK")

# --- media failure is swallowed ---
boom_voice = NS(speech=lambda text: (_ for _ in ()).throw(RuntimeError("tts down")))
bot3 = Bot(
    client=fake_client([completion(content="Sure.")]),
    prices=fake_prices, images=fake_images, voice=boom_voice,
)
out3 = list(bot3.respond("ok", []))
assert out3[1][0] == "Sure."                             # text still yielded
assert out3[1][2] == SKIP                                # failed audio -> skip
print("respond media-failure OK")
print("ALL respond checks passed")
PY
```
Expected before Step 1: import error (FAIL). After Step 1: prints the three `OK` lines then `ALL respond checks passed`.

> Note: `gr.skip()` returns a no-op update **dict** (`{'__type__': 'update'}`) in Gradio 6, so `== SKIP` (where `SKIP = gr.skip()`) is the right way to assert "box unchanged". If a future Gradio version changes that representation, update the `SKIP` comparison accordingly; the text/image string asserts are version-independent.

- [ ] **Step 3: Run the smoke check to verify it PASSES.**

Run the Step 2 command. Expected: `ALL respond checks passed`.

- [ ] **Step 4: Lint + type check.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run ruff format gradio_app/airline_multimodal.py
uv run ruff check gradio_app/airline_multimodal.py
uv run mypy gradio_app/airline_multimodal.py
```
Expected: all pass. (If `mypy` flags the `choice.model_dump(...)` append, the `cast(ChatCompletionMessageParam, ...)` already covers it; if it flags `tools=[self.TOOL]`, confirm `TOOL` is annotated `ChatCompletionFunctionToolParam`.)

- [ ] **Step 5: Commit.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
git add gradio_app/airline_multimodal.py
git commit -m "Add Bot: price tool, message mapping, and multimodal respond loop"
```

---

### Task 5: `build_demo` and `launch` — the three-box UI

**Files:**
- Modify: `gradio_app/airline_multimodal.py` (add `build_demo`, `launch`, `__main__` guard at the end)

**Interfaces:**
- Consumes: `Bot`, `gr`.
- Produces: `build_demo(bot: Bot | None = None) -> gr.Blocks`, `launch(**kwargs: Any) -> None`.

- [ ] **Step 1: Add `build_demo`, `launch`, and the entry-point guard.**

Append to the module:

```python
def build_demo(bot: Bot | None = None) -> gr.Blocks:
    """Build the three-box UI: chat (left), destination image + voice (right).

    ``additional_outputs`` components must exist in the ``Blocks`` context, so ``image``
    and ``audio`` are defined with ``render=False`` and ``.render()``-ed into the right
    column. Accepts an optional pre-built ``Bot`` (so ``launch`` can seed its store, and
    tests can inject fakes); defaults to a fresh ``Bot`` for import-time smoke checks.
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
                    type="messages",
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
    before serving so the first request reads a ready table. Generated image (PIL) and
    voice (bytes) values are handled by Gradio's own cache — no temp files or
    ``allowed_paths`` needed.
    """
    load_dotenv()
    bot = Bot()
    bot.prices.ensure_seeded()
    build_demo(bot).launch(**kwargs)


if __name__ == "__main__":
    launch()
```

- [ ] **Step 2: Write the `build_demo` smoke check (fake bot, no network) and run it.**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run python - <<'PY'
from types import SimpleNamespace as NS
import gradio as gr
from gradio_app.airline_multimodal import build_demo, Bot

# a fake bot whose respond is never called at build time
fake = NS(respond=lambda message, history: iter([("", gr.skip(), gr.skip())]))
demo = build_demo(fake)  # type: ignore[arg-type]
assert isinstance(demo, gr.Blocks), type(demo)
print("build_demo OK ->", type(demo).__name__)
PY
```
Expected: `build_demo OK -> Blocks`. No `OPENROUTER_API_KEY` needed (the fake bot owns no client).

- [ ] **Step 3: Confirm the whole module still imports with no side effects.**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run python -c "import gradio_app.airline_multimodal as m; print('import OK', bool(m.build_demo))"
```
Expected: `import OK True` with no key set.

- [ ] **Step 4: Lint + type check.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
uv run ruff format gradio_app/airline_multimodal.py
uv run ruff check gradio_app/airline_multimodal.py
uv run mypy gradio_app/airline_multimodal.py
```
Expected: `All checks passed!` and `Success: no issues found in 1 source file`.

- [ ] **Step 5: Commit.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
git add gradio_app/airline_multimodal.py
git commit -m "Add build_demo (three-box UI) and launch for airline_multimodal"
```

---

### Task 6: Document the app in the README

**Files:**
- Modify: `README.md` (add a section for the new app, following the existing gradio-app entries)

**Interfaces:**
- Consumes: nothing (docs only).
- Produces: nothing (docs only).

- [ ] **Step 1: Read the existing README gradio-app sections to match style.**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
sed -n '60,95p' README.md
```
Note the heading depth, the `uv run python -m gradio_app.<name>` line format, and the env-var note style used by the `airline` entry.

- [ ] **Step 2: Add a section for `gradio_app.airline_multimodal`.**

Add an entry (place it just after the existing `gradio_app.airline` section), matching the surrounding format. Content to include:

```markdown
### Airline assistant with image & voice (`gradio_app.airline_multimodal`)

A multimodal airline-ticket assistant with three boxes — chat, a generated image of the
destination city, and a spoken (TTS) rendering of every reply. Everything runs through a
single OpenAI-SDK client pointed at **OpenRouter**: chat + function calling
(`openai/gpt-4o-mini`), the price tool (SQLite dummy prices), image generation
(`openai/gpt-image-1`), and voice (`hexgrad/kokoro-82m` TTS). Ask to fly somewhere and the
bot calls `get_airline_price`, generates an image of that city, and speaks the reply.

Run it:

    uv run python -m gradio_app.airline_multimodal

Needs `OPENROUTER_API_KEY` in `.env`. Self-contained (no imports from `gradio_app.airline`);
the price database `gradio_app/airline_prices.db` is generated and seeded on launch.
```

(Match the exact heading level and prose conventions of the neighboring entries; adjust the model IDs in the text if the constants changed.)

- [ ] **Step 3: Verify the edit reads correctly.**

Run:
```bash
cd /home/aprimus/PycharmProjects/llm_engineering
grep -n "airline_multimodal" README.md
```
Expected: the new run command and heading appear.

- [ ] **Step 4: Commit.**

```bash
cd /home/aprimus/PycharmProjects/llm_engineering
git add README.md
git commit -m "Document airline_multimodal (image + voice) app in README"
```

---

## Verification (whole change)

1. **Import (no side effects, no key):** `uv run python -c "import gradio_app.airline_multimodal"` succeeds with no `OPENROUTER_API_KEY` set.
2. **Lint & types:** `uv run ruff check gradio_app/airline_multimodal.py` → `All checks passed!`; `uv run mypy gradio_app/airline_multimodal.py` → `Success`.
3. **Unit smoke checks pass:** the Task 1 (`PriceStore`), Task 2 (`ImageStudio._decode`), Task 3 (`VoiceStudio.speech`), Task 4 (`respond` fakes), and Task 5 (`build_demo`) checks all print their `OK` lines.
4. **Manual (hits OpenRouter; needs `OPENROUTER_API_KEY` in `.env`):** `uv run python -m gradio_app.airline_multimodal`, then:
   - Type **"Hi"** → the bot introduces itself and asks *"How can I help you today for your trip?"*, and the **voice** box plays the reply. The image box stays empty.
   - Type **"I would like to go to London"** → the bot calls the tool, the reply contains **`120 EUR`**, an **image of London** appears, and the **voice** box plays the priced reply.
   - Ask a follow-up (e.g. "and to Rome?") → stays concise, reports `140 EUR`, shows a Rome image, and speaks it.

## Notes

- **No new dependencies.** `openai`, `gradio`, `httpx` (transitive via `openai`), `Pillow` (transitive via `gradio`), and `python-dotenv` are already present; `sqlite3`/`base64`/`json` are stdlib.
- **`gradio_app/airline.py` is untouched** — this is a sibling module, and nothing imports across the two.
- **Image path mismatch is intentional:** `ImageStudio` uses `httpx` against `/api/v1/images` rather than the SDK's `images.generate()` (which posts to `/images/generations`). If OpenRouter later aliases the SDK path, `image()` can be simplified to `self.client.images.generate(...)`.
- **Incremental lint caveat:** `ruff` flags unused imports, so if you commit strictly per task, only import what each task uses (adding later imports as needed) — or implement Tasks 1–5 before the first commit. The Task 1 import block lists the final set.