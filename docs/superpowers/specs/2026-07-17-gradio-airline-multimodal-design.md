# Design: `gradio_app` airline ticket assistant with image + voice (`gr.ChatInterface`)

**Date:** 2026-07-17
**Status:** Approved

## Summary

A new, **self-contained** Gradio module — `gradio_app/airline_multimodal.py` — that turns
the airline-ticket assistant into a three-box multimodal demo: a **chat** transcript, a
generated **image** of the destination city, and a spoken **voice** rendering of every reply.

Everything runs through **one provider**: the OpenAI SDK pointed at **OpenRouter** with a
single `OPENROUTER_API_KEY`, exactly like `gradio_app/multibot.py`. OpenRouter now covers all
three modalities, so no second client or key is needed:

| Modality | Endpoint / SDK call | Model |
| --- | --- | --- |
| Chat + function calling | `client.chat.completions.create(...)` | `openai/gpt-4o-mini` |
| Image | raw `POST /api/v1/images` via `client.post("/images", ...)` | `openai/gpt-image-1` |
| Voice (TTS) | `client.audio.speech.create(...)` (OpenAI-SDK compatible) | `hexgrad/kokoro-82m` (OpenRouter has no OpenAI TTS model) |

**Behavior (the target UX):**

- User types "Hi" → the bot introduces itself and asks *"How can I help you today for your
  trip?"* (driven by the system prompt, no special-casing) → the reply is spoken as **audio**.
- User says "I would like to go to London" → the bot calls the **`get_airline_price`** tool,
  which reads a **SQLite** database of dummy prices → because a destination city is now known,
  an **image of that city** is generated → the priced reply is spoken as **audio**.
- **Voice is generated for every bot reply.** **Image is generated only when the price tool
  fires** (i.e. when a destination city is known); otherwise the image box is left unchanged.

This module is a **sibling** of `gradio_app/airline.py`, not a modification of it. It does **not
import** from `airline.py` or any other project module — the SQLite pricing logic is
re-implemented locally so the module stands alone. `airline.py` (native Anthropic +
`gr.ChatInterface`) is untouched.

## Non-goals (YAGNI)

- **No model dropdown.** The chat, image, and voice models are fixed constants. (Deliberate
  departure from the `Model`-enum pattern in `multibot.py`; a single fixed chat model was the
  chosen scope.)
- **No token-by-token streaming of the chat text.** The reply is produced per-turn
  (non-streaming) with **staged yields** so the UI still updates progressively (image first,
  then text, then audio). Streaming is a possible later enhancement, explicitly out of scope.
- **No real pricing / image / voice backends.** Prices are dummy SQLite data; images and audio
  are whatever the models return. This is a toy.
- **No persistence of generated media.** The image is returned as a PIL image and the voice as
  raw MP3 bytes; Gradio's own cache handles serving them (no temp files). They are not catalogued.

## Providers & configuration

A module-level `_client()` factory (mirroring `multibot.py`) calls `load_dotenv()` and returns
an `OpenAI` client pointed at OpenRouter:

```python
def _client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
```

Model / voice IDs are module or `ClassVar` constants:

- `CHAT_MODEL = "openai/gpt-4o-mini"`
- `IMAGE_MODEL = "openai/gpt-image-1"`
- `TTS_MODEL = "hexgrad/kokoro-82m"` (OpenRouter exposes no OpenAI TTS model; Kokoro is cheap
  and mp3-capable)
- `TTS_VOICE = "af_heart"` (Kokoro voices are prefixed: `af_*` female, `am_*` male)

Importing the module has **no side effects**: `_client()` (which reads the env var) and all
network/DB work happen only when a `Bot`/studio is instantiated or when `launch()` runs — never
at import. `build_demo()` accepts an injected `Bot`, so tests build the UI with fakes and no key.

## Units (small, injected, independently testable)

Four cooperating units. `Bot` owns its chat client and the three resources are **injected** into
it, so each can be faked in tests. Every unit is a `@dataclass(frozen=True)`.

### `PriceStore` — SQLite dummy prices (self-contained, local copy)

Re-implemented locally (a copy of the design proven in `airline.py`), **not imported**. Owns
every database concern — path, connection, seeding, lookup.

- `SEED: ClassVar[dict[str, int]]` — the seed rows (`london` 120, `paris` 95, `rome` 140,
  `berlin` 110, `new york` 480, `tokyo` 720, `sydney` 910).
- `DEFAULT_PATH: ClassVar[Path] = Path(__file__).with_name("airline_prices.db")` and a
  `path: Path = DEFAULT_PATH` field (alt DB injectable for tests). Building the path does no I/O.
  The file is already gitignored (`gradio_app/airline_prices.db`); the schema/seed are identical
  to `airline.py`'s, so sharing the same file is safe (`ensure_seeded` is idempotent). No new
  gitignore entry.
- `_connect()` opens one connection per call (SQLite's same-thread rule; Gradio uses worker
  threads).
- `ensure_seeded()` — `CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price
  INTEGER)`, then **only if empty** `INSERT OR IGNORE` the seed rows and commit. The only writer;
  idempotent and race-safe. Called by `launch()` before serving.
- `price(location) -> str` — a **pure reader**: `SELECT price FROM prices WHERE city = ?` on the
  lowercased/stripped key via `contextlib.closing(_connect())`; returns `"<n> EUR"` if found,
  else the deterministic fallback `100 + sum(ord(c) for c in key) % 900`. Never seeds.

### `ImageStudio` — destination image generation

Self-contained dataclass owning its own OpenRouter client (`client: OpenAI =
field(default_factory=_client)`).

- `image(city: str) -> PIL.Image.Image` — because the OpenAI SDK's `images.generate()` targets a
  different path (`/images/generations`) than OpenRouter's Image API (`/api/v1/images`), this
  makes a raw `httpx.post` to `{client.base_url}/images` reusing the client's key and base URL
  (`{"model": IMAGE_MODEL, "prompt": f"A scenic travel photograph of {city}."}`; response carries
  `data[0].b64_json`). It base64-decodes the first image and returns a **PIL image** — `gr.Image`
  accepts a PIL image directly, so no temp file. A pure `_decode(b64_json) -> Image.Image`
  staticmethod holds the offline-testable core.

### `VoiceStudio` — text-to-speech

Self-contained dataclass owning its own OpenRouter client (`field(default_factory=_client)`).

- `speech(text: str) -> bytes` — calls `client.audio.speech.create(model=TTS_MODEL,
  voice=TTS_VOICE, input=text, response_format="mp3")` and returns `response.content` (raw MP3
  **bytes**). `gr.Audio` accepts bytes directly and caches them itself — no temp file. This
  mirrors the reference notebook's `talker` (`return response.content` → `gr.Audio`).

### `Bot` — orchestrator

- Owns its chat client: `client: OpenAI = field(default_factory=_client)`.
- Injected resources: `prices: PriceStore = field(default_factory=PriceStore)`,
  `images: ImageStudio = field(default_factory=ImageStudio)`,
  `voice: VoiceStudio = field(default_factory=VoiceStudio)`.
- Config fields: `model: str = CHAT_MODEL`, `system: str = SYSTEM`, `max_tokens: int = 1024`.
- `SYSTEM: ClassVar[str]` — a short persona: introduce yourself on the first message and ask
  *"How can I help you today for your trip?"*, keep every reply short (a sentence or two), and
  **call `get_airline_price` whenever the customer names a destination they want to fly to** (e.g.
  "I would like to go to London") — not only on an explicit price question — never guess. Prices
  are in euros (EUR).
- **Tool.** `get_airline_price(self, location: str) -> str` is a **bound method** returning
  `self.prices.price(location)`. Its function-calling schema is a hand-written constant
  `TOOL` (`{"type": "function", "function": {"name": "get_airline_price", "description": ...,
  "parameters": {location: string}}}`). *Note:* the OpenAI SDK has no `beta_tool`/`tool_runner`
  auto-schema helper (unlike Anthropic in `airline.py`), so the schema is written by hand and the
  loop is hand-rolled. `PriceStore` stays pure — no knowledge of the tool/prompt layer.
- `_to_messages(message, history) -> list[ChatCompletionMessageParam]` — system message +
  Gradio `"messages"`-format history (already `{"role", "content"}` dicts, mapped directly, only
  `user`/`assistant` turns with content kept) + the new user turn.
- `respond(message, history)` — the generator wired to `gr.ChatInterface` (see Data flow).

## Data flow — `Bot.respond`

`respond(message, history)` is a generator that yields **one** tuple `(chat_message, image_value,
audio_value)` feeding `gr.ChatInterface` **plus** two `additional_outputs` (`image`, `audio`). The
**chat_message slot always carries the reply text** (a string); `gr.skip()` in an
`additional_outputs` slot leaves that box unchanged.

1. Build messages via `_to_messages`. First completion **with tools**:
   `client.chat.completions.create(model, messages, tools=[TOOL])`.
2. **If the model returns a `get_airline_price` tool call:**
   - Parse `location` from `tool_call.function.arguments` (JSON); run the bound tool
     (`self.prices.price(location)`); append the assistant tool-call message and the tool-result
     message to `messages`.
   - Generate the destination image (`image_value = self.images.image(...)`).
   - Second completion **without tools** for the final text (`final_text`).
3. **If no tool call:** `final_text` is the first completion's content and `image_value =
   gr.skip()` (image box left unchanged).
4. Generate voice for `final_text` (`self.voice.speech(final_text)` → MP3 bytes) and **yield**
   `(final_text, image_value, audio_bytes)`.

**Voice is generated for every reply** — `audio_value` is always the reply's MP3 bytes (matching
the reference notebook's `talker`, which returns audio on every turn), so it is never `gr.skip()`
except when TTS itself fails. `gr.Audio(autoplay=True)` plays it on arrival. The **image** box
updates only on a tool call (a known destination); otherwise it retains whatever was last shown.
The single-yield shape (rather than a staged text-then-audio pair) keeps the audio output
unambiguously present on each reply.

## UI — `build_demo(bot)`

`gr.Blocks` with the chat on the left and the image + audio boxes stacked on the right. Because
`ChatInterface`'s `additional_outputs` components must already exist in the `Blocks` context, the
`gr.Image` and `gr.Audio` are defined with `render=False` first, then `.render()`ed into the
right column:

```python
def build_demo(bot: Bot | None = None) -> gr.Blocks:
    bot = bot if bot is not None else Bot()
    with gr.Blocks(title="llm-engineering — airline (voice + image)") as demo:
        image = gr.Image(label="Destination", render=False)
        audio = gr.Audio(label="Voice", autoplay=True, render=False)
        with gr.Row():
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=bot.respond,
                    type="messages",
                    additional_outputs=[image, audio],
                    title="Airline tickets",
                )
            with gr.Column(scale=1):
                image.render()
                audio.render()
    return demo
```

Three boxes: chat (left, wider), destination image and voice player (right, stacked).

## `launch()`

Mirrors `airline.py`: `load_dotenv()` (also done in `_client()`), build the `Bot`, call
`bot.prices.ensure_seeded()` before serving (so the first request reads a ready table — `price`
never seeds), then `build_demo(bot).launch(**kwargs)`. Run with
`uv run python -m gradio_app.airline_multimodal`.

## Error handling

- **Media failures never break the chat.** `self.images.image(...)` and `self.voice.speech(...)`
  calls are wrapped in `try/except`; on failure the corresponding box is yielded as `gr.skip()`
  (left unchanged) and the conversation continues. The text reply always goes through.
- **Unknown cities** are handled by `PriceStore`'s deterministic fallback — no error path.
- **Missing `OPENROUTER_API_KEY`** fails fast at launch with a clear `KeyError` from `_client()`.
- **SQLite threading** — connection-per-call in `PriceStore` satisfies the same-thread rule under
  Gradio's worker threads.

## Testing / verification

**No test framework in this repo** — verification is **import/run smoke checks**
(`uv run python -c "…"`) that inject fakes, plus `ruff` + `mypy`, plus a manual API run,
matching the existing gradio modules (and the `airline.py` plan). TDD still applies at the
smoke-check level: write the check, watch it fail, implement, watch it pass, commit. Injected
fakes (a fake `OpenAI` client, fake studios) keep every check offline and key-free.

- **`PriceStore`** — against a temp DB: `ensure_seeded` is idempotent; seed cities return known
  prices (`"London"` → `"120 EUR"`); unknown cities return the stable char-sum fallback; `price`
  against an unseeded DB raises `sqlite3.OperationalError` (documents the launch-seeds-first
  contract).
- **`ImageStudio._decode`** — feed it a base64-encoded PNG and assert it returns a PIL image of
  the right size (the pure, offline core; the networked `image()` path is exercised manually).
- **`VoiceStudio.speech`** — with a fake client whose `audio.speech.create` returns an object
  carrying canned `.content` bytes: asserts `speech` returns exactly those bytes.
- **`Bot._to_messages`** — maps system + history + new turn to OpenAI messages; drops
  metadata/empty turns.
- **`Bot.respond`** — with a fake chat client scripted to return (a) a tool call then final text,
  and (b) plain text; and fake `ImageStudio`/`VoiceStudio`. Asserts: the tool dispatches to
  `prices.price`; the image is generated **only** on the tool-call path; audio is generated on
  **every** path; the yielded tuples have the right shape and `gr.skip()` where expected; a media
  exception is swallowed and the text still yields.
- **Import-time smoke** — importing the module has no side effects; `build_demo(bot)` with an
  injected fake `Bot` builds a `gr.Blocks` without network or key.
- **Manual** (hits OpenRouter; needs `OPENROUTER_API_KEY`) — run the app: "Hi" → intro reply +
  audio; "I'd like to go to London" → priced reply + a London image + audio.

## Files

- **New:** `gradio_app/airline_multimodal.py`
- **New:** tests (e.g. `tests/gradio_app/test_airline_multimodal.py`, matching the repo's test
  layout).
- **Changed:** `README.md` — document the new app and how to run it (following the pattern of the
  other gradio apps). `.gitignore` already covers `gradio_app/airline_prices.db`.
- **Unchanged:** `gradio_app/airline.py` and every other module — no imports across modules.