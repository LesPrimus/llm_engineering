# Design: `gradio_app` guitar-store chat (`gr.ChatInterface`)

**Date:** 2026-07-13
**Status:** Approved

## Goal

Add a Gradio chat app for an online guitar store, in a new module
`gradio_app/chatinterface.py`. It is the direct sibling of `gradio_app/app.py`
— a single **native-Anthropic** `Bot` reached through the Claude SDK, reading
`ANTHROPIC_API_KEY` — with one departure: it uses **`gr.ChatInterface`** (a
multi-turn chat with conversation history) instead of the stateless
`gr.Interface`, so the bot sees the whole conversation, not just the latest
message.

The bot is a general, self-contained **guitar-store sales assistant**: it
recommends acoustic/electric/bass guitars and gear by budget, skill, and style,
and explains gear in approachable terms. It carries **no fictional store**
(no invented catalog, prices, or stock).

New/changed files: `gradio_app/chatinterface.py` (new) and `README.md` (add a
section). `gradio_app/__init__.py`, `app.py`, `multibot.py`,
`website_summarizer.py`, `helpers.py`, `main.py`, and `pyproject.toml` are
untouched — the native Anthropic SDK is already a dependency.

## `Bot` dataclass

Mirrors `app.py`'s native-Claude `Bot`: a frozen dataclass owning its own
`Anthropic` client via `default_factory`. The persona lives on the class as a
`ClassVar[str]` (excluded from `__init__`); the `system` instance field defaults
to it, so per-instance override still works.

```python
from typing import ClassVar

@dataclass(frozen=True)
class Bot:
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

    model: str = "claude-sonnet-5"     # same default as app.py
    system: str = SYSTEM               # instance field defaults to the class constant
    max_tokens: int = 1024
    client: Anthropic = field(default_factory=Anthropic)

    def chat(self, message: str, history: list[dict[str, str]]) -> Iterator[str]:
        messages: list[MessageParam] = [
            MessageParam(role=turn["role"], content=turn["content"])
            for turn in history
            if turn.get("role") in ("user", "assistant") and turn.get("content")
        ]
        messages.append(MessageParam(role="user", content=message))
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=messages,
        ) as stream:
            reply = ""
            for text in stream.text_stream:
                reply += text
                yield reply
```

Rationale:

- **`gr.ChatInterface` calls `fn(message, history)`.** With `type="messages"`
  (set explicitly on the interface), `history` is a list of
  `{"role": "user"|"assistant", "content": str}` dicts. Anthropic's
  `MessageParam` uses the same `role`/`content` shape, so history maps **1:1** to
  Anthropic messages — no format translation, just wrapping.
- The comprehension keeps only `user`/`assistant` turns with non-empty content, so
  any metadata/tool entries Gradio might include are skipped. `turn["role"]` is a
  `str`; a small `cast(...)`/`Literal` handling keeps mypy satisfied against
  `MessageParam`'s `Literal["user", "assistant"]` role during implementation.
- The **system prompt goes in `system=`**, not the messages list — same as
  `app.py`. The persona is never part of the turn history.
- Streaming contract is identical to `app.py`: yield the reply accumulated so far
  on each text chunk, which `gr.ChatInterface` renders progressively.

## UI

`build_demo() -> gr.Blocks` constructs one `Bot` and wires a `gr.ChatInterface`:

```python
def build_demo() -> gr.Blocks:
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        type="messages",
        title="llm-engineering — guitar store",
    )
```

- **`type="messages"`** is set explicitly so `history` arrives as role/content
  dicts (matching the `MessageParam` mapping) and does not depend on the Gradio
  default.
- `gr.ChatInterface` owns the transcript, input textbox, and multi-turn history
  state — no manual state wiring needed.
- Structure follows `app.py`'s lighter `build_demo()` / `launch()` factory
  functions (not `website_summarizer.py`'s app class): this is a single bot, so
  the app-class wrapper — which exists there to own *multiple* summarizers — buys
  nothing here.

## Bootstrap

Identical to `app.py`:

```python
def launch(**kwargs: Any) -> None:
    load_dotenv()          # so ANTHROPIC_API_KEY is present before Bot builds its client
    build_demo().launch(**kwargs)

if __name__ == "__main__":
    launch()
```

`load_dotenv()` runs in `launch()` — not at import — before `build_demo()` builds
the `Bot` (whose `Anthropic()` reads `ANTHROPIC_API_KEY`). Importing the module
has no side effects. Run with `uv run python -m gradio_app.chatinterface`.

## README

Add a short "Guitar store chat" subsection under **Web UI**, matching the style of
the existing entries: what it does (multi-turn guitar-store chat, native Claude),
the run command (`uv run python -m gradio_app.chatinterface`), and that it uses
`ANTHROPIC_API_KEY` (like the single-model app, not OpenRouter).

## Error handling

No extra guard layer — consistent with `app.py`. `gr.ChatInterface` won't send an
empty message, and Anthropic SDK errors during `chat` propagate for Gradio to
surface, exactly as `app.py` does. (No URL/fetch step exists here, so there is
nothing analogous to the summarizer's fetch guards.)

## Verification

Repo convention is no test framework — verify with import/run smoke checks:

1. `import gradio_app.chatinterface` succeeds with **no side effects** (no client
   built, no `.env` read at import).
2. `Bot.SYSTEM` is a class attribute (present on `Bot` itself, not an `__init__`
   parameter); a default `Bot()`'s `.system` equals `Bot.SYSTEM`.
3. `chat` maps a sample `history` (a couple of role/content dicts) plus a new
   message to the expected `MessageParam` list ordering — can be checked by
   constructing the messages list without hitting the network.
4. `build_demo()` returns a `gr.Blocks`.
5. `ruff format`, `ruff check`, and `mypy` pass on the new module.
6. Launch `uv run python -m gradio_app.chatinterface`, hold a short multi-turn
   conversation ("recommend a first electric guitar" → follow-up), and confirm the
   reply streams and earlier turns are remembered (this step hits the Anthropic
   API; run once by hand).

## Out of scope

- Any model selector / dropdown — this is a single native-Claude bot (unlike
  `multibot.py`).
- A fictional store: no invented catalog, prices, stock, or policies baked into
  the prompt.
- Tool use, retrieval, or a real product backend.
- Persisting conversations across sessions (in-session history only, held by
  `gr.ChatInterface`).
- An app-class wrapper (`app.py`'s function factory is used instead).
- Changes to `gradio_app/app.py`, `multibot.py`, `website_summarizer.py`,
  `__init__.py`, `helpers.py`, `main.py`, `pyproject.toml`, `openrouter/`, or
  `claude/`.
- Adding a test framework (pytest) — verification stays smoke-check based.