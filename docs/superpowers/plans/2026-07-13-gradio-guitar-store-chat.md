# Guitar-Store Chat (gr.ChatInterface) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-turn Gradio chatbot for an online guitar store, built on `gr.ChatInterface` and backed by the native Anthropic (Claude) SDK.

**Architecture:** One new module, `gradio_app/chatinterface.py` — the native-Claude sibling of `gradio_app/app.py`, but on `gr.ChatInterface` (conversation history) instead of the stateless `gr.Interface`. A frozen `Bot` dataclass owns its own `Anthropic` client; its guitar-store persona lives on the class as a `ClassVar[str]` and the `system` field defaults to it. `gr.ChatInterface` calls `fn(message, history)`; with `type="messages"`, `history` is a list of `{"role", "content"}` dicts that map 1:1 to Anthropic `MessageParam`s. A small `@staticmethod` performs that mapping so it can be verified without a network call.

**Tech Stack:** Python 3.14, uv, Gradio (`gr.ChatInterface`), the Anthropic Python SDK (already a dependency), python-dotenv.

## Global Constraints

- Python >= 3.14; manage everything through `uv` (`uv run …`). **No new dependencies** — `anthropic`, `gradio`, and `python-dotenv` are already in `pyproject.toml`.
- No test framework in this repo — verification is **import/run smoke checks** (`uv run python -c "…"`), plus `ruff` and `mypy`, matching the existing gradio modules.
- **Native Anthropic SDK, not OpenRouter.** This is the sibling of `gradio_app/app.py`: `from anthropic import Anthropic`, `from anthropic.types import MessageParam`, streamed via `client.messages.stream(...)`. It reads **`ANTHROPIC_API_KEY`** (not `OPENROUTER_API_KEY`). Default model `claude-sonnet-5`, matching `app.py`.
- Follow the `app.py` shape: a **frozen dataclass that owns its client** via `field(default_factory=Anthropic)`; module-level `build_demo()` / `launch()` factory functions (no app-class wrapper — this is a single bot). Importing the module must have **no side effects** (no client built at import, no network, no `.env` read at import). `load_dotenv()` runs inside `launch()` — before `build_demo()` builds the `Bot` — so `ANTHROPIC_API_KEY` is present when `Anthropic()` reads it.
- The persona is a **`ClassVar[str]`** on `Bot` (excluded from `__init__`); the `system` instance field defaults to it.
- **Single bot — no model selector/dropdown** (unlike `multibot.py`). Use `gr.ChatInterface` with `type="messages"`.
- **No fictional store** — the persona invents no catalog, prices, stock, or policies; it explicitly declines to make those up.
- Keep personal info out of committed files (public repo). Git commits use the repo-local identity; **no Claude co-author trailer**.

---

### Task 1: `gradio_app/chatinterface.py` — the guitar-store chat app

**Files:**
- Create: `gradio_app/chatinterface.py`

**Interfaces:**
- Consumes: nothing new — `anthropic.Anthropic`, `anthropic.types.MessageParam`, `gradio`, `dotenv.load_dotenv` (all already available).
- Produces:
  - `class Bot` — frozen dataclass. Class attribute `SYSTEM: ClassVar[str]` (the persona). Fields: `model: str = "claude-sonnet-5"`, `system: str = SYSTEM`, `max_tokens: int = 1024`, `client: Anthropic = field(default_factory=Anthropic)`.
  - `Bot._to_messages(message: str, history: list[dict[str, Any]]) -> list[MessageParam]` — `@staticmethod` mapping Gradio history + the new user message to Anthropic messages (pure, no network).
  - `Bot.chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]` — streams the reply accumulated so far on each chunk.
  - `build_demo() -> gr.Blocks`; `launch(**kwargs: Any) -> None`. Entry point: `python -m gradio_app.chatinterface`.

- [ ] **Step 1: Write the import/ClassVar smoke check and run it to confirm it fails**

Run:
```bash
uv run python -c "import gradio_app.chatinterface as m; assert isinstance(m.Bot.SYSTEM, str) and 'guitar' in m.Bot.SYSTEM.lower(); assert 'SYSTEM' not in m.Bot.__dataclass_fields__, 'SYSTEM must be a ClassVar, not a field'; b = m.Bot.__dataclass_fields__; assert set(b) == {'model', 'system', 'max_tokens', 'client'}, list(b); print('OK')"
```
Expected: **FAIL** with `ModuleNotFoundError: No module named 'gradio_app.chatinterface'` (file not created yet).

- [ ] **Step 2: Create `gradio_app/chatinterface.py`**

```python
"""A multi-turn Gradio chat app for an online guitar store.

Chat with a friendly guitar-store sales assistant: ask for recommendations by
budget, skill, and style, and follow up — the bot sees the whole conversation.
Built on ``gr.ChatInterface`` (which owns the transcript and history) and backed
by the native Anthropic SDK, so replies stream straight from Claude.

This is the sibling of ``gradio_app/app.py`` — same self-contained ``Bot``
dataclass that owns its own ``Anthropic`` client — but on ``gr.ChatInterface``
(multi-turn) rather than the stateless ``gr.Interface``. ``gr.ChatInterface``
calls ``fn(message, history)``; with ``type="messages"`` the history is a list of
``{"role", "content"}`` dicts, which map 1:1 to Anthropic ``MessageParam``s. The
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
    def _to_messages(
        message: str, history: list[dict[str, Any]]
    ) -> list[MessageParam]:
        """Map a Gradio ``type="messages"`` history plus the new user ``message``
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
        type="messages",
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
```

- [ ] **Step 3: Run the import/ClassVar smoke check to verify it passes**

Run:
```bash
uv run python -c "import gradio_app.chatinterface as m; assert isinstance(m.Bot.SYSTEM, str) and 'guitar' in m.Bot.SYSTEM.lower(); assert 'SYSTEM' not in m.Bot.__dataclass_fields__, 'SYSTEM must be a ClassVar, not a field'; b = m.Bot.__dataclass_fields__; assert set(b) == {'model', 'system', 'max_tokens', 'client'}, list(b); print('OK')"
```
Expected: **PASS**, prints `OK`. (Reads `Bot` off the class only — no `Bot()` construction, so no `ANTHROPIC_API_KEY` and no network are needed.)

- [ ] **Step 4: Verify the history mapping without a network call**

Run:
```bash
uv run python -c "from gradio_app.chatinterface import Bot; msgs = Bot._to_messages('what about a bass?', [{'role': 'user', 'content': 'first electric?'}, {'role': 'assistant', 'content': 'A Squier Strat is a great start.'}, {'role': 'system', 'content': 'ignored'}, {'role': 'user', 'content': ''}]); assert msgs == [{'role': 'user', 'content': 'first electric?'}, {'role': 'assistant', 'content': 'A Squier Strat is a great start.'}, {'role': 'user', 'content': 'what about a bass?'}], msgs; print('messages OK')"
```
Expected: **PASS**, prints `messages OK`. Confirms: prior turns map 1:1 and keep order, the new user message is appended last, and non-user/assistant or empty-content entries are dropped. (`_to_messages` is a `@staticmethod`, so no `Bot()` is built — no key, no network.)

- [ ] **Step 5: Verify `build_demo()` constructs a `gr.Blocks` without a network call**

Run:
```bash
uv run python -c "from dotenv import load_dotenv; load_dotenv(); import gradio as gr; from gradio_app.chatinterface import build_demo; d = build_demo(); assert isinstance(d, gr.Blocks), type(d); print('build_demo OK')"
```
Expected: **PASS**, prints `build_demo OK`. (`gr.ChatInterface` subclasses `gr.Blocks`. This builds `Bot()`, whose `Anthropic()` reads `ANTHROPIC_API_KEY` via `load_dotenv` but makes **no** network call — client construction is lazy. Requires `.env` with `ANTHROPIC_API_KEY` present.)

- [ ] **Step 6: Lint and type-check**

Run:
```bash
uv run ruff format gradio_app/chatinterface.py
uv run ruff check gradio_app/chatinterface.py
uv run mypy gradio_app/chatinterface.py
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add gradio_app/chatinterface.py
git commit -m "Add guitar-store chat Gradio app (gr.ChatInterface, native Claude)"
```

---

### Task 2: Document the new app in `README.md`

**Files:**
- Modify: `README.md` (add a subsection under "## Web UI", after the Website-summarizer block ending at the `links are not followed.` line, before `## Development`)

**Interfaces:**
- Consumes: the entry point from Task 1 (`python -m gradio_app.chatinterface`).
- Produces: nothing (docs only).

- [ ] **Step 1: Add the Guitar-store-chat subsection**

Insert the following immediately after the Website-summarizer block's last line (`followed.`) and before the `## Development` heading:

```markdown

**Guitar store chat.** A multi-turn chatbot for an online guitar store, built on
`gr.ChatInterface`. Ask for gear recommendations and keep following up — Claude
sees the whole conversation and its reply streams back through the Anthropic API:

```bash
uv run python -m gradio_app.chatinterface
```

It reads `ANTHROPIC_API_KEY` from your `.env` (native Claude, like the
single-model app above) — no OpenRouter key needed.
```

- [ ] **Step 2: Verify the edit reads correctly**

Run:
```bash
uv run python -c "t = open('README.md').read(); assert 'Guitar store chat' in t and 'gradio_app.chatinterface' in t; print('README OK')"
```
Expected: **PASS**, prints `README OK`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document guitar-store chat app in README"
```

---

## Verification (whole feature)

After all tasks, confirm end to end:

1. `uv run ruff check .` and `uv run mypy gradio_app/chatinterface.py` pass.
2. Import smoke (no side effects): `uv run python -c "import gradio_app.chatinterface; print('import OK')"`.
3. Mapping smoke: the Task 1 Step 4 `_to_messages` check passes.
4. **Manual (hits the Anthropic API):** `uv run python -m gradio_app.chatinterface`, then hold a short multi-turn conversation — e.g. "recommend a first electric guitar" → "what about something for metal?" — and confirm the reply **streams** and that earlier turns are **remembered** (the follow-up should build on the first answer). Run this once by hand; it requires `ANTHROPIC_API_KEY` in `.env`.

## Notes / Out of scope

- No model selector (single native-Claude bot), no fictional store/catalog/prices, no tool use or retrieval, no cross-session persistence, no app-class wrapper — all deferred per the spec.
- `gradio_app/__init__.py`, `app.py`, `multibot.py`, `website_summarizer.py`, `helpers.py`, `main.py`, `pyproject.toml`, `openrouter/`, and `claude/` are not modified (this is a standalone entry point, like the other gradio apps).