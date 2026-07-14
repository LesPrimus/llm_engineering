# Airline Ticket Chat (gr.ChatInterface + tool use) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-turn Gradio airline-ticket assistant, built on `gr.ChatInterface` and backed by the native Anthropic (Claude) SDK, that calls a `get_airline_price` tool returning dummy EUR prices.

**Architecture:** One new module, `gradio_app/airline.py` — the sibling of `gradio_app/chatinterface.py` (native-Claude `Bot` on `gr.ChatInterface`), with one addition: a `get_airline_price` tool. A frozen `Bot` dataclass owns its own `Anthropic` client; the persona and tool list live on the class as `ClassVar`s. `chat()` runs a **streaming manual tool loop**: it streams a turn, and if Claude stops with `stop_reason == "tool_use"`, it runs `get_airline_price`, appends the result, and streams the next turn — accumulating text across turns into one growing string so `gr.ChatInterface` never resets the displayed reply.

**Tech Stack:** Python 3.14, uv, Gradio (`gr.ChatInterface`), the Anthropic Python SDK (already a dependency), python-dotenv.

## Global Constraints

- Python >= 3.14; manage everything through `uv` (`uv run …`). **No new dependencies** — `anthropic`, `gradio`, and `python-dotenv` are already in `pyproject.toml`.
- No test framework in this repo — verification is **import/run smoke checks** (`uv run python -c "…"`), plus `ruff` and `mypy`, matching the existing gradio modules.
- **Native Anthropic SDK, not OpenRouter.** Like `gradio_app/chatinterface.py`: `from anthropic import Anthropic`, `from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam`, streamed via `client.messages.stream(...)`. It reads **`ANTHROPIC_API_KEY`** (not `OPENROUTER_API_KEY`). Default model `claude-sonnet-5`, matching `chatinterface.py`.
- Follow the `chatinterface.py` shape: a **frozen dataclass that owns its client** via `field(default_factory=Anthropic)`; module-level `build_demo()` / `launch()` factory functions (no app-class wrapper — single bot). Importing the module must have **no side effects** (no client built at import, no network, no `.env` read at import). `load_dotenv()` runs inside `launch()` — before `build_demo()` builds the `Bot` — so `ANTHROPIC_API_KEY` is present when `Anthropic()` reads it.
- The persona (`SYSTEM`) and tool list (`TOOLS`) are **`ClassVar`s** on `Bot` (excluded from `__init__`); the `system` instance field defaults to `SYSTEM`.
- **Do NOT pass `type="messages"` to `gr.ChatInterface`** — this Gradio version's `ChatInterface` does not accept that keyword (mypy rejects it), and the default already delivers messages-format history. `gradio_app/chatinterface.py` omits it for the same reason.
- **Under Python 3.14, `Bot.__dataclass_fields__` includes `ClassVar` pseudo-fields** (marked `_FIELD_CLASSVAR`). To assert `SYSTEM`/`TOOLS` are not init fields, use `dataclasses.fields(Bot)` (which excludes `ClassVar`s), **not** membership in `__dataclass_fields__`.
- **Reply short and concise**, and **never invent prices** — the bot must call `get_airline_price`. `get_airline_price` returns **dummy data** only (no real backend) and never raises.
- Keep personal info out of committed files (public repo). Git commits use the repo-local identity; **no Claude co-author trailer**.

---

### Task 1: `gradio_app/airline.py` — the airline-ticket chat app with tool use

**Files:**
- Create: `gradio_app/airline.py`

**Interfaces:**
- Consumes: nothing new — `anthropic.Anthropic`, `anthropic.types.{MessageParam, ToolParam, ToolResultBlockParam}`, `gradio`, `dotenv.load_dotenv` (all already available).
- Produces:
  - `get_airline_price(location: str) -> str` — module-level function returning a dummy EUR price string, e.g. `"120 EUR"`.
  - `TOOL: ToolParam` — the Anthropic tool definition (name `get_airline_price`, one required `location` string param).
  - `class Bot` — frozen dataclass. Class attributes `SYSTEM: ClassVar[str]` and `TOOLS: ClassVar[list[ToolParam]] = [TOOL]`. Fields: `model: str = "claude-sonnet-5"`, `system: str = SYSTEM`, `max_tokens: int = 1024`, `client: Anthropic = field(default_factory=Anthropic)`.
  - `Bot._to_messages(message: str, history: list[dict[str, Any]]) -> list[MessageParam]` — `@staticmethod` mapping Gradio history + the new user message to Anthropic messages (pure, no network).
  - `Bot.chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]` — streams the reply accumulated so far, running the `get_airline_price` tool loop as needed.
  - `build_demo() -> gr.Blocks`; `launch(**kwargs: Any) -> None`. Entry point: `python -m gradio_app.airline`.

- [ ] **Step 1: Write the import/ClassVar smoke check and run it to confirm it fails**

Run:
```bash
uv run python -c "import dataclasses, gradio_app.airline as m; assert isinstance(m.Bot.SYSTEM, str) and 'airline' in m.Bot.SYSTEM.lower(); assert isinstance(m.Bot.TOOLS, list) and m.Bot.TOOLS[0]['name'] == 'get_airline_price'; names = {f.name for f in dataclasses.fields(m.Bot)}; assert names == {'model', 'system', 'max_tokens', 'client'}, names; print('OK')"
```
Expected: **FAIL** with `ModuleNotFoundError: No module named 'gradio_app.airline'` (file not created yet).

- [ ] **Step 2: Create `gradio_app/airline.py`**

```python
"""A multi-turn Gradio chat app: an airline ticket assistant with tool use.

Chat with a concise airline-ticket assistant and ask what a flight costs — the
bot calls a ``get_airline_price`` tool that returns a price in euros. Built on
``gr.ChatInterface`` (which owns the transcript and history) and backed by the
native Anthropic SDK, so replies stream straight from Claude.

This is the sibling of ``gradio_app/chatinterface.py`` — same self-contained
``Bot`` dataclass that owns its own ``Anthropic`` client, on ``gr.ChatInterface``
— but it declares a tool. ``chat`` runs a streaming manual tool loop: it streams a
turn; if Claude stops with ``stop_reason == "tool_use"`` it runs the tool, appends
the ``tool_result``, and streams the next turn. Text accumulates across turns into
one growing string so the UI never resets. The persona (``SYSTEM``) and tool list
(``TOOLS``) live on ``Bot`` as ``ClassVar``s. The bot is built inside
``build_demo`` (not at import), so importing this module has no side effects.

``get_airline_price`` returns dummy data (a small known-city table plus a
deterministic fallback) — there is no real pricing backend. Run it with
``uv run python -m gradio_app.airline``. It reads ``ANTHROPIC_API_KEY`` from
``.env`` (native Claude, like ``gradio_app.chatinterface``).
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import gradio as gr
from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv

# Dummy EUR ticket prices for a handful of destinations. Anything not listed
# gets a deterministic fallback (see get_airline_price) so unknown cities still
# vary and the demo stays reproducible.
_PRICES: dict[str, int] = {
    "london": 120,
    "paris": 95,
    "rome": 140,
    "berlin": 110,
    "new york": 480,
    "tokyo": 720,
    "sydney": 910,
}


def get_airline_price(location: str) -> str:
    """Return a dummy ticket price to ``location``, formatted as ``"<n> EUR"``.

    Known cities come from ``_PRICES`` (case-insensitive). Unknown locations get
    a deterministic fallback derived from the name, so repeated calls are stable
    and the demo needs no real pricing backend. Never raises.
    """
    key = location.strip().lower()
    if key in _PRICES:
        price = _PRICES[key]
    else:
        # Deterministic pseudo-price in the 100–999 range for unknown cities.
        price = 100 + sum(ord(c) for c in key) % 900
    return f"{price} EUR"


TOOL: ToolParam = {
    "name": "get_airline_price",
    "description": (
        "Get the price of an airline ticket to a destination city, in euros "
        "(EUR). Call this whenever the user asks the price or cost of a flight "
        "to a place, or asks how much it is to fly somewhere."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The destination city, e.g. 'London' or 'Tokyo'.",
            },
        },
        "required": ["location"],
    },
}


@dataclass(frozen=True)
class Bot:
    """An airline-ticket chatbot backed by Claude and a price-lookup tool."""

    SYSTEM: ClassVar[str] = (
        "You are a helpful airline ticket assistant. Help customers with flights "
        "and ticket prices. Keep every reply short and concise — a sentence or "
        "two. When a customer asks the price or cost of a flight to a place, call "
        "the get_airline_price tool to look it up; never guess or invent a price. "
        "Prices are in euros (EUR)."
    )
    TOOLS: ClassVar[list[ToolParam]] = [TOOL]

    model: str = "claude-sonnet-5"
    system: str = SYSTEM
    max_tokens: int = 1024
    client: Anthropic = field(default_factory=Anthropic)

    @staticmethod
    def _to_messages(message: str, history: list[dict[str, Any]]) -> list[MessageParam]:
        """Map a Gradio messages-format history plus the new user ``message`` to
        an Anthropic ``messages`` list.

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
        """Stream Claude's reply to ``message``, running the price tool as needed.

        Streams a turn; if Claude stops to call ``get_airline_price``, runs it,
        appends the ``tool_result``, and streams the next turn. ``reply``
        accumulates across every turn so ``gr.ChatInterface`` sees one growing
        string and never resets the displayed text.
        """
        messages = self._to_messages(message, history)
        reply = ""
        while True:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                messages=messages,
                tools=self.TOOLS,
            ) as stream:
                for text in stream.text_stream:
                    reply += text
                    yield reply
                final = stream.get_final_message()

            messages.append(MessageParam(role="assistant", content=final.content))
            if final.stop_reason != "tool_use":
                return

            tool_results: list[ToolResultBlockParam] = []
            for block in final.content:
                if block.type == "tool_use" and block.name == "get_airline_price":
                    args = cast(dict[str, Any], block.input)
                    result = get_airline_price(str(args["location"]))
                    tool_results.append(
                        ToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=block.id,
                            content=result,
                        )
                    )
            messages.append(MessageParam(role="user", content=tool_results))


def build_demo() -> gr.Blocks:
    """Build the UI: a multi-turn airline chat streamed from Claude with tool use."""
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        title="llm-engineering — airline tickets",
    )


def launch(**kwargs: Any) -> None:
    """Load environment variables, build the demo, and serve it.

    ``load_dotenv`` runs here (not at import) so ``ANTHROPIC_API_KEY`` is
    available before ``Bot`` builds its ``Anthropic`` client, matching
    ``gradio_app/chatinterface.py``.
    """
    load_dotenv()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
```

- [ ] **Step 3: Run the import/ClassVar smoke check to verify it passes**

Run:
```bash
uv run python -c "import dataclasses, gradio_app.airline as m; assert isinstance(m.Bot.SYSTEM, str) and 'airline' in m.Bot.SYSTEM.lower(); assert isinstance(m.Bot.TOOLS, list) and m.Bot.TOOLS[0]['name'] == 'get_airline_price'; names = {f.name for f in dataclasses.fields(m.Bot)}; assert names == {'model', 'system', 'max_tokens', 'client'}, names; print('OK')"
```
Expected: **PASS**, prints `OK`. Confirms `SYSTEM`/`TOOLS` are `ClassVar`s (absent from `dataclasses.fields`) and the four init fields are exactly `model`, `system`, `max_tokens`, `client`. (Reads `Bot` off the class only — no `Bot()` construction, so no `ANTHROPIC_API_KEY` and no network are needed.)

- [ ] **Step 4: Verify `get_airline_price` (known cities, case/whitespace, deterministic fallback)**

Run:
```bash
uv run python -c "from gradio_app.airline import get_airline_price as g; assert g('London') == '120 EUR', g('London'); assert g(' tokyo ') == '720 EUR', g(' tokyo '); assert g('London') == g('LONDON'); r1 = g('Reykjavik'); r2 = g('Reykjavik'); assert r1 == r2 and r1.endswith(' EUR'), r1; print('price OK', r1)"
```
Expected: **PASS**, prints `price OK 176 EUR`. Confirms: known-city lookup, case-insensitivity, whitespace stripping, and a stable `"<n> EUR"` fallback for unknown cities.

- [ ] **Step 5: Verify the history mapping without a network call**

Run:
```bash
uv run python -c "from gradio_app.airline import Bot; msgs = Bot._to_messages('how much to Rome?', [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'Hello!'}, {'role': 'system', 'content': 'x'}, {'role': 'user', 'content': ''}]); assert msgs == [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'Hello!'}, {'role': 'user', 'content': 'how much to Rome?'}], msgs; print('messages OK')"
```
Expected: **PASS**, prints `messages OK`. Confirms prior turns map 1:1 and keep order, the new user message is appended last, and non-user/assistant or empty-content entries are dropped. (`_to_messages` is a `@staticmethod`, so no `Bot()` is built — no key, no network.)

- [ ] **Step 6: Verify `build_demo()` constructs a `gr.Blocks` without a network call**

Run:
```bash
uv run python -c "from dotenv import load_dotenv; load_dotenv(); import gradio as gr; from gradio_app.airline import build_demo; d = build_demo(); assert isinstance(d, gr.Blocks), type(d); print('build_demo OK')"
```
Expected: **PASS**, prints `build_demo OK`. (`gr.ChatInterface` subclasses `gr.Blocks`. This builds `Bot()`, whose `Anthropic()` reads `ANTHROPIC_API_KEY` via `load_dotenv` but makes **no** network call — client construction is lazy. Requires `.env` with `ANTHROPIC_API_KEY` present.)

- [ ] **Step 7: Lint and type-check**

Run:
```bash
uv run ruff format gradio_app/airline.py
uv run ruff check gradio_app/airline.py
uv run mypy gradio_app/airline.py
```
Expected: all pass (`ruff format` may reformat; `ruff check` prints `All checks passed!`; `mypy` prints `Success: no issues found in 1 source file`).

- [ ] **Step 8: Commit**

```bash
git add gradio_app/airline.py
git commit -m "Add airline-ticket chat Gradio app with get_airline_price tool"
```

---

### Task 2: Document the new app in `README.md`

**Files:**
- Modify: `README.md` (add a subsection under "## Web UI", after the Website-summarizer block ending `links are not followed.`, before `## Development`)

**Interfaces:**
- Consumes: the entry point from Task 1 (`python -m gradio_app.airline`).
- Produces: nothing (docs only).

- [ ] **Step 1: Add the Airline-ticket-chat subsection**

Insert the following immediately after the Website-summarizer block's last line (`followed.`) and before the `## Development` heading:

````markdown

**Airline ticket chat.** A multi-turn airline-ticket assistant built on
`gr.ChatInterface`. Ask what a flight costs and Claude calls a `get_airline_price`
tool that returns a dummy price in euros; the reply streams back through the
Anthropic API:

```bash
uv run python -m gradio_app.airline
```

It reads `ANTHROPIC_API_KEY` from your `.env` (native Claude, like the
single-model app above) — no OpenRouter key needed. Prices are placeholder data,
not a real fare lookup.
````

- [ ] **Step 2: Verify the edit reads correctly**

Run:
```bash
uv run python -c "t = open('README.md').read(); assert 'Airline ticket chat' in t and 'gradio_app.airline' in t and 'get_airline_price' in t; print('README OK')"
```
Expected: **PASS**, prints `README OK`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document airline-ticket chat app in README"
```

---

## Verification (whole feature)

After all tasks, confirm end to end:

1. `uv run ruff check .` and `uv run mypy gradio_app/airline.py` pass.
2. Import smoke (no side effects): `uv run python -c "import gradio_app.airline; print('import OK')"`.
3. Tool smoke: the Task 1 Step 4 `get_airline_price` check passes (`price OK 176 EUR`).
4. Mapping smoke: the Task 1 Step 5 `_to_messages` check passes.
5. **Manual (hits the Anthropic API):** `uv run python -m gradio_app.airline`, then ask "how much is a flight to Tokyo?" and confirm the bot **calls the tool** and streams a **short** answer containing `720 EUR`; ask a follow-up like "and to Rome?" and confirm it stays concise and reports `140 EUR`, and that earlier turns are remembered. Run this once by hand; it requires `ANTHROPIC_API_KEY` in `.env`.

## Notes / Out of scope

- No model selector (single native-Claude bot), no real pricing backend, no extra tool params (origin/dates/passengers/cabin), no additional tools/retrieval/booking, no cross-session persistence, no app-class wrapper — all deferred per the spec.
- `gradio_app/__init__.py`, `app.py`, `chatinterface.py`, `multibot.py`, `website_summarizer.py`, `helpers.py`, `main.py`, `pyproject.toml`, `openrouter/`, and `claude/` are not modified (this is a standalone entry point, like the other gradio apps).