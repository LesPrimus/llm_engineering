# Airline Chat — SQLite-backed pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-memory `_PRICES` dict in `gradio_app/airline.py` with a SQLite database that `_price()` reads, seeded on launch — with identical behavior for the seven known cities and the same deterministic fallback for unknown ones.

**Architecture:** The public surface is unchanged — `get_airline_price` (a `@beta_tool`) still wraps a plain `_price()` core. `_price()` becomes a **pure reader**: it runs one `SELECT` against a `prices(city, price)` table and falls back to a deterministic char-sum price for cities not in the table. A new `_ensure_db()` helper creates and seeds the table (idempotent, race-safe) and is called **once by `launch()`** before serving — never by `_price()`. The database file (`gradio_app/airline_prices.db`) is generated, not committed, and gitignored. Each lookup opens its own connection (thread-safe for Gradio's worker threads).

**Tech Stack:** Python 3.14, uv, `sqlite3` + `pathlib` + `contextlib` (all stdlib — **no new dependency**), Gradio, the Anthropic Python SDK (`@beta_tool` / `tool_runner`), python-dotenv.

**Design reference:** `docs/superpowers/specs/2026-07-14-gradio-airline-chat-tool-use-design.md` — see the **Update (2026-07-15): SQLite-backed pricing** section.

> **Update (2026-07-15, during execution):** at the user's request the DB
> operations were encapsulated in an injected **`PriceStore`** dataclass rather
> than module-level helpers. Final shape (see the spec's updated 2026-07-15 note,
> which is the design of record):
> - `PriceStore` (`@dataclass(frozen=True)`) owns `SEED` (`ClassVar`), `DEFAULT_PATH`
>   (`ClassVar`), a `path` field, `_connect()`, `ensure_seeded()` (only writer), and
>   `price()` (pure reader). Replaces the module-level `_DB_PATH`/`_SEED`/`_connect`/
>   `_ensure_db`/`_price`.
> - `Bot` gains `prices: PriceStore = field(default_factory=PriceStore)`; init fields
>   are `model, system, max_tokens, client, prices`.
> - `get_airline_price` is no longer module-level — it is a `Bot` method
>   `get_airline_price(self, location)` returning `self.prices.price(...)`, and
>   `Bot.chat` wraps the bound method: `tools=[beta_tool(self.get_airline_price)]`
>   (the bound method drops `self`, so the derived schema is `{location}`).
> - `build_demo(bot: Bot | None = None)`; `launch()` builds the bot, calls
>   `bot.prices.ensure_seeded()`, then `build_demo(bot)`.
> The Task 1 code block and smoke checks below describe the earlier module-level
> shape; the delivered module and its smoke checks use the `PriceStore` API
> (`PriceStore().ensure_seeded()` / `.price(...)`, `Bot(prices=...)`).

## Global Constraints

- Python >= 3.14; run everything through `uv` (`uv run …`). **No new dependencies** — `sqlite3`, `pathlib`, and `contextlib` are stdlib; `anthropic`, `gradio`, and `python-dotenv` are already in `pyproject.toml`.
- No test framework in this repo — verification is **import/run smoke checks** (`uv run python -c "…"`), plus `ruff` and `mypy`, matching the existing gradio modules.
- **Only `gradio_app/airline.py`, `.gitignore`, and `README.md` change.** `Bot`, `_to_messages`, `chat`, the `@beta_tool` `get_airline_price`, `build_demo`, and every other module are untouched.
- Importing the module must have **no side effects**: no client built, no `.env` read, and **no DB file created** at import. `_DB_PATH` is a plain `Path` constant (no I/O). Seeding happens in `launch()`, not at import.
- `_price()` is a **pure reader** — it never creates or seeds the table. The contract is: `launch()` calls `_ensure_db()` before any read. A `SELECT` against an unseeded database raises `sqlite3.OperationalError`; a direct smoke check must call `_ensure_db()` first.
- Behavior parity: the seven seed cities keep their exact prices (`"London"` → `"120 EUR"`); lookups are case-insensitive and whitespace-stripped; unknown cities keep the deterministic `100 + sum(ord(c) for c in key) % 900` fallback (stable across calls).
- The generated DB file (`gradio_app/airline_prices.db`) is **gitignored and never committed**. Commits use explicit `git add <paths>` so the DB is never staged.
- Keep personal info out of committed files (public repo). Git commits use the repo-local identity; **no Claude co-author trailer**. Commit directly to `master` (no feature branch).

---

### Task 1: Swap the dict for a SQLite lookup in `gradio_app/airline.py`

**Files:**
- Modify: `gradio_app/airline.py` (imports, module docstring, `_PRICES`→`_SEED`+DB helpers, `_price`, `launch`)
- Modify: `.gitignore` (ignore the generated DB)

**Interfaces:**
- Consumes: `sqlite3`, `pathlib.Path`, `contextlib.closing` (stdlib); nothing new from other tasks.
- Produces:
  - `_DB_PATH: Path` — module constant, `gradio_app/airline_prices.db`.
  - `_SEED: dict[str, int]` — the seven seed city→price rows (was `_PRICES`).
  - `_connect() -> sqlite3.Connection` — opens a connection to `_DB_PATH`.
  - `_ensure_db() -> None` — creates the `prices(city TEXT PRIMARY KEY, price INTEGER)` table and seeds it from `_SEED` if empty (idempotent, race-safe).
  - `_price(location: str) -> str` — **pure reader**; returns `"<n> EUR"` (SELECT hit) or the deterministic fallback. Raises `sqlite3.OperationalError` if the table does not exist.
  - `get_airline_price` (`@beta_tool`) — unchanged; still `return _price(location)`.
  - `launch(**kwargs) -> None` — now calls `_ensure_db()` after `load_dotenv()`, before `build_demo()`.

- [ ] **Step 1: Write the SQLite smoke check and run it to confirm it FAILS**

Run:
```bash
rm -f gradio_app/airline_prices.db
uv run python -c "
import os, sqlite3
import gradio_app.airline as m
# import must not create the DB (no side effects at import)
assert not os.path.exists(m._DB_PATH), f'DB created at import: {m._DB_PATH}'
# _price is a pure reader: an unseeded DB has no table, so SELECT raises
try:
    m._price('London')
    raise AssertionError('_price should raise on an unseeded DB')
except sqlite3.OperationalError:
    pass
# seed, then read
m._ensure_db()
assert m._price('London') == '120 EUR', m._price('London')
assert m._price(' tokyo ') == '720 EUR', m._price(' tokyo ')
assert m._price('London') == m._price('LONDON'), 'not case-insensitive'
r1, r2 = m._price('Reykjavik'), m._price('Reykjavik')
assert r1 == r2 and r1.endswith(' EUR'), r1
# _ensure_db is idempotent (safe to call again)
m._ensure_db()
assert m._price('Paris') == '95 EUR', m._price('Paris')
# the tool delegates to _price and returns the same string
assert m.get_airline_price('Berlin') == '110 EUR', m.get_airline_price('Berlin')
print('sqlite pricing OK', r1)
"
```
Expected: **FAIL** with `AttributeError: module 'gradio_app.airline' has no attribute '_DB_PATH'` — the current module still uses the `_PRICES` dict and defines neither `_DB_PATH` nor `_ensure_db`.

- [ ] **Step 2: Rewrite `gradio_app/airline.py`**

Replace the entire file with the content below. Changes vs. the current file: the last paragraph of the module docstring (dict → SQLite); three new stdlib imports (`sqlite3`, `contextlib.closing`, `pathlib.Path`); `_PRICES` becomes `_SEED` alongside new `_DB_PATH`/`_connect`/`_ensure_db`; `_price` becomes a pure SQLite reader; `launch` calls `_ensure_db()`. `Bot`, `_to_messages`, `chat`, `get_airline_price`, and `build_demo` are unchanged.

```python
"""A multi-turn Gradio chat app: an airline ticket assistant with tool use.

Chat with a concise airline-ticket assistant and ask what a flight costs — the
bot calls a ``get_airline_price`` tool that returns a price in euros. Built on
``gr.ChatInterface`` (which owns the transcript and history) and backed by the
native Anthropic SDK, so replies stream straight from Claude.

This is the sibling of ``gradio_app/chatinterface.py`` — same self-contained
``Bot`` dataclass that owns its own ``Anthropic`` client, on ``gr.ChatInterface``
— but it declares a tool. Instead of a hand-written tool loop it uses the SDK's
**tool runner**: ``get_airline_price`` is a ``@beta_tool`` (its schema is derived
from the signature and docstring), and ``client.beta.messages.tool_runner(...,
stream=True)`` drives the loop — each iteration yields one turn's message stream,
and the runner runs the tool and continues automatically. ``chat`` accumulates
text across turns into one growing string so the UI never resets. The persona
(``SYSTEM``) lives on ``Bot`` as a ``ClassVar``. The bot is built inside
``build_demo`` (not at import), so importing this module has no side effects.

``get_airline_price`` returns dummy data from a small **SQLite** database (a seed
table of known cities plus a deterministic fallback) — there is no real pricing
backend. The database (``airline_prices.db``, next to this module) is created and
seeded by ``launch`` before serving; ``_price`` only reads it. The tool runner is
a **beta** SDK helper. Run it with ``uv run python -m gradio_app.airline``. It
reads ``ANTHROPIC_API_KEY`` from ``.env`` (native Claude, like
``gradio_app.chatinterface``).
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import gradio as gr
from anthropic import Anthropic, beta_tool
from anthropic.types.beta import BetaMessageParam
from dotenv import load_dotenv

# SQLite database of dummy EUR ticket prices, next to this module. It is
# generated and seeded on launch (see _ensure_db) and gitignored — seed data,
# not a real fare backend. Assigning the path does no I/O, so importing this
# module stays side-effect-free.
_DB_PATH = Path(__file__).with_name("airline_prices.db")

# Seed rows for the prices table. Cities not listed get a deterministic fallback
# in _price, so unknown cities still vary and the demo stays reproducible.
_SEED: dict[str, int] = {
    "london": 120,
    "paris": 95,
    "rome": 140,
    "berlin": 110,
    "new york": 480,
    "tokyo": 720,
    "sydney": 910,
}


def _connect() -> sqlite3.Connection:
    """Open a connection to the price database — one per lookup, so calls from
    Gradio's worker threads never share a connection (SQLite's default
    same-thread restriction)."""
    return sqlite3.connect(_DB_PATH)


def _ensure_db() -> None:
    """Create and seed the ``prices`` table if it is missing or empty.

    Idempotent and race-safe: ``CREATE TABLE IF NOT EXISTS``, then — only when the
    table has no rows — ``INSERT OR IGNORE`` the ``_SEED`` rows and commit. Called
    once by ``launch`` before serving; ``_price`` is a pure reader and never seeds.
    """
    with closing(_connect()) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price INTEGER)"
        )
        if conn.execute("SELECT 1 FROM prices LIMIT 1").fetchone() is None:
            conn.executemany(
                "INSERT OR IGNORE INTO prices (city, price) VALUES (?, ?)",
                list(_SEED.items()),
            )
            conn.commit()


def _price(location: str) -> str:
    """Return a dummy ticket price to ``location``, formatted as ``"<n> EUR"``.

    A pure reader: it looks the lowercased, stripped city up in the ``prices``
    table (seeded on launch by ``_ensure_db``). Unknown cities get a deterministic
    fallback derived from the name, so repeated calls are stable and the demo needs
    no real pricing backend. Assumes the table exists — a ``SELECT`` against an
    unseeded database raises ``sqlite3.OperationalError``, and ``launch`` seeds
    before any read. This is the plain, directly-testable core wrapped by the
    ``get_airline_price`` tool.
    """
    key = location.strip().lower()
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT price FROM prices WHERE city = ?", (key,)
        ).fetchone()
    price = row[0] if row is not None else 100 + sum(ord(c) for c in key) % 900
    return f"{price} EUR"


@beta_tool
def get_airline_price(location: str) -> str:
    """Get the price of an airline ticket to a destination city, in euros (EUR).

    Call this whenever the user asks the price or cost of a flight to a place, or
    asks how much it is to fly somewhere.

    Args:
        location: The destination city, e.g. 'London' or 'Tokyo'.
    """
    return _price(location)


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

    model: str = "claude-sonnet-5"
    system: str = SYSTEM
    max_tokens: int = 1024
    client: Anthropic = field(default_factory=Anthropic)

    @staticmethod
    def _to_messages(
        message: str, history: list[dict[str, Any]]
    ) -> list[BetaMessageParam]:
        """Map a Gradio messages-format history plus the new user ``message`` to
        an Anthropic (beta) ``messages`` list for the tool runner.

        Gradio history entries are ``{"role": "user"|"assistant", "content": str}``
        dicts — the same shape as ``BetaMessageParam`` — so each maps directly.
        Only ``user``/``assistant`` turns with non-empty content are kept (any
        metadata/tool entries are skipped). The system prompt is *not* included
        here; it is passed separately via ``system=``.
        """
        messages: list[BetaMessageParam] = [
            BetaMessageParam(
                role=cast(Literal["user", "assistant"], turn["role"]),
                content=turn["content"],
            )
            for turn in history
            if turn.get("role") in ("user", "assistant") and turn.get("content")
        ]
        messages.append(BetaMessageParam(role="user", content=message))
        return messages

    def chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]:
        """Stream Claude's reply to ``message``, running the price tool as needed.

        The SDK tool runner drives the loop: with ``stream=True`` each iteration
        yields one turn's message stream, and when Claude calls
        ``get_airline_price`` the runner runs it and continues to the next turn
        automatically. ``reply`` accumulates across every turn so
        ``gr.ChatInterface`` sees one growing string and never resets the text.
        """
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=[get_airline_price],
            messages=self._to_messages(message, history),
            stream=True,
        )
        reply = ""
        for stream in runner:
            for text in stream.text_stream:
                reply += text
                yield reply


def build_demo() -> gr.Blocks:
    """Build the UI: a multi-turn airline chat streamed from Claude with tool use."""
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        title="llm-engineering — airline tickets",
    )


def launch(**kwargs: Any) -> None:
    """Load environment variables, seed the price database, build the demo, and serve.

    ``load_dotenv`` runs here (not at import) so ``ANTHROPIC_API_KEY`` is available
    before ``Bot`` builds its ``Anthropic`` client. ``_ensure_db`` then creates and
    seeds the SQLite price database before serving, so the first request reads a
    ready table (``_price`` never seeds). Matches ``gradio_app/chatinterface.py``;
    importing this module has no side effects.
    """
    load_dotenv()
    _ensure_db()
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
```

- [ ] **Step 3: Run the SQLite smoke check to verify it PASSES**

Run:
```bash
rm -f gradio_app/airline_prices.db
uv run python -c "
import os, sqlite3
import gradio_app.airline as m
assert not os.path.exists(m._DB_PATH), f'DB created at import: {m._DB_PATH}'
try:
    m._price('London')
    raise AssertionError('_price should raise on an unseeded DB')
except sqlite3.OperationalError:
    pass
m._ensure_db()
assert m._price('London') == '120 EUR', m._price('London')
assert m._price(' tokyo ') == '720 EUR', m._price(' tokyo ')
assert m._price('London') == m._price('LONDON'), 'not case-insensitive'
r1, r2 = m._price('Reykjavik'), m._price('Reykjavik')
assert r1 == r2 and r1.endswith(' EUR'), r1
m._ensure_db()
assert m._price('Paris') == '95 EUR', m._price('Paris')
assert m.get_airline_price('Berlin') == '110 EUR', m.get_airline_price('Berlin')
print('sqlite pricing OK', r1)
"
```
Expected: **PASS**, prints `sqlite pricing OK 176 EUR`. Confirms: import creates no DB, an unseeded read raises, seeding then reading returns the seed prices, case-insensitivity and whitespace stripping, a stable fallback for unknown cities, `_ensure_db` is idempotent, and the `@beta_tool` delegates to `_price`.

- [ ] **Step 4: Ignore the generated DB in `.gitignore`**

Append this block to the end of `.gitignore` (after the existing `# Gradio` / `.gradio/` block):
```gitignore

# SQLite
gradio_app/airline_prices.db
```

- [ ] **Step 5: Verify the DB is ignored and unstaged**

Run:
```bash
git check-ignore gradio_app/airline_prices.db && git status --porcelain gradio_app/airline_prices.db
```
Expected: `git check-ignore` prints `gradio_app/airline_prices.db` (exit 0 — it is ignored); `git status --porcelain` prints **nothing** (the DB is not tracked and not shown as untracked). If instead the DB appears in `git status`, the `.gitignore` entry is wrong — fix it before continuing.

- [ ] **Step 6: Verify `build_demo()` still constructs a `gr.Blocks` (no network)**

Run:
```bash
uv run python -c "from dotenv import load_dotenv; load_dotenv(); import gradio as gr; from gradio_app.airline import build_demo; d = build_demo(); assert isinstance(d, gr.Blocks), type(d); print('build_demo OK')"
```
Expected: **PASS**, prints `build_demo OK`. (`gr.ChatInterface` subclasses `gr.Blocks`; building `Bot()` reads `ANTHROPIC_API_KEY` via `load_dotenv` but makes no network call — client construction is lazy. Requires `.env` with `ANTHROPIC_API_KEY`.)

- [ ] **Step 7: Lint and type-check**

Run:
```bash
uv run ruff format gradio_app/airline.py
uv run ruff check gradio_app/airline.py
uv run mypy gradio_app/airline.py
```
Expected: all pass (`ruff format` may reformat; `ruff check` prints `All checks passed!`; `mypy` prints `Success: no issues found in 1 source file`). If `ruff format` changes the file, re-run the Step 3 smoke check to confirm it still passes.

- [ ] **Step 8: Commit** (explicit paths — never stage the generated DB)

```bash
git add gradio_app/airline.py .gitignore
git commit -m "Back airline price lookup with a SQLite DB seeded on launch"
```

---

### Task 2: Update the README to describe the SQLite-backed pricing

**Files:**
- Modify: `README.md` (the "Airline ticket chat" subsection under "## Web UI", around lines 66–77)

**Interfaces:**
- Consumes: the behavior from Task 1 (prices come from a SQLite DB seeded on launch).
- Produces: nothing (docs only).

- [ ] **Step 1: Update the tool sentence**

In `README.md`, replace:
```markdown
tool that returns a dummy price in euros; the reply streams back through the
```
with:
```markdown
tool that looks the price up in a small SQLite database and returns it in euros;
the reply streams back through the
```

- [ ] **Step 2: Update the closing "placeholder data" sentence**

In `README.md`, replace:
```markdown
single-model app above) — no OpenRouter key needed. Prices are placeholder data,
not a real fare lookup.
```
with:
```markdown
single-model app above) — no OpenRouter key needed. Prices are placeholder data
in a local SQLite database (`airline_prices.db`, created and seeded on launch),
not a real fare lookup.
```

- [ ] **Step 3: Verify the edit reads correctly**

Run:
```bash
uv run python -c "t = open('README.md').read(); assert 'Airline ticket chat' in t and 'gradio_app.airline' in t and 'SQLite' in t and 'airline_prices.db' in t; print('README OK')"
```
Expected: **PASS**, prints `README OK`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Note SQLite-backed airline pricing in README"
```

---

## Verification (whole change)

After both tasks, confirm end to end:

1. `uv run ruff check gradio_app/airline.py` and `uv run mypy gradio_app/airline.py` pass.
2. **Import has no side effects** (including no DB file): `rm -f gradio_app/airline_prices.db && uv run python -c "import os, gradio_app.airline as m; assert not os.path.exists(m._DB_PATH); print('import OK')"` prints `import OK`.
3. **Pricing smoke:** the Task 1 Step 3 check passes (`sqlite pricing OK 176 EUR`).
4. **DB is gitignored:** `git status --porcelain` shows no `airline_prices.db` entry after the app has run/seeded.
5. **Manual (hits the Anthropic API):** `uv run python -m gradio_app.airline`, then ask "how much is a flight to Tokyo?" — confirm the bot **calls the tool** and streams a **short** answer containing `720 EUR`; ask a follow-up like "and to Rome?" and confirm it stays concise, reports `140 EUR`, and remembers earlier turns. Run once by hand; requires `ANTHROPIC_API_KEY` in `.env`.

## Notes / Out of scope

- No schema beyond `prices(city TEXT PRIMARY KEY, price INTEGER)`; no ORM, no migrations, no connection pooling — one connection per lookup is fine for a local demo.
- Still dummy data — no real fare backend, no extra tool params (origin/dates/passengers/cabin), no additional tools/retrieval/booking, no cross-session chat persistence (all deferred per the spec).
- No changes to `Bot`, `_to_messages`, `chat`, the `@beta_tool` definition, `build_demo`, or any other module (`app.py`, `chatinterface.py`, `multibot.py`, `website_summarizer.py`, `__init__.py`, `helpers.py`, `main.py`, `pyproject.toml`, `openrouter/`, `claude/`).
- No test framework added — verification stays smoke-check based, matching the repo.