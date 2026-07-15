# Design: `gradio_app` airline ticket chat with tool use (`gr.ChatInterface`)

**Date:** 2026-07-14
**Status:** Approved

> **Update (2026-07-14):** After the initial implementation (a hand-written
> streaming tool loop), the module was refactored to use the SDK's **tool
> runner** helpers — `get_airline_price` is a `@beta_tool` (schema derived from
> its signature/docstring) and `chat` drives the loop with
> `client.beta.messages.tool_runner(..., stream=True)`, iterating one message
> stream per turn. This deletes the manual `TOOL` dict, the `TOOLS` `ClassVar`,
> the `stop_reason`/`tool_result` bookkeeping, and the manual `while` loop. The
> pricing logic lives in a plain `_price()` helper wrapped by the tool (so it
> stays directly unit-checkable), and the beta path uses `BetaMessageParam`. The
> tradeoff is a dependency on the **beta** `tool_runner`/`@beta_tool` surface.
> The sections below describe the original manual-loop design.

> **Update (2026-07-15): SQLite-backed pricing.** The in-memory `_PRICES` dict is
> replaced by a **SQLite** database that `_price()` reads. The public surface is
> unchanged — `get_airline_price` (the `@beta_tool`) still wraps the plain
> `_price()` core, prices for the seven seed cities are identical (`"London"` →
> `"120 EUR"`), and unknown cities still get the deterministic char-sum fallback.
> Only `gradio_app/airline.py` and `.gitignore` change; `sqlite3` is stdlib, so no
> new dependency.
>
> - **DB file:** `_DB_PATH = Path(__file__).with_name("airline_prices.db")` — a
>   module constant (no I/O at import, so importing stays side-effect-free). The
>   file is **generated, not committed**; `.gitignore` gains a `# SQLite` section
>   ignoring `gradio_app/airline_prices.db`.
> - **Schema & seed:** one table, `prices(city TEXT PRIMARY KEY, price INTEGER)`.
>   The former dict becomes `_SEED: dict[str, int]` (the same seven city→price
>   rows), used only as seed data.
> - **Seeding — launch only.** A helper `_ensure_db()` opens a connection,
>   `CREATE TABLE IF NOT EXISTS prices (...)`, and, **only if the table is empty**,
>   `INSERT OR IGNORE`s the seed rows and commits (idempotent, race-safe).
>   `launch()` calls `_ensure_db()` once — after `load_dotenv()`, before
>   `build_demo()` — so the DB is auto-created and seeded on startup. `_price()`
>   does **not** seed.
> - **`_price()` is a pure reader.** It opens a connection via
>   `contextlib.closing(_connect())` (a raw `sqlite3` connection used as a plain
>   context manager commits but does **not** close, so `closing` is required),
>   runs `SELECT price FROM prices WHERE city = ?` on the lowercased/stripped key,
>   and returns `"<n> EUR"` if found or the deterministic
>   `100 + sum(ord(c) for c in key) % 900` fallback otherwise. It never creates or
>   seeds the table.
>   ```python
>   def _price(location: str) -> str:
>       key = location.strip().lower()
>       with closing(_connect()) as conn:
>           row = conn.execute(
>               "SELECT price FROM prices WHERE city = ?", (key,)
>           ).fetchone()
>       price = row[0] if row is not None else 100 + sum(ord(c) for c in key) % 900
>       return f"{price} EUR"
>   ```
> - **Contract:** the DB must be seeded (via `launch()` → `_ensure_db()`) before
>   any read. Because `_price()` is a pure reader, a `SELECT` against a DB with no
>   `prices` table raises `sqlite3.OperationalError` — so a direct-call smoke check
>   (e.g. `get_airline_price("London")`) must call `_ensure_db()` first. In the
>   running app this is automatic: `launch()` always seeds before serving.
> - **Threading:** one connection **per lookup**, opened and closed inside
>   `_price()` / `_ensure_db()`. This sidesteps SQLite's default same-thread
>   restriction (Gradio and the tool runner may call from worker threads) with no
>   shared connection and no `check_same_thread` juggling.
> - **Docs:** the module docstring and the README airline subsection change "a
>   small known-city table" / "dummy … table" wording to say prices come from a
>   **SQLite DB seeded on launch** (still placeholder data, not a real fare
>   backend).
>
> The "Verification" section below is updated in-place for the SQLite change
> (seed before the direct `_price`/tool smoke check).

## Goal

Add a Gradio chat app for an airline ticket assistant, in a new module
`gradio_app/airline.py`. It is the direct sibling of
`gradio_app/chatinterface.py` (the guitar-store chat) — a single
**native-Anthropic** `Bot` reached through the Claude SDK on
**`gr.ChatInterface`**, reading `ANTHROPIC_API_KEY`, streaming replies — with one
new capability: a **tool call**, `get_airline_price`, that returns a ticket price
in euros (EUR) for a destination.

The assistant replies **short and concise**, and it must call the tool for prices
rather than inventing them. `get_airline_price` returns **dummy data** for now
(no real pricing backend).

New/changed files: `gradio_app/airline.py` (new) and `README.md` (add a section).
`gradio_app/__init__.py`, `app.py`, `chatinterface.py`, `multibot.py`,
`website_summarizer.py`, `helpers.py`, `main.py`, and `pyproject.toml` are
untouched — the native Anthropic SDK is already a dependency.

## `get_airline_price` — the tool implementation

A plain module-level function returning **dummy** EUR prices. A small dict covers
a handful of known destinations; anything else falls back to a **deterministic**
value derived from the location string, so unknown cities still vary and the demo
stays reproducible (no randomness). The lookup is case-insensitive.

```python
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

    Known cities come from ``_PRICES``; unknown locations get a deterministic
    fallback derived from the name so the demo is reproducible.
    """
    key = location.strip().lower()
    if key in _PRICES:
        price = _PRICES[key]
    else:
        # Deterministic pseudo-price in a sensible range for unknown cities.
        price = 100 + sum(ord(c) for c in key) % 900
    return f"{price} EUR"
```

Rationale:

- Returns a **string** — the shape a `tool_result` block's `content` expects. The
  model turns it into a short sentence for the user.
- Deterministic fallback (char-sum, not `random`) keeps repeated runs stable and
  side-effect-free, matching the repo's no-surprises style.

## Tool definition

The Anthropic tool schema. The description is **prescriptive about when to call
it**, which improves the should-call rate on recent Claude models.

```python
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
```

## `Bot` dataclass

Mirrors `chatinterface.py`'s native-Claude `Bot`: a frozen dataclass owning its
own `Anthropic` client via `default_factory`. The persona lives on the class as a
`ClassVar[str]` (excluded from `__init__`); the `system` instance field defaults
to it. The tool list is also a `ClassVar` (deterministic, never per-instance).

```python
@dataclass(frozen=True)
class Bot:
    SYSTEM: ClassVar[str] = (
        "You are a helpful airline ticket assistant. Help customers with flights "
        "and ticket prices. Keep every reply short and concise — a sentence or "
        "two. When a customer asks the price or cost of a flight to a place, call "
        "the get_airline_price tool to look it up; never guess or invent a price. "
        "Prices are in euros (EUR)."
    )
    TOOLS: ClassVar[list[ToolParam]] = [TOOL]

    model: str = "claude-sonnet-5"     # same default as chatinterface.py
    system: str = SYSTEM               # instance field defaults to the class constant
    max_tokens: int = 1024
    client: Anthropic = field(default_factory=Anthropic)
```

`_to_messages(message, history)` is **identical** to the sibling: it maps Gradio's
`{"role", "content"}` history dicts plus the new message 1:1 to Anthropic
`MessageParam`s, keeping only `user`/`assistant` turns with non-empty content, and
leaves the system prompt out (it goes in `system=`). A small `cast(...)`/`Literal`
keeps mypy satisfied against `MessageParam`'s `Literal["user", "assistant"]` role.

## Data flow — the streaming tool loop

`chat(message, history)` is a generator that `gr.ChatInterface` drives. It streams
turn 1; if Claude calls the tool, it runs it, appends the result, and streams the
next turn — accumulating text across turns into one growing string so the UI never
resets.

```python
def chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]:
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

        messages.append({"role": "assistant", "content": final.content})
        if final.stop_reason != "tool_use":
            return

        tool_results: list[ToolResultBlockParam] = []
        for block in final.content:
            if block.type == "tool_use":
                result = get_airline_price(**block.input)   # only tool: location
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})
```

Rationale:

- **`reply` accumulates across the whole loop.** Turn 1 usually emits little or no
  text before the tool call; turn 2 streams the actual answer. Keeping one growing
  string means the displayed text grows smoothly and never resets between turns.
- **`get_final_message()`** gives the full assistant message (text + `tool_use`
  blocks) and the `stop_reason` — the standard manual streaming-plus-tools loop
  from the Anthropic SDK.
- Append the assistant's **full `content`** (not just text) so the `tool_use`
  blocks are preserved, then send all `tool_result` blocks back in a single `user`
  message keyed by matching `tool_use_id`.
- The **tool round-trip lives entirely inside one `chat()` call.** Gradio's history
  only ever stores the final assistant text, so subsequent turns map cleanly as
  plain user/assistant `MessageParam`s — no tool blocks leak into `history`.
- `get_airline_price(**block.input)` unpacks the validated `{"location": ...}`
  input; the schema's `required: ["location"]` guarantees the key is present.

## UI

`build_demo() -> gr.Blocks` constructs one `Bot` and wires a `gr.ChatInterface`:

```python
def build_demo() -> gr.Blocks:
    bot = Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        type="messages",
        title="llm-engineering — airline tickets",
    )
```

- **`type="messages"`** is set explicitly so `history` arrives as role/content
  dicts, matching the `MessageParam` mapping.
- `gr.ChatInterface` owns the transcript, input textbox, and multi-turn history —
  no manual state wiring. Same lightweight `build_demo()`/`launch()` factory shape
  as `chatinterface.py` (no app-class wrapper — this is a single bot).

## Bootstrap

Identical to `chatinterface.py`:

```python
def launch(**kwargs: Any) -> None:
    load_dotenv()          # so ANTHROPIC_API_KEY is present before Bot builds its client
    build_demo().launch(**kwargs)

if __name__ == "__main__":
    launch()
```

`load_dotenv()` runs in `launch()` — not at import — before `build_demo()` builds
the `Bot` (whose `Anthropic()` reads `ANTHROPIC_API_KEY`). Importing the module
has no side effects. Run with `uv run python -m gradio_app.airline`.

## README

Add a short "Airline ticket chat" subsection under **Web UI**, matching the
existing entries: what it does (multi-turn airline assistant, native Claude, a
`get_airline_price` tool returning dummy EUR prices), the run command
(`uv run python -m gradio_app.airline`), and that it uses `ANTHROPIC_API_KEY`.

## Error handling

No extra guard layer — consistent with the sibling. `gr.ChatInterface` won't send
an empty message; `get_airline_price` never raises (unknown locations hit the
deterministic fallback); Anthropic SDK errors during `chat` propagate for Gradio
to surface. A `max_iterations`-style loop cap is unnecessary: there is one small
tool and the model resolves the price in a single round-trip.

## Verification

Repo convention is no test framework — verify with import/run smoke checks:

1. `import gradio_app.airline` succeeds with **no side effects** (no client built,
   no `.env` read at import, no DB file created at import).
2. After `_ensure_db()` seeds the database (required — `_price` is a pure reader),
   `get_airline_price("London")` returns `"120 EUR"`; an unknown city returns a
   stable `"<n> EUR"` across repeated calls; the lookup is case-insensitive.
3. `Bot.SYSTEM` and `Bot.TOOLS` are class attributes (not `__init__` params); a
   default `Bot()`'s `.system` equals `Bot.SYSTEM`.
4. `_to_messages` maps a sample `history` plus a new message to the expected
   `MessageParam` ordering — checked without hitting the network.
5. `build_demo()` returns a `gr.Blocks`.
6. `ruff format`, `ruff check`, and `mypy` pass on the new module.
7. Launch `uv run python -m gradio_app.airline`, ask "how much is a flight to
   Tokyo?", and confirm the bot calls the tool and streams a short answer with the
   EUR price (this step hits the Anthropic API; run once by hand).

## Out of scope

- Any model selector / dropdown — this is a single native-Claude bot (unlike
  `multibot.py`).
- A real pricing backend — `get_airline_price` returns dummy data only.
- More tool parameters (origin, dates, passengers, cabin class) — the tool takes a
  single `location`, as requested.
- Additional tools, retrieval, or booking.
- Persisting conversations across sessions (in-session history only, held by
  `gr.ChatInterface`).
- Changes to `gradio_app/app.py`, `chatinterface.py`, `multibot.py`,
  `website_summarizer.py`, `__init__.py`, `helpers.py`, `main.py`,
  `pyproject.toml`, `openrouter/`, or `claude/`.
- Adding a test framework (pytest) — verification stays smoke-check based.