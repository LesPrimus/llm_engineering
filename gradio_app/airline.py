"""A multi-turn Gradio chat app: an airline ticket assistant with tool use.

Chat with a concise airline-ticket assistant and ask what a flight costs — the
bot calls a ``get_airline_price`` tool that returns a price in euros. Built on
``gr.ChatInterface`` (which owns the transcript and history) and backed by the
native Anthropic SDK, so replies stream straight from Claude.

This is the sibling of ``gradio_app/chatinterface.py`` — same self-contained
``Bot`` dataclass that owns its own ``Anthropic`` client, on ``gr.ChatInterface``
— but it declares a tool. Prices live in an injected ``PriceStore`` dataclass that
owns every SQLite concern (path, connection, seeding, lookup); ``Bot`` holds one
and asks it for prices. Instead of a hand-written tool loop it uses the SDK's
**tool runner**: ``chat`` wraps ``Bot``'s ``get_airline_price`` method — bound to
the injected store — with ``beta_tool`` (its schema is derived from the signature
and docstring), and ``client.beta.messages.tool_runner(..., stream=True)`` drives
the loop — each iteration yields one turn's message stream, and the runner runs
the tool and continues automatically. ``chat`` accumulates text across turns into
one growing string so the UI never resets. The persona (``SYSTEM``) lives on
``Bot`` as a ``ClassVar``. The bot is built inside ``build_demo`` (not at import),
so importing this module has no side effects.

``PriceStore`` returns dummy data from a small **SQLite** database (a seed table of
known cities plus a deterministic fallback) — there is no real pricing backend.
The database (``airline_prices.db``, next to this module) is created and seeded by
``launch`` before serving; ``PriceStore.price`` only reads it. The tool runner is a
**beta** SDK helper. Run it with ``uv run python -m gradio_app.airline``. It reads
``ANTHROPIC_API_KEY`` from ``.env`` (native Claude, like ``gradio_app.chatinterface``).
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


@dataclass(frozen=True)
class PriceStore:
    """SQLite-backed store of dummy EUR ticket prices, injected into ``Bot``.

    Owns every database concern — the file path, the connection, seeding, and the
    price lookup — so ``Bot`` just holds one and asks it for prices. ``price`` is a
    pure reader; ``ensure_seeded`` (called by ``launch``) is the only writer.
    """

    # Seed rows. Cities not listed get a deterministic fallback in ``price``, so
    # unknown cities still vary and the demo stays reproducible.
    SEED: ClassVar[dict[str, int]] = {
        "london": 120,
        "paris": 95,
        "rome": 140,
        "berlin": 110,
        "new york": 480,
        "tokyo": 720,
        "sydney": 910,
    }
    # Default database file, next to this module. Building the path does no I/O, so
    # importing this module stays side-effect-free; the file itself is gitignored.
    DEFAULT_PATH: ClassVar[Path] = Path(__file__).with_name("airline_prices.db")

    path: Path = DEFAULT_PATH

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to ``path`` — one per call, so lookups from Gradio's
        worker threads never share a connection (SQLite's same-thread rule)."""
        return sqlite3.connect(self.path)

    def ensure_seeded(self) -> None:
        """Create and seed the ``prices`` table if it is missing or empty.

        Idempotent and race-safe: ``CREATE TABLE IF NOT EXISTS``, then — only when
        the table has no rows — ``INSERT OR IGNORE`` the ``SEED`` rows and commit.
        Called once by ``launch`` before serving; ``price`` never seeds.
        """
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
        """Return a dummy ticket price to ``location``, formatted as ``"<n> EUR"``.

        A pure reader: it looks the lowercased, stripped city up in the ``prices``
        table (seeded on launch by ``ensure_seeded``). Unknown cities get a
        deterministic fallback derived from the name, so repeated calls are stable
        and the demo needs no real pricing backend. Assumes the table exists — a
        ``SELECT`` against an unseeded database raises ``sqlite3.OperationalError``,
        and ``launch`` seeds before any read.
        """
        key = location.strip().lower()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT price FROM prices WHERE city = ?", (key,)
            ).fetchone()
        price = row[0] if row is not None else 100 + sum(ord(c) for c in key) % 900
        return f"{price} EUR"


@dataclass(frozen=True)
class Bot:
    """An airline-ticket chatbot backed by Claude and an injected price store."""

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
    prices: PriceStore = field(default_factory=PriceStore)

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

    def get_airline_price(self, location: str) -> str:
        """Get the price of an airline ticket to a destination city, in euros (EUR).

        Call this whenever the user asks the price or cost of a flight to a place, or
        asks how much it is to fly somewhere.

        Args:
            location: The destination city, e.g. 'London' or 'Tokyo'.
        """
        return self.prices.price(location)

    def chat(self, message: str, history: list[dict[str, Any]]) -> Iterator[str]:
        """Stream Claude's reply to ``message``, running the price tool as needed.

        Wraps the bound ``get_airline_price`` method (which reads the injected
        ``PriceStore``) with ``beta_tool``, then lets the SDK tool runner drive the
        loop: with ``stream=True`` each iteration yields one turn's message stream,
        and when Claude calls the tool the runner runs it and continues
        automatically. ``reply`` accumulates across every turn so
        ``gr.ChatInterface`` sees one growing string and never resets.
        """
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=[beta_tool(self.get_airline_price)],
            messages=self._to_messages(message, history),
            stream=True,
        )
        reply = ""
        for stream in runner:
            for text in stream.text_stream:
                reply += text
                yield reply


def build_demo(bot: Bot | None = None) -> gr.Blocks:
    """Build the UI: a multi-turn airline chat streamed from Claude with tool use.

    Accepts an optional pre-built ``Bot`` so ``launch`` can seed that bot's price
    store before serving; defaults to a fresh ``Bot`` for import-time smoke checks.
    """
    bot = bot if bot is not None else Bot()
    return gr.ChatInterface(
        fn=bot.chat,
        title="llm-engineering — airline tickets",
    )


def launch(**kwargs: Any) -> None:
    """Load environment variables, seed the price database, build the demo, and serve.

    ``load_dotenv`` runs here (not at import) so ``ANTHROPIC_API_KEY`` is available
    before ``Bot`` builds its ``Anthropic`` client. The bot's injected
    ``PriceStore`` is seeded (``ensure_seeded``) before serving, so the first
    request reads a ready table (``price`` never seeds). Importing this module has
    no side effects.
    """
    load_dotenv()
    bot = Bot()
    bot.prices.ensure_seeded()
    build_demo(bot).launch(**kwargs)


if __name__ == "__main__":
    launch()
