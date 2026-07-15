# llm-engineering

Experiments with LLMs through [OpenRouter](https://openrouter.ai), using the OpenAI SDK pointed at OpenRouter's OpenAI-compatible API. One key, many models: `main.py` sends the same prompt to several models (GPT, Claude, Gemini, Llama, DeepSeek) and prints their answers side by side.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- An [OpenRouter API key](https://openrouter.ai/keys)

## Setup

```bash
uv sync
```

Create a `.env` file in the project root (it's gitignored):

```
OPENROUTER_API_KEY=sk-or-...
```

## Usage

```bash
uv run main.py
```

To try other models, add their IDs from <https://openrouter.ai/models> to the `MODELS` list in `main.py`.

## Web UI

Two Gradio apps live in the `gradio_app` package; both open a local page in your
browser.

**Single-model (Claude).** Type a message and Claude Sonnet's reply streams back
through the Anthropic API:

```bash
uv run python -m gradio_app.app
```

It reads `ANTHROPIC_API_KEY` from your `.env`, so add that key alongside
`OPENROUTER_API_KEY`.

**Multibot selector.** Pick GPT, Claude, or DeepSeek from a dropdown; the reply
streams back through OpenRouter:

```bash
uv run python -m gradio_app.multibot
```

It uses `OPENROUTER_API_KEY` — no Anthropic key needed.

**Website summarizer.** Paste a company's website URL and pick GPT, Claude, or
DeepSeek; the landing page is fetched with BeautifulSoup and a short Markdown
summary streams back:

```bash
uv run python -m gradio_app.website_summarizer
```

It uses `OPENROUTER_API_KEY` too. Only the given page is fetched — links are not
followed.

**Airline ticket chat.** A multi-turn airline-ticket assistant built on
`gr.ChatInterface`. Ask what a flight costs and Claude calls a `get_airline_price`
tool that looks the price up in a small SQLite database and returns it in euros;
the reply streams back through the Anthropic API:

```bash
uv run python -m gradio_app.airline
```

It reads `ANTHROPIC_API_KEY` from your `.env` (native Claude, like the
single-model app above) — no OpenRouter key needed. Prices are placeholder data
in a local SQLite database (`airline_prices.db`, created and seeded on launch by
an injected `PriceStore`), not a real fare lookup.

## Development

Formatting, linting, and type checking:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy main.py
```

A pre-commit hook formats staged Python files with ruff. Enable it once per clone:

```bash
uv run pre-commit install
```