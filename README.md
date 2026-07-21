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

**Airline assistant with image & voice.** A multimodal take on the airline
assistant with three boxes — chat, a generated image of the destination city, and
a spoken (TTS) rendering of every reply. Everything runs through a single
OpenAI-SDK client pointed at OpenRouter: chat + function calling
(`openai/gpt-4o-mini`), the price tool (the same SQLite dummy prices), image
generation (`openai/gpt-image-1`), and voice (`hexgrad/kokoro-82m` TTS —
OpenRouter has no OpenAI TTS model). Ask to fly somewhere and the bot calls
`get_airline_price`, generates an image of that city, and speaks the reply:

```bash
uv run python -m gradio_app.airline_multimodal
```

It reads `OPENROUTER_API_KEY` from your `.env` — one key covers all three
modalities. Self-contained (no imports from `gradio_app.airline`); the image
(PIL) and voice (bytes) outputs are served straight from Gradio's cache, so there
are no temp files.

## Notebooks

**Open-model tour** (`notebooks/hf_open_models_tour.ipynb`). Runs five open Hugging
Face models — Llama 3.2 1B, Phi-4-mini, Gemma 3 270M, Qwen3-4B and the
DeepSeek-R1-Distill reasoning model — against the same prompt, at the tokenizer /
`AutoModelForCausalLM` level rather than behind a `pipeline`. Each is loaded in
4-bit and freed again before the next, so all five fit comfortably.

Built for **Google Colab on a T4 GPU**, not for local execution — open it via
*File → Open notebook → GitHub*, or upload it. It needs an `HF_TOKEN` in Colab's
secrets and accepted licences for the two gated models (Llama 3.2 and Gemma 3);
the other three run without a token. The notebook's first cell walks through both.

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