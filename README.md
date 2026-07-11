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

A small Gradio app lives in the `gradio_app` package: type a message and
Claude Sonnet's reply streams back through the Anthropic API. Launch it with:

```bash
uv run python -m gradio_app.app
```

This opens a local page in your browser. It reads `ANTHROPIC_API_KEY` from your
`.env`, so add that key alongside `OPENROUTER_API_KEY`.

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