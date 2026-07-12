# Design: `gradio_app` multibot selector

**Date:** 2026-07-12
**Status:** Approved

## Goal

Add a model selector to `gradio_app` so the user can pick which bot answers —
**GPT**, **Claude**, or **DeepSeek** — from a dropdown. All three are reached
through a single uniform client: the OpenAI SDK pointed at OpenRouter's
OpenAI-compatible API (the same client `main.py` and `openrouter/chat.py` already
use). Selecting a bot just swaps the OpenRouter model ID.

This lives in a **new module**, `gradio_app/multibot.py`, leaving the existing
`gradio_app/app.py` (the single-Claude streaming app) untouched. Claude is
reached as an OpenRouter model ID like the other two. Only
`gradio_app/multibot.py` (new) and `README.md` change; `gradio_app/__init__.py`
keeps exporting the original app.

## Model selection: an enum + the `Bot` dataclass

The set of selectable models is a `Model` enum. Each member's **name** is the
dropdown label; its **value** is the OpenRouter model ID:

```python
class Model(Enum):
    GPT = "openai/gpt-4o-mini"
    Claude = "anthropic/claude-sonnet-4.5"
    DeepSeek = "deepseek/deepseek-chat"
```

Member names double as display labels (`GPT`, `Claude`, `DeepSeek`), so no
separate label→id map is needed and the selection is inspectable without a
client. GPT and Claude IDs match those already used in `openrouter/chat.py`;
DeepSeek is the standard chat ID.

The `Bot` frozen dataclass is the uniform client interface. It pairs a chosen
`Model` with an OpenRouter client and a system prompt, and streams the reply.
Each `Bot` owns its client via a `default_factory` — mirroring the `Bot` in
`gradio_app/app.py`:

```python
@dataclass(frozen=True)
class Bot:
    @staticmethod
    def _client() -> OpenAI:
        load_dotenv()  # so OPENROUTER_API_KEY is present
        return OpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"])

    model: Model
    client: OpenAI = field(default_factory=_client)
    system: str = "You are a helpful assistant."

    def chat(self, message: str) -> Iterator[str]:
        # OpenRouter (OpenAI SDK) streaming — identical for every model
        stream = self.client.chat.completions.create(
            model=self.model.value, stream=True, messages=[...],
        )
        reply = ""
        for chunk in stream:
            ...
            yield reply
```

`chat` yields the reply accumulated so far on each chunk, preserving the
streaming contract `gradio_app/app.py` already gives Gradio. `_client` is a
`@staticmethod` used as the field's `default_factory`: it calls `load_dotenv`
(so `OPENROUTER_API_KEY` is present) and builds an `OpenAI` client aimed at
`https://openrouter.ai/api/v1`, reading the key explicitly because the OpenAI
SDK's default env var is `OPENAI_API_KEY`.

## UI

`build_demo() -> gr.Blocks` constructs one `Bot` per `Model` member (each builds
its own client via the `default_factory`), keyed by label, then wires a
`gr.Interface`:

- **inputs:** `gr.Dropdown(choices=[m.name for m in Model], value=Model.Claude.name, label="Model")`
  and `gr.Textbox(label="You")`.
- **output:** `gr.Textbox(label="Reply")`.
- **fn:** `respond(name, message)` looks up `bots[name]` and `yield from`
  `bot.chat(message)`, so streaming still works.
- `flagging_mode="never"` is kept.

Default selection is **Claude**, preserving `gradio_app/app.py`'s behavior. Bots
are constructed inside `build_demo` (not at import), so each client — and its
`load_dotenv` call — runs lazily when the UI is built. Importing the module has
no side effects (mirroring `openrouter/chat.py`).

`launch(**kwargs)` just calls `build_demo().launch(**kwargs)` — the env is loaded
by each bot's client factory, not here — with an `if __name__ == "__main__"`
entry so `python -m gradio_app.multibot` serves it.

## Error handling

Unchanged in spirit — OpenRouter/OpenAI SDK errors propagate and Gradio surfaces
them. No new try/except layer.

## Verification

Consistent with the repo convention (no test framework — verify with import/run
smoke checks, per the scaffold plan):

1. `{m.name: m.value for m in Model}` maps each label to the expected OpenRouter
   model ID.
2. `Bot(Model.Claude, client=<dummy>)` exposes the right `.model.value`; and
   `build_demo()` returns a `gr.Blocks` (each bot's factory calls `load_dotenv`
   and builds an `OpenAI` client — no network call until a message is sent).
3. `ruff format`, `ruff check`, and `mypy gradio_app` all pass.
4. Launch `uv run python -m gradio_app.multibot`, confirm the page serves and the
   dropdown switches between the three bots.

## Out of scope

- Multi-turn conversation history (each request is a single user turn, as today).
- Per-bot personas / distinct system prompts (all share one default).
- Changes to `gradio_app/app.py` (the existing Claude app) or `gradio_app/__init__.py`.
- Changes to `main.py`, `openrouter/`, or `claude/`.
- Adding a test framework (pytest) — verification stays smoke-check based.