# Design: `gradio_app` scaffold

**Date:** 2026-07-11
**Status:** Approved

## Goal

Install Gradio and add a new `gradio_app/` package (peer to the existing
`openrouter/` and `claude/` packages) containing a **runnable placeholder** UI.
No specific application yet — this is a scaffold that proves Gradio is installed
and wired correctly, and gives a clear place to build the real UI into later.

## Dependency

- `uv add gradio` — adds `gradio` to `[project.dependencies]` in
  `pyproject.toml` and updates `uv.lock`.

## Package layout

Peer to `openrouter/` and `claude/`:

```
gradio_app/
├── __init__.py    # from .app import build_demo, launch;  __all__ = ["build_demo", "launch"]
├── app.py         # build_demo() -> gr.Blocks placeholder;  launch(**kwargs)
└── __main__.py    # launch()  →  enables `uv run python -m gradio_app`
```

## Behavior

- **`build_demo() -> gr.Blocks`** — factory that returns a `gr.Blocks` with a
  placeholder (a title + a "Placeholder — replace me" markdown block). A factory
  rather than a module-level `demo` value so that importing the package has no
  side effects (consistent with how `openrouter/chat.py` builds its client
  lazily) and so the demo is easy to construct in a test.
- **`launch(**kwargs) -> None`** — calls `load_dotenv()` (so API keys are
  available when the real UI is built, matching `main.py`), builds the demo via
  `build_demo()`, and calls `.launch(**kwargs)`.
- **`__main__.py`** — calls `launch()`, so `uv run python -m gradio_app` opens
  the browser. The existing `main.py` (the adversarial conversation runner) is
  left untouched.

## Loose ends

- **mypy / type stubs:** Gradio ships no type stubs, so `import gradio` will
  trip `mypy`. Add a minimal `[tool.mypy]` override
  (`ignore_missing_imports` for the `gradio` module) in `pyproject.toml` and
  confirm `ruff format`, `ruff check`, and `mypy` all pass on the new package.
- **README:** add a short "Web UI" section documenting the run command.

## Verification

Run `uv run python -m gradio_app`, confirm the placeholder page serves on the
local Gradio URL, then shut it down.

## Out of scope

- Any real UI (chat box, model comparison, conversation viewer). Those come
  later, built into this scaffold.
- Changes to `main.py` or the existing `openrouter/` / `claude/` packages.