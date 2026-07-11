# gradio_app Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install Gradio and add a `gradio_app/` package with a runnable placeholder UI, peer to the existing `openrouter/` and `claude/` packages.

**Architecture:** A new `gradio_app/` package exposes a `build_demo()` factory (returns a `gr.Blocks` placeholder) and a `launch()` function (loads env, builds the demo, serves it). A `__main__.py` makes `uv run python -m gradio_app` open the browser. Nothing else in the repo changes except `pyproject.toml` (dependency + mypy config) and `README.md` (run docs).

**Tech Stack:** Python ≥ 3.14, `uv` for dependency management, `gradio`, `python-dotenv`, `ruff` + `mypy` for lint/type checks.

## Global Constraints

- Python `>=3.14`; dependencies managed with `uv` (`uv add`, `uv run`) — never edit `uv.lock` by hand.
- Package name is exactly `gradio_app` (a bare `gradio/` would shadow the installed library).
- No module-level `demo` / no import-time side effects — the demo is built via a `build_demo()` factory (mirrors the lazy client in `openrouter/chat.py`).
- Do not modify `main.py`, `openrouter/`, or `claude/`.
- This project has no test framework; verify with import/run smoke checks, not pytest.
- Commits carry no Claude co-author trailer.
- Code must pass `uv run ruff format .`, `uv run ruff check .`, and `uv run mypy`.

---

### Task 1: Install Gradio and configure mypy

**Files:**
- Modify: `pyproject.toml` (adds `gradio` to `[project.dependencies]` via `uv add`; add a `[tool.mypy]` override)
- Modify: `uv.lock` (updated automatically by `uv add`)

**Interfaces:**
- Consumes: nothing.
- Produces: an installed `gradio` importable as `import gradio as gr`; a mypy config that ignores gradio's missing type stubs.

- [ ] **Step 1: Add the dependency**

Run:
```bash
uv add gradio
```
Expected: `pyproject.toml` gains `"gradio>=..."` under `[project.dependencies]`, `uv.lock` updates, and the resolve/install succeeds.

- [ ] **Step 2: Verify gradio imports**

Run:
```bash
uv run python -c "import gradio as gr; print(gr.__version__)"
```
Expected: prints a version string (e.g. `5.x.x`) with no traceback.

- [ ] **Step 3: Confirm mypy currently flags gradio's missing stubs**

Run:
```bash
uv run python -c "import pathlib; pathlib.Path('_stub_probe.py').write_text('import gradio as gr\n')"
uv run mypy _stub_probe.py
```
Expected: mypy reports a missing-stubs / `import-untyped` error for `gradio` (this is the problem we fix next). If mypy instead passes cleanly, gradio ships stubs — skip Step 4 and just delete the probe.

- [ ] **Step 4: Add the mypy override to `pyproject.toml`**

Append to `pyproject.toml`:
```toml
[[tool.mypy.overrides]]
module = ["gradio.*", "gradio"]
ignore_missing_imports = true
```

- [ ] **Step 5: Verify the override silences the error, then clean up the probe**

Run:
```bash
uv run mypy _stub_probe.py
rm _stub_probe.py
```
Expected: `Success: no issues found` before removing the probe file.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add gradio dependency and mypy stub override"
```

---

### Task 2: Create the `gradio_app` package with a runnable placeholder

**Files:**
- Create: `gradio_app/__init__.py`
- Create: `gradio_app/app.py`
- Create: `gradio_app/__main__.py`
- Modify: `README.md` (add a "Web UI" section)

**Interfaces:**
- Consumes: `gradio` (Task 1); `dotenv.load_dotenv` (already a project dependency).
- Produces:
  - `gradio_app.build_demo() -> gr.Blocks`
  - `gradio_app.launch(**kwargs: object) -> None`
  - `python -m gradio_app` entry point that serves the placeholder.

- [ ] **Step 1: Write `gradio_app/app.py`**

```python
"""A placeholder Gradio app for the llm-engineering project.

This is a scaffold: ``build_demo`` returns an empty-ish ``gr.Blocks`` UI meant
to be replaced with a real interface later. Keeping the demo behind a factory
(rather than a module-level value) means importing this package has no side
effects, mirroring the lazy client in ``openrouter/chat.py``.
"""

from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv


def build_demo() -> gr.Blocks:
    """Build and return the Gradio UI. Replace the placeholder with real UI."""
    with gr.Blocks(title="llm-engineering") as demo:
        gr.Markdown(
            "# llm-engineering\n"
            "Placeholder — replace me with a real UI."
        )
    return demo


def launch(**kwargs: object) -> None:
    """Load environment variables, build the demo, and serve it.

    ``load_dotenv`` runs here (not at import) so API keys are available once the
    real UI is built, matching how ``main.py`` bootstraps the project.
    """
    load_dotenv()
    build_demo().launch(**kwargs)
```

- [ ] **Step 2: Write `gradio_app/__init__.py`**

```python
from .app import build_demo, launch

__all__ = ["build_demo", "launch"]
```

- [ ] **Step 3: Write `gradio_app/__main__.py`**

```python
from gradio_app import launch

if __name__ == "__main__":
    launch()
```

- [ ] **Step 4: Smoke-test the factory (no server)**

Run:
```bash
uv run python -c "import gradio as gr; from gradio_app import build_demo; d = build_demo(); assert isinstance(d, gr.Blocks); print('build_demo OK')"
```
Expected: prints `build_demo OK` with no traceback.

- [ ] **Step 5: Lint and type-check the new package**

Run:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy gradio_app
```
Expected: ruff reports files formatted/unchanged and no lint errors; mypy prints `Success: no issues found`.

- [ ] **Step 6: Verify the server actually serves the placeholder**

Start it in the background, check it responds, then stop it:
```bash
uv run python -m gradio_app > /tmp/gradio_app.log 2>&1 &
GRADIO_PID=$!
sleep 8
curl -sSf -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:7860/
kill $GRADIO_PID
```
Expected: `HTTP 200`. (If port 7860 is busy, check `/tmp/gradio_app.log` for the actual URL Gradio printed and curl that instead.)

- [ ] **Step 7: Add a "Web UI" section to `README.md`**

Insert after the existing `## Usage` section:
```markdown
## Web UI

A placeholder Gradio app lives in the `gradio_app` package. Launch it with:

```bash
uv run python -m gradio_app
```

This opens a local page in your browser. It's currently a scaffold — replace the
placeholder in `gradio_app/app.py` with a real interface.
```

- [ ] **Step 8: Commit**

```bash
git add gradio_app README.md
git commit -m "Add gradio_app package with runnable placeholder UI"
```

---

## Self-Review

**Spec coverage:**
- Dependency (`uv add gradio`) → Task 1. ✓
- Package layout `__init__.py` / `app.py` / `__main__.py` → Task 2, Steps 1–3. ✓
- `build_demo()` factory, no import side effects → Task 2, Step 1. ✓
- `launch(**kwargs)` with `load_dotenv()` → Task 2, Step 1. ✓
- `python -m gradio_app` runnable → Task 2, Steps 3 & 6. ✓
- mypy override for gradio stubs → Task 1, Steps 3–5. ✓
- README "Web UI" section → Task 2, Step 7. ✓
- Verification (page serves) → Task 2, Step 6. ✓
- Out of scope (`main.py`, existing packages untouched) → honored; no task modifies them. ✓

**Placeholder scan:** No "TBD"/"TODO"/"handle edge cases" left in plan steps. The word "Placeholder" appears only as intentional UI copy. ✓

**Type consistency:** `build_demo() -> gr.Blocks` and `launch(**kwargs: object) -> None` are named identically in the Interfaces block, `app.py`, `__init__.py`, and the smoke tests. ✓