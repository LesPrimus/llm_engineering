# Company Website Summarizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gradio app that fetches a company's landing page and streams back a live-rendering Markdown summary, with a model dropdown (GPT / Claude / DeepSeek via OpenRouter).

**Architecture:** Two concerns in two files. `helpers.py` (project root) fetches and cleans a single web page with httpx + BeautifulSoup into a frozen `Website` dataclass — no LLM knowledge. `gradio_app/website_summarizer.py` mirrors the existing `multibot.py` shape (local `Model` enum, self-contained `Summarizer` dataclass owning its OpenRouter client) and wires a `gr.Interface` whose callback fetches then summarizes.

**Tech Stack:** Python 3.14, uv, Gradio, httpx (already present transitively), BeautifulSoup (`beautifulsoup4`, new), OpenAI SDK pointed at OpenRouter, python-dotenv.

## Global Constraints

- Python >= 3.14; manage everything through `uv` (`uv run …`, `uv add …`).
- No test framework in this repo — verification is **import/run smoke checks** (`uv run python -c "…"`), plus `ruff` and `mypy`, matching the multibot scaffold convention.
- Follow the established `multibot.py` shape: `Model` **Enum** (member name = UI label, value = OpenRouter model ID); a **frozen dataclass that owns its client** via `field(default_factory=_client)` where `_client` is a `@staticmethod` calling `load_dotenv()` itself. Importing a module must have **no side effects** (no network, no client built at import).
- Main page only — **never follow links / crawl**.
- Keep personal info out of committed files (public repo). Git commits use the repo-local identity; **no Claude co-author trailer**.
- OpenRouter model IDs must match those already in `multibot.py`: `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4.5`, `deepseek/deepseek-chat`.

---

### Task 1: `helpers.py` — the `Website` fetcher

**Files:**
- Create: `helpers.py` (project root, next to `main.py`)
- Modify: `pyproject.toml` (add `beautifulsoup4` dependency — done via `uv add`)

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `class Website` — a frozen dataclass with fields `url: str`, `title: str`, `text: str`, and a classmethod `Website.fetch(url: str) -> Website` that performs a live HTTP GET and returns the parsed page. Raises `httpx.HTTPError` (incl. `httpx.HTTPStatusError` on non-2xx) on failure.

- [ ] **Step 1: Add the BeautifulSoup dependency**

Run:
```bash
uv add beautifulsoup4
```
Expected: `pyproject.toml` gains `beautifulsoup4` under `[project].dependencies`, `uv.lock` updates, and the package installs into `.venv`.

- [ ] **Step 2: Write the smoke check and run it to confirm it fails**

Run:
```bash
uv run python -c "from helpers import Website; w = Website.fetch('https://example.com'); assert 'Example Domain' in w.title, w.title; assert w.text, 'empty text'; print('OK:', repr(w.title))"
```
Expected: **FAIL** with `ModuleNotFoundError: No module named 'helpers'` (file not created yet).

- [ ] **Step 3: Create `helpers.py`**

```python
"""Fetch and clean a single web page for summarization.

``Website.fetch`` does one HTTP GET (no link crawling) and returns a frozen
``Website`` holding the page URL, title, and readable body text with markup and
boilerplate tags stripped. It has no LLM knowledge and no import-time side
effects — nothing happens until ``Website.fetch`` is called.
"""

from dataclasses import dataclass
from typing import ClassVar

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class Website:
    """A single fetched web page: its URL, title, and cleaned body text."""

    # A browser-like User-Agent; many sites reject the default httpx agent.
    _HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": "Mozilla/5.0 (compatible; llm-engineering/0.1)"
    }
    # Markup/boilerplate tags removed before extracting text.
    _STRIP_TAGS: ClassVar[list[str]] = ["script", "style", "img", "input"]

    url: str
    title: str
    text: str

    @classmethod
    def fetch(cls, url: str) -> "Website":
        """Fetch ``url`` (main page only) and return the cleaned ``Website``.

        Sends a browser-like User-Agent and follows redirects. Raises
        ``httpx.HTTPStatusError`` on a non-2xx response (and other
        ``httpx.HTTPError`` subclasses on connection failures) so callers can
        surface a fetch error without parsing junk.
        """
        response = httpx.get(url, headers=cls._HEADERS, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Bind to locals so mypy narrows the Optional cleanly (soup.title /
        # soup.body are dynamic attributes, not narrowable when re-accessed).
        title_tag = soup.title
        title = (title_tag.get_text(strip=True) if title_tag else "") or url
        for tag in soup(cls._STRIP_TAGS):
            tag.decompose()
        body = soup.body
        text = body.get_text("\n", strip=True) if body else ""
        return cls(url=url, title=title, text=text)
```

- [ ] **Step 4: Run the smoke check to verify it passes**

Run:
```bash
uv run python -c "from helpers import Website; w = Website.fetch('https://example.com'); assert 'Example Domain' in w.title, w.title; assert w.text, 'empty text'; print('OK:', repr(w.title))"
```
Expected: **PASS**, prints something like `OK: 'Example Domain'`.

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run ruff format helpers.py
uv run ruff check helpers.py
uv run mypy helpers.py
```
Expected: all pass. If mypy reports missing type stubs for `bs4`, add them with `uv add --dev types-beautifulsoup4` and re-run (recent `beautifulsoup4` ships inline types, so this is usually unnecessary — do not add the stubs preemptively, to avoid a duplicate-module clash).

- [ ] **Step 6: Commit**

```bash
git add helpers.py pyproject.toml uv.lock
git commit -m "Add Website fetcher (httpx + BeautifulSoup) in helpers.py"
```

---

### Task 2: `gradio_app/website_summarizer.py` — the summarizer app

**Files:**
- Create: `gradio_app/website_summarizer.py`

**Interfaces:**
- Consumes: `from helpers import Website` (Task 1) — `Website.fetch(url) -> Website`, fields `url`/`title`/`text`.
- Produces: `class Model(Enum)` (GPT/Claude/DeepSeek); `class Summarizer` (frozen dataclass, fields `model: Model`, `client: OpenAI`, `system: str`; method `summarize(website: Website) -> Iterator[str]`); `build_demo() -> gr.Blocks`; `launch(**kwargs: Any) -> None`. Entry point: `python -m gradio_app.website_summarizer`.

- [ ] **Step 1: Write the smoke checks and run them to confirm they fail**

Run:
```bash
uv run python -c "from gradio_app.website_summarizer import Model; assert {m.name: m.value for m in Model} == {'GPT': 'openai/gpt-4o-mini', 'Claude': 'anthropic/claude-sonnet-4.5', 'DeepSeek': 'deepseek/deepseek-chat'}; print('enum OK')"
```
Expected: **FAIL** with `ModuleNotFoundError: No module named 'gradio_app.website_summarizer'`.

- [ ] **Step 2: Create `gradio_app/website_summarizer.py`**

```python
"""A Gradio app that summarizes a company's website in Markdown.

Paste a company URL and pick a model — GPT, Claude, or DeepSeek — from a
dropdown. The page's readable text is fetched by ``helpers.Website`` (main page
only, no crawling) and a short Markdown summary streams back into a live
``gr.Markdown`` output. Every model is reached through an OpenAI client pointed
at OpenRouter, so switching models is just a different model ID.

This mirrors ``gradio_app/multibot.py``: a ``Model`` enum lists the choices
(member name is the UI label, value is the OpenRouter model ID) and a
``Summarizer`` dataclass pairs a chosen ``Model`` with its own OpenRouter client
and a summarization system prompt. Clients are built inside ``build_demo`` (not
at import), so importing this module has no side effects.

Run it with ``uv run python -m gradio_app.website_summarizer``.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import gradio as gr
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam as Message,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
)

from helpers import Website

SYSTEM = (
    "You analyze the landing page of a company website and write a short "
    "summary in Markdown. Ignore navigation, cookie banners, and other "
    "boilerplate. Cover what the company does, its products or services, and "
    "any notable news if present. Respond in Markdown."
)


class Model(Enum):
    """Selectable models. Member name is the UI label; value is the OpenRouter ID."""

    GPT = "openai/gpt-4o-mini"
    Claude = "anthropic/claude-sonnet-4.5"
    DeepSeek = "deepseek/deepseek-chat"


@dataclass(frozen=True)
class Summarizer:
    """One model, reached through its own OpenRouter client, that summarizes a page."""

    @staticmethod
    def _client() -> OpenAI:
        # Load .env here so OPENROUTER_API_KEY is present. The OpenAI SDK's
        # default env var is OPENAI_API_KEY, so point the client at OpenRouter
        # and resolve the key explicitly rather than relying on the SDK fallback.
        load_dotenv()
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    model: Model
    client: OpenAI = field(default_factory=_client)
    system: str = SYSTEM

    def summarize(self, website: Website) -> Iterator[str]:
        """Stream a Markdown summary of ``website``.

        Yields the summary accumulated so far on each text chunk, so Gradio can
        render the Markdown output as the response arrives.
        """
        user = (
            f"Company website: {website.url}\n"
            f"Page title: {website.title}\n\n"
            f"Page contents:\n{website.text}\n\n"
            "Summarize this company in Markdown."
        )
        messages: list[Message] = [
            SystemMessage(role="system", content=self.system),
            UserMessage(role="user", content=user),
        ]
        stream = self.client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                reply += delta
                yield reply


def build_demo() -> gr.Blocks:
    """Build the UI: pick a model, enter a company URL, stream a Markdown summary."""
    summarizers = {model.name: Summarizer(model) for model in Model}

    def respond(name: str, url: str) -> Iterator[str]:
        url = url.strip()
        if not url:
            yield "Enter a company website URL to summarize."
            return
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        try:
            website = Website.fetch(url)
        except httpx.HTTPError as error:
            yield f"**Couldn't fetch {url}:** {error}"
            return
        yield from summarizers[name].summarize(website)

    return gr.Interface(
        fn=respond,
        inputs=[
            gr.Dropdown(
                choices=[model.name for model in Model],
                value=Model.Claude.name,
                label="Model",
            ),
            gr.Textbox(label="Company website URL"),
        ],
        outputs=gr.Markdown(label="Summary"),
        title="llm-engineering — website summarizer",
        flagging_mode="never",
    )


def launch(**kwargs: Any) -> None:
    """Build the demo and serve it.

    Each summarizer's client factory calls ``load_dotenv`` as it builds, so
    ``OPENROUTER_API_KEY`` is read from ``.env`` without ``launch`` having to
    manage the environment.
    """
    build_demo().launch(**kwargs)


if __name__ == "__main__":
    launch()
```

- [ ] **Step 3: Run the enum smoke check to verify it passes**

Run:
```bash
uv run python -c "from gradio_app.website_summarizer import Model; assert {m.name: m.value for m in Model} == {'GPT': 'openai/gpt-4o-mini', 'Claude': 'anthropic/claude-sonnet-4.5', 'DeepSeek': 'deepseek/deepseek-chat'}; print('enum OK')"
```
Expected: **PASS**, prints `enum OK`.

- [ ] **Step 4: Verify `build_demo()` constructs without a network call**

Run:
```bash
uv run python -c "import gradio as gr; from gradio_app.website_summarizer import build_demo; demo = build_demo(); assert isinstance(demo, gr.Blocks); print('build_demo OK')"
```
Expected: **PASS**, prints `build_demo OK`. (This builds each `Summarizer`'s OpenRouter client, which reads `OPENROUTER_API_KEY` from `.env` via `load_dotenv` but makes no network call — client construction is lazy. Requires `.env` present.)

- [ ] **Step 5: Lint and type-check**

Run:
```bash
uv run ruff format gradio_app/website_summarizer.py
uv run ruff check gradio_app/website_summarizer.py
uv run mypy gradio_app/website_summarizer.py
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add gradio_app/website_summarizer.py
git commit -m "Add company website summarizer Gradio app (Markdown, model selector)"
```

---

### Task 3: Document the new app in `README.md`

**Files:**
- Modify: `README.md` (add a subsection under "## Web UI", after the Multibot selector block ending at the `no Anthropic key needed` line)

**Interfaces:**
- Consumes: the entry point from Task 2 (`python -m gradio_app.website_summarizer`).
- Produces: nothing (docs only).

- [ ] **Step 1: Add the Website-summarizer subsection**

Insert the following immediately after the line `It uses `OPENROUTER_API_KEY` — no Anthropic key needed.` and before `## Development`:

```markdown

**Website summarizer.** Paste a company's website URL and pick GPT, Claude, or
DeepSeek; the landing page is fetched with BeautifulSoup and a short Markdown
summary streams back:

```bash
uv run python -m gradio_app.website_summarizer
```

It uses `OPENROUTER_API_KEY` too. Only the given page is fetched — links are not
followed.
```

- [ ] **Step 2: Verify the edit reads correctly**

Run:
```bash
uv run python -c "t = open('README.md').read(); assert 'Website summarizer' in t and 'gradio_app.website_summarizer' in t; print('README OK')"
```
Expected: **PASS**, prints `README OK`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document website summarizer app in README"
```

---

## Verification (whole feature)

After all tasks, confirm end to end:

1. `uv run ruff check .` and `uv run mypy helpers.py gradio_app/website_summarizer.py` pass.
2. Import smoke: `uv run python -c "import helpers, gradio_app.website_summarizer; print('imports OK')"`.
3. Live fetch: the Task 1 `example.com` smoke check passes.
4. **Manual (hits OpenRouter):** `uv run python -m gradio_app.website_summarizer`, enter a real company URL (e.g. `anthropic.com`), pick each model, and confirm the Markdown summary streams and renders. Run this once by hand.

## Notes / Out of scope

- No link crawling, no JS-rendered content (static HTML only), no multi-turn refinement, no per-model prompts — all deferred per the spec.
- `gradio_app/__init__.py`, `app.py`, and `multibot.py` are not modified (the summarizer is a standalone entry point, like `multibot.py`).