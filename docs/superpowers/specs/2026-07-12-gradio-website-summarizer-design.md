# Design: `gradio_app` company website summarizer

**Date:** 2026-07-12
**Status:** Approved

## Goal

Add a Gradio app that summarizes a company's website. The user pastes a company
URL and picks a model — **GPT**, **Claude**, or **DeepSeek** — from a dropdown
(same selection shape as `gradio_app/multibot.py`); the app fetches that page's
readable text and streams back a short **Markdown** summary that renders live.

Two concerns are kept separate:

- **Fetching** lives in a new project-root `helpers.py` — plain website fetching
  with BeautifulSoup, no LLM knowledge. **Main page only — no link crawling.**
- **Summarizing** lives in a new module, `gradio_app/website_summarizer.py`,
  following the established `multibot.py` shape (own `Model` enum, self-contained
  dataclass that owns its OpenRouter client).

New/changed files: `helpers.py` (new), `gradio_app/website_summarizer.py` (new),
`pyproject.toml` (add `beautifulsoup4` dep + `types-beautifulsoup4` dev dep), and
`README.md`. `gradio_app/__init__.py`, `app.py`, and `multibot.py` are untouched.

## Fetching: `helpers.py`

httpx is already available transitively (via `anthropic`/`openai`), so only
`beautifulsoup4` is a genuinely new dependency. A frozen `Website` dataclass
holds the fetched page, and a `fetch` classmethod is the factory:

```python
@dataclass(frozen=True)
class Website:
    url: str
    title: str
    text: str

    _HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; llm-engineering/0.1)"}

    @classmethod
    def fetch(cls, url: str) -> "Website":
        response = httpx.get(url, headers=cls._HEADERS, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        for tag in soup(["script", "style", "img", "input"]):
            tag.decompose()
        body = soup.body
        text = body.get_text("\n", strip=True) if body else ""
        return cls(url=url, title=title, text=text)
```

Rationale:

- A **browser-like `User-Agent`** is sent because many sites reject the default
  httpx agent.
- `follow_redirects=True` so bare `http`→`https` / `example.com`→`www.` hops
  resolve (httpx does not follow redirects by default).
- `raise_for_status()` turns a non-200 into an `httpx.HTTPStatusError`, which the
  summarizer's callback catches and renders (see Error handling).
- `script`, `style`, `img`, `input` are stripped so the summary sees prose, not
  boilerplate/markup. Only the fetched page is parsed — links are never followed.
- Importing `helpers` has no side effects; no network call happens until
  `Website.fetch` is called.

## Summarizing: `gradio_app/website_summarizer.py`

Mirrors `multibot.py`. A **local** `Model` enum keeps the module fully
independent (member **name** = dropdown label, **value** = OpenRouter model ID):

```python
class Model(Enum):
    GPT = "openai/gpt-4o-mini"
    Claude = "anthropic/claude-sonnet-4.5"
    DeepSeek = "deepseek/deepseek-chat"
```

A frozen `Summarizer` dataclass owns its OpenRouter client via a
`default_factory` `@staticmethod` that calls `load_dotenv()` itself — same
self-contained pattern as `multibot.Bot`:

```python
SYSTEM = (
    "You analyze the landing page of a company website and write a short "
    "summary in Markdown. Ignore navigation, cookie banners, and boilerplate. "
    "Cover what the company does, its products or services, and any notable "
    "news if present. Respond in Markdown."
)

@dataclass(frozen=True)
class Summarizer:
    @staticmethod
    def _client() -> OpenAI:
        load_dotenv()  # so OPENROUTER_API_KEY is present
        return OpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"])

    model: Model
    client: OpenAI = field(default_factory=_client)
    system: str = SYSTEM

    def summarize(self, website: Website) -> Iterator[str]:
        user = (
            f"Company website: {website.url}\n"
            f"Page title: {website.title}\n\n"
            f"Page contents:\n{website.text}\n\n"
            "Summarize this company in Markdown."
        )
        stream = self.client.chat.completions.create(
            model=self.model.value, stream=True,
            messages=[SystemMessage(role="system", content=self.system),
                      UserMessage(role="user", content=user)],
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                reply += delta
                yield reply
```

`summarize` takes an already-fetched `Website` (not a URL), so the LLM concern
stays separate from the fetch concern and each is testable alone. It yields the
reply accumulated so far on each chunk — the same streaming contract the other
apps use, which drives the live-rendering `gr.Markdown` output.

## UI

`build_demo() -> gr.Blocks` constructs one `Summarizer` per `Model` member (each
builds its own client via the `default_factory`), keyed by label, then wires a
`gr.Interface`:

- **inputs:** `gr.Dropdown(choices=[m.name for m in Model], value=Model.Claude.name, label="Model")`
  and `gr.Textbox(label="Company website URL")`.
- **output:** `gr.Markdown(label="Summary")` — renders headings/lists/bold live
  as the summary streams.
- **fn:** the `respond(name, url)` generator orchestrates fetch → summarize
  (see Error handling for the guards); on success it does
  `yield from summarizers[name].summarize(website)`.
- `flagging_mode="never"` is kept.

Default selection is **Claude**. Summarizers are built inside `build_demo` (not
at import), so each client — and its `load_dotenv` call — runs lazily when the UI
is built; importing the module has no side effects.

`launch(**kwargs)` calls `build_demo().launch(**kwargs)`, with an
`if __name__ == "__main__"` entry so `uv run python -m gradio_app.website_summarizer`
serves it.

## Error handling

The `respond(name, url)` callback guards the two failure points and renders
friendly Markdown in the output instead of throwing a raw Gradio error:

- **Blank URL** → yield `"Enter a company website URL to summarize."` and return.
- **Bare domain** (no `http`/`https` scheme) → prepend `https://` before
  fetching.
- **Fetch failure** (`httpx.HTTPError`, e.g. connection error or non-200) → catch
  and yield `` f"**Couldn't fetch {url}:** {error}"``, then return — the LLM is
  never called.

OpenRouter/OpenAI SDK errors during `summarize` propagate and Gradio surfaces
them, as in `multibot.py` (no extra layer there).

## Verification

Consistent with the repo convention (no test framework — verify with import/run
smoke checks):

1. `import helpers` and `import gradio_app.website_summarizer` succeed with no
   side effects.
2. `Website.fetch("https://example.com")` returns a `Website` whose `.title`
   contains "Example Domain" and whose `.text` is non-empty — confirms live fetch
   + parse without any LLM call.
3. `{m.name: m.value for m in Model}` maps each label to the expected OpenRouter
   model ID; `build_demo()` returns a `gr.Blocks`.
4. `ruff format`, `ruff check`, and `mypy` all pass.
5. Launch `uv run python -m gradio_app.website_summarizer`, enter a real company
   URL, and confirm the Markdown summary streams and renders (this step hits
   OpenRouter; run once by hand).

## Out of scope

- Crawling or following links / summarizing multiple pages (main page only).
- JavaScript-rendered content (httpx fetches static HTML, no headless browser).
- Multi-turn conversation or refinement of the summary.
- Per-model distinct prompts (all share one summarization system prompt).
- Changes to `gradio_app/app.py`, `multibot.py`, `__init__.py`, `main.py`,
  `openrouter/`, or `claude/`.
- Adding a test framework (pytest) — verification stays smoke-check based.