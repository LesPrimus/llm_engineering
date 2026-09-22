"""Web search, for the facts the model was not trained on or does not recall."""

import os
from functools import cache
from typing import Any, Literal

from dotenv import load_dotenv
from tavily import TavilyClient  # type: ignore[import-untyped]

# Enough results to cross-check a claim against a second source, few enough
# that a handful of searches still leaves room in the context for the reasoning.
MAX_RESULTS = 5

# "advanced" reads further into each page before picking the snippet, which
# is worth the extra latency: a snippet that already contains the fact saves
# the agent a page fetch it would otherwise have to make.
SEARCH_DEPTH = "advanced"

# Plain aliases rather than ``type`` statements: pydantic moves a ``type``
# alias out into ``$defs`` in the schema, where the model has to follow a
# reference to find the values it may pass.
Topic = Literal["general", "news", "finance"]
TimeRange = Literal["day", "week", "month", "year"]


def search(
    query: str, topic: Topic = "general", time_range: TimeRange | None = None
) -> str:
    """The top results for a query, as the text the model reads.

    Each result is its title, its URL and a snippet, with its date when Tavily
    has one. ``time_range`` drops results older than that.
    """
    response = _client().search(
        query,
        topic=topic,
        time_range=time_range,
        search_depth=SEARCH_DEPTH,
        max_results=MAX_RESULTS,
    )
    if not (results := response.get("results", [])):
        return f"No results for {query!r}. Try different or broader search terms."
    return "\n\n".join(_format(result) for result in results)


def _format(result: dict[str, Any]) -> str:
    """One result as the lines the model reads: what, where, when, the extract."""
    content = " ".join((result.get("content") or "").split())
    # Only the news topic dates its results, and undated news is a story
    # without a year on it — worth the line whenever Tavily sends one.
    published = result.get("published_date")
    when = f" ({published})" if published else ""
    return (
        f"{result.get('title') or '(untitled)'}{when}\n"
        f"{result.get('url', '')}\n{content}"
    )


@cache
def _client() -> TavilyClient:
    """The Tavily client, built on the first search so import needs no key.

    Cached because the client holds a connection pool, and an agent loop
    searches more than once — and because the ``.env`` is then read once
    rather than per call.
    """
    # So a tool works from a notebook or a bare script, without every caller
    # having to remember the key this one needs. Already-set variables win,
    # so a key exported in the shell still overrides the file.
    load_dotenv()
    # Tavily falls back to a rate-limited keyless mode when the variable is
    # missing, which fails much later and much less clearly than this does.
    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is unset: web search needs a Tavily key")
    return TavilyClient()
