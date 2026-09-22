"""Web search, for the facts the model was not trained on or does not recall."""

import os
from functools import cache
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from pydantic import Field
from tavily import TavilyClient  # type: ignore[import-untyped]

from .server import server

# Enough results to cross-check a claim against a second source, few enough
# that a handful of searches still leaves room in the context for the reasoning.
MAX_RESULTS = 5

# "advanced" reads further into each page before picking the snippet, which
# is worth the extra latency: a snippet that already contains the fact saves
# the agent a page fetch it would otherwise have to make.
SEARCH_DEPTH = "advanced"


@server.tool(structured_output=False)
def web_search(
    query: Annotated[
        str,
        Field(
            description="What to look up, phrased as a search engine query, "
            'e.g. "GAIA benchmark validation split size".'
        ),
    ],
    topic: Annotated[
        Literal["general", "news", "finance"],
        Field(
            description='Which index to search: "news" for current events, '
            '"finance" for markets and tickers, "general" for everything else.'
        ),
    ] = "general",
    time_range: Annotated[
        Literal["day", "week", "month", "year"] | None,
        Field(
            description="Discard results older than this. Set it only when the "
            "question is about something recent; it hides older pages that may "
            "hold the answer."
        ),
    ] = None,
) -> str:
    """Search the web and return the top results: title, URL and a snippet of each.

    Use it for anything you do not know, are unsure of, or that may have changed
    since you were trained. Search one fact at a time — a narrow query returns
    better snippets than a broad one — and search again to check what you found.
    For something that happened recently, search the news topic and bound how
    far back the results may come from.
    The snippets are extracts, not whole pages, so treat them as a way to decide
    which sources are worth trusting rather than as the full story.
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
