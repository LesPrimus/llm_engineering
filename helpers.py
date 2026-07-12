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
