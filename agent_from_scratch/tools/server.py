"""The MCP server the tools are offered through.

Each tool is declared here as the model sees it — its name, what it is for,
its arguments and what each one means — and hands the work to a module of its
own, which knows nothing of MCP. A tool's docstring is sent to the model
verbatim, so it is written for the model to read; notes for us go in comments.

The tools are registered with ``structured_output=False``: they answer in prose
for the model to read, and structured output would only send the same string a
second time, as ``{"result": ...}``.

Run it as ``python -m agent_from_scratch.tools``: it speaks MCP over its stdin
and stdout, which is how a client launches a local server and talks to it.
"""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from .calculator import calculate
from .web_search import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_LIMIT,
    TimeRange,
    Topic,
    search,
)

server = MCPServer("agent-tools")


@server.tool(structured_output=False)
def calculator(
    expression: Annotated[
        str,
        Field(
            description="Numbers, + - * / // % ** and parentheses, "
            'e.g. "2 * (3 + 4) ** 2".'
        ),
    ],
) -> str:
    """Evaluate an arithmetic expression exactly and return the result.

    Use it for any arithmetic rather than working the numbers out yourself.
    """
    try:
        return calculate(expression)
    except (SyntaxError, ArithmeticError, ValueError) as error:
        # Each of these is the expression's fault — a typo, a division by zero,
        # a number too big to write out — so the model is told which, to fix
        # it. Only a ToolError's message reaches the model; MCP withholds the
        # text of any other exception.
        raise ToolError(str(error)) from error


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
        Topic,
        Field(
            description='Which index to search: "news" for current events, '
            '"finance" for markets and tickers, "general" for everything else.'
        ),
    ] = "general",
    time_range: Annotated[
        TimeRange | None,
        Field(
            description="Discard results older than this. Set it only when the "
            "question is about something recent; it hides older pages that may "
            "hold the answer."
        ),
    ] = None,
    max_results: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_RESULTS_LIMIT,
            description="How many results to return. The default is enough to "
            "check a claim against a second source; ask for more only when those "
            "do not settle it, since every result takes room in your context.",
        ),
    ] = DEFAULT_MAX_RESULTS,
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
    return search(query, topic, time_range, max_results)
