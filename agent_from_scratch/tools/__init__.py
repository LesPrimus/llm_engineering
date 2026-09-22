"""Tools the agent can call, offered to it over MCP.

A tool has two readers. The model sees a JSON schema — a name, a description,
the arguments and their types — and answers with a call to it. The server sees
a Python function to run on the arguments. :data:`server` derives the one from
the other, so a tool is written as a function and nothing else. What the model
should know about a single argument goes in ``Annotated[..., Field(description=...)]``.

A tool that carries something between calls — a client, a cache, settings worth
overriding — is written instead as a frozen dataclass with a ``__call__``, whose
fields hold that state and whose ``__name__`` is the name the model calls it by.
The server takes either.

A tool that fails does not bring the server down: the call comes back as an
error result. Only a ``ToolError`` carries its message to the model, so a tool
raises one for a failure the model can act on — arguments it can fix, say.
Anything else is taken for a bug, and the model reads just that the tool failed.

A tool's docstring is sent to the model verbatim, so it is written for the model
to read; notes for us go in comments. One module per tool, exported here, and
registered in :mod:`~agent_from_scratch.tools.server`.
"""

from .calculator import calculator
from .server import server
from .web_search import WebSearch

__all__ = [
    "WebSearch",
    "calculator",
    "server",
]
