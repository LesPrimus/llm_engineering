"""Tools the agent can call, offered to it over MCP.

A tool has two readers. The model sees a JSON schema — a name, a description,
the arguments and their types — and answers with a call to it. The server sees
a Python function to run on the arguments. :mod:`~agent_from_scratch.tools.server`
declares each tool once, as a function the schema is derived from, and what the
model should know about a single argument goes in
``Annotated[..., Field(description=...)]``. The work itself is done in a module
per tool, plain Python that can be called without a server.

What a tool keeps between calls — a client, say — sits behind a cached function
in its module, built on the first call so that importing the tool needs no key.

A tool that fails does not bring the server down: the call comes back as an
error result. Only a ``ToolError`` carries its message to the model, so the
server raises one for a failure the model can act on — arguments it can fix, say.
Anything else is taken for a bug, and the model reads just that the tool failed.
"""

# Nothing is imported here: ``-m`` on the server imports this package first, and
# a package that imported the server would have it built twice, once under its
# name and once more as ``__main__``.
