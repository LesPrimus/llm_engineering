"""The MCP server the tools are offered through.

Run it as ``python -m agent_from_scratch.tools``: it speaks MCP over its stdin
and stdout, which is how a client launches a local server and talks to it.
"""

import inspect

from mcp.server.mcpserver import MCPServer

from .calculator import calculator
from .web_search import WebSearch

server = MCPServer("agent-tools")

for tool in (calculator, WebSearch()):
    server.add_tool(
        tool,
        # MCPServer would send ``__doc__`` as written, indentation and all;
        # getdoc dedents it into the text the model reads.
        description=inspect.getdoc(tool),
        # The tools answer in prose for the model to read, and structured output
        # would only send the same string a second time, as ``{"result": ...}``.
        structured_output=False,
    )
