"""The MCP server the tools are offered through.

Each tool registers itself here with ``@tool`` when its module is imported, and
the package imports every tool module, so however the server is reached it has
all of them.

Run it as ``python -m agent_from_scratch.tools``: it speaks MCP over its stdin
and stdout, which is how a client launches a local server and talks to it.
"""

from mcp.server.mcpserver import MCPServer

server = MCPServer("agent-tools")

# The tools answer in prose for the model to read, and structured output would
# only send the same string a second time, as ``{"result": ...}``.
tool = server.tool(structured_output=False)
