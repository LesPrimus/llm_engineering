from typing import TYPE_CHECKING, Any

from mcp import ClientSession, types

from agent_from_scratch.helpers import format_tool_definition
from agent_from_scratch.tools.base import BaseTool

if TYPE_CHECKING:
    # models imports the tools for LlmRequest: ExecutionContext is only needed
    # for annotations, so importing it at runtime would close the cycle.
    from agent_from_scratch.models import ExecutionContext


class MCPToolError(Exception):
    """A tool on an MCP server reported that its call failed."""


class MCPTool(BaseTool):
    """Wraps a tool on an MCP server, reached through an open session, as a BaseTool."""

    def __init__(self, session: ClientSession, mcp_tool: types.Tool):
        # The session belongs to the caller, who keeps it open while the tool is in use.
        self.session = session
        description = mcp_tool.description or ""

        super().__init__(
            name=mcp_tool.name,
            description=description,
            # The server already sends the schema, so it is used as is.
            tool_definition=format_tool_definition(
                mcp_tool.name, description, mcp_tool.input_schema
            ),
        )

    async def execute(self, context: ExecutionContext | None = None, **kwargs) -> Any:
        """Call the tool on the server and return the text it answers with."""
        # The context stays here: the server runs in another process.
        result = await self.session.call_tool(self.name, kwargs)
        if not isinstance(result, types.CallToolResult):
            raise MCPToolError(f"Tool '{self.name}' did not return a tool result")

        # Only text is kept; the tools answer in prose.
        text = "\n".join(
            block.text
            for block in result.content
            if isinstance(block, types.TextContent)
        )
        if result.is_error:
            raise MCPToolError(text)
        return text


async def load_mcp_tools(session: ClientSession) -> list[MCPTool]:
    """Wrap every tool the session's server offers."""
    result = await session.list_tools()
    return [MCPTool(session, mcp_tool) for mcp_tool in result.tools]
