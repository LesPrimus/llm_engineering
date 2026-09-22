import asyncio
import sys

from mcp import StdioServerParameters, stdio_client, ClientSession

server_params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agent_from_scratch.tools.server"],
)


async def method_name():
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")


if __name__ == "__main__":
    asyncio.run(method_name())
