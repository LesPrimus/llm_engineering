import asyncio
from dataclasses import dataclass

from pydantic import BaseModel

from agent_from_scratch.client import LlmClient
from agent_from_scratch.execution_context import ExecutionContext
from agent_from_scratch.models import LlmRequest, Message, ToolCall, ToolResult
from agent_from_scratch.tools.base import BaseTool


@dataclass(frozen=True)
class Agent:
    model: LlmClient
    tools: list[BaseTool] | None = None
    instructions: str = ""
    max_steps: int = 10
    name: str = "agent"

    async def run(
        self, user_input: str, context: ExecutionContext | None = None
    ) -> str | BaseModel | None:
        if context is None:
            context = ExecutionContext()
        context.add_event(Message(author="user", role="user", content=user_input))

        while context.final_result is None and context.current_step < self.max_steps:
            await self.step(context)
        return context.final_result

    async def step(self, context: ExecutionContext) -> None:
        request = LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=context.events,
            tools=self.tools or [],
        )
        response = await self.model.generate(request)
        events = [
            event.model_copy(update={"author": self.name}) for event in response.content
        ]
        for event in events:
            context.add_event(event)

        tool_calls = [event for event in events if isinstance(event, ToolCall)]
        if tool_calls:
            results = await asyncio.gather(
                *(self._run_tool(call, context) for call in tool_calls)
            )
            for result in results:
                context.add_event(result)
        else:
            # No tool calls means the model has answered: its last message is the result.
            context.final_result = next(
                (
                    event.content
                    for event in reversed(events)
                    if isinstance(event, Message)
                ),
                None,
            )
        context.increment_step()

    async def _run_tool(self, call: ToolCall, context: ExecutionContext) -> ToolResult:
        tools = {tool.name: tool for tool in self.tools or []}
        status, output = "success", None
        if call.name not in tools:
            status, output = "error", f"Unknown tool '{call.name}'"
        else:
            try:
                output = await tools[call.name](context, **call.arguments)
            except Exception as e:
                # Hand the error back to the model so it can retry or change course.
                status, output = "error", str(e)
        return ToolResult(
            author=self.name,
            tool_call_id=call.tool_call_id,
            name=call.name,
            status=status,
            content=[output],
        )


async def main() -> None:
    client = LlmClient()
    agent = Agent(model=client, instructions="Hey yooo")
    print(await agent.run(user_input="hello"))


if __name__ == "__main__":
    asyncio.run(main())
