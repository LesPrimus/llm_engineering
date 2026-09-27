import asyncio
from dataclasses import dataclass
from functools import cached_property

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from agent_from_scratch.client import LlmClient
from agent_from_scratch.models import (
    ExecutionContext,
    LlmRequest,
    Message,
    ToolCall,
    ToolResult,
)
from agent_from_scratch.tools.base import BaseTool, FinalAnswerTool, tool
from agent_from_scratch.tools.calculator import calculate


@dataclass(frozen=True)
class Agent:
    model: LlmClient
    tools: list[BaseTool] | None = None
    instructions: str = ""
    max_steps: int = 10
    name: str = "agent"
    # The shape the final result must take, parsed into an instance of it.
    structured_output: type[BaseModel] | None = None

    @cached_property
    def _fallback_answer_tool(self) -> FinalAnswerTool | None:
        """The tool that carries the answer, where ``response_format`` cannot.

        None when there is nothing to fall back from: no ``structured_output``,
        or a provider that enforces it natively through ``response_format``.

        It is used even with no other tools: calling it is then the model's
        only move, which a provider that cannot take the format still allows.
        """
        if (
            self.structured_output is None
            or self.model.supports_native_structured_output()
        ):
            return None
        return FinalAnswerTool(self.structured_output)

    @cached_property
    def _tools(self) -> list[BaseTool]:
        """Every tool the model is offered, the answer tool among them."""
        answer_tool = [self._fallback_answer_tool] if self._fallback_answer_tool else []
        return [*(self.tools or []), *answer_tool]

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
            tools=self._tools,
            # A required tool call means the answer can only come as final_answer.
            tool_choice="required" if self._fallback_answer_tool else None,
            response_format=None
            if self._fallback_answer_tool
            else self.structured_output,
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
                if (
                    self._fallback_answer_tool
                    and result.name == self._fallback_answer_tool.name
                    and result.status == "success"
                ):
                    context.final_result = result.content[0]
        else:
            # No tool calls means the model has answered: its last message is the result.
            text = next(
                (
                    event.content
                    for event in reversed(events)
                    if isinstance(event, Message)
                ),
                None,
            )
            if self.structured_output is None or text is None:
                context.final_result = text
            else:
                try:
                    context.final_result = self.structured_output.model_validate_json(
                        text
                    )
                except ValidationError as e:
                    # Hand the mismatch back so the model can answer again.
                    context.add_event(
                        Message(
                            author="user",
                            role="user",
                            content=f"Your answer does not match the schema: {e}",
                        )
                    )
        context.increment_step()

    async def _run_tool(self, call: ToolCall, context: ExecutionContext) -> ToolResult:
        tools = {tool.name: tool for tool in self._tools}
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


class Calculation(BaseModel):
    expression: str
    value: int


async def main() -> None:
    load_dotenv()
    agent = Agent(
        model=LlmClient(),
        tools=[tool(calculate)],
        instructions="Use the calculate tool for any arithmetic.",
        structured_output=Calculation,
    )
    context = ExecutionContext()
    print("result:", await agent.run("What is 1234 * 5678 + 91?", context))
    for event in context.events:
        match event:
            case ToolCall(name=name, arguments=arguments):
                print(f"  {event.author}: call {name}({arguments})")
            case ToolResult(content=content, status=status):
                print(f"  {event.author}: {status} {content}")
            case Message(content=content):
                print(f"  {event.author}: {content}")
    print("steps:", context.current_step)


if __name__ == "__main__":
    asyncio.run(main())
