import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, overload

from agent_from_scratch.helpers import (
    function_to_description,
    function_to_input_schema,
    format_tool_definition,
)
from agent_from_scratch.execution_context import ExecutionContext


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
    ):
        self.name = name or type(self).__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext | None = None, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext | None = None, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters

        super().__init__(
            name=name or func.__name__,
            description=description or function_to_description(func),
        )
        # Needs self.name and self.description, so it runs after super().__init__.
        self._tool_definition = tool_definition or self._generate_definition()

    async def execute(self, context: ExecutionContext | None = None, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            if context is None:
                raise ValueError(f"Tool '{self.name}' requires an ExecutionContext")
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> dict[str, Any]:
        """Generate tool definition from function signature."""
        # execute injects the context, so the model is never asked for it.
        exclude = {"context"} if self.needs_context else ()
        parameters = function_to_input_schema(self.func, exclude=exclude)
        return format_tool_definition(self.name, self.description, parameters)


@overload
def tool(func: Callable, /) -> FunctionTool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    tool_definition: dict[str, Any] | None = None,
) -> Callable[[Callable], FunctionTool]: ...


def tool(
    func: Callable | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    tool_definition: dict[str, Any] | None = None,
) -> FunctionTool | Callable[[Callable], FunctionTool]:
    """Turn a function into a FunctionTool, as ``@tool`` or ``@tool(name=...)``."""

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            f, name=name, description=description, tool_definition=tool_definition
        )

    # Bare @tool hands the function straight in; @tool(...) returns the decorator.
    if func is None:
        return decorator
    return decorator(func)
