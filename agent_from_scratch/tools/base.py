import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from agent_from_scratch.helpers import function_to_input_schema, format_tool_definition
from agent_from_scratch.models import ExecutionContext


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        tool_definition: dict[str, Any] | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
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
            description=description or (func.__doc__ or "").strip(),
        )
        # Needs self.name and self.description, so it runs after super().__init__.
        self._tool_definition = tool_definition or self._generate_definition()

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)
