"""The :class:`Tool` wrapper: one function, and the schema derived from it.

Writing the schema by hand leaves two copies of one signature to keep in step,
so :class:`Tool` derives it instead: the name from the function, the description
from its docstring, the arguments from its type hints.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from pydantic import TypeAdapter


@dataclass(frozen=True)
class Tool:
    """A function the model can call, and the schema it is offered under."""

    function: Callable[..., str]

    def __post_init__(self) -> None:
        if not inspect.getdoc(self.function):
            raise ValueError(f"{self.name} needs a docstring: it is the description")

    @property
    def name(self) -> str:
        """What the model calls the tool by, and what its calls come back naming."""
        return self.function.__name__

    @property
    def schema(self) -> dict[str, Any]:
        """The tool's entry in a chat completion's ``tools`` list."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": inspect.getdoc(self.function),
                "parameters": self._adapter.json_schema(),
            },
        }

    def call(self, arguments: str) -> str:
        """Run the function on the model's JSON arguments, checked against its signature.

        Raises ``pydantic.ValidationError`` when the arguments do not fit the
        signature, and whatever the function raises when they do and it fails
        anyway. Telling the model what went wrong is the agent loop's job.
        """
        return self._adapter.validate_json(arguments)

    @cached_property
    def _adapter(self) -> TypeAdapter[str]:
        """Validates arguments against the signature, and calls the function with them."""
        return TypeAdapter(self.function)
