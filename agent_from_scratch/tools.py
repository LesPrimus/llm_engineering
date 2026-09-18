"""Tools the agent can call: plain functions, and the schema a model knows them by.

A tool has two readers. The model sees a JSON schema — a name, a description,
the arguments and their types — and answers with a call to it, the arguments
as a JSON string. The agent loop sees a Python function to run on them. Writing
the schema by hand leaves two copies of one signature to keep in step, so
:class:`Tool` derives it instead: the name from the function, the description
from its docstring, the arguments from its type hints. What the model should
know about a single argument goes in ``Annotated[..., Field(description=...)]``.

A tool's docstring is sent to the model verbatim, so it is written for the model
to read; notes for us go in comments.
"""

import ast
import inspect
import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any

from pydantic import Field, TypeAdapter


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


# ``9 ** 9 ** 9`` is eleven characters and a computation that never finishes.
# Integer powers are refused past this many bits, about 3,000 digits; a float
# power overflows on its own, with an error rather than a hang.
MAX_POWER_BITS = 10_000

BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculator(
    expression: Annotated[
        str,
        Field(
            description="Numbers, + - * / // % ** and parentheses, "
            'e.g. "2 * (3 + 4) ** 2".'
        ),
    ],
) -> str:
    """Evaluate an arithmetic expression exactly and return the result.

    Use it for any arithmetic rather than working the numbers out yourself.
    """
    # Walked by hand rather than handed to ``eval``: the expression is written
    # by a model, and a model can write ``__import__("os")`` as easily as ``1 + 1``.
    return str(_evaluate(ast.parse(expression, mode="eval").body))


def _evaluate(node: ast.expr) -> int | float | complex:
    """The value of one node, which has to be a number or arithmetic on numbers."""
    match node:
        case ast.Constant(value=int() | float() as value) if not isinstance(
            value, bool
        ):
            return value
        case ast.BinOp(left, op, right) if type(op) in BINARY_OPERATORS:
            left_value, right_value = _evaluate(left), _evaluate(right)
            if isinstance(op, ast.Pow):
                _check_power(left_value, right_value)
            return BINARY_OPERATORS[type(op)](left_value, right_value)
        case ast.UnaryOp(op, operand) if type(op) in UNARY_OPERATORS:
            return UNARY_OPERATORS[type(op)](_evaluate(operand))
    raise ValueError(f"not arithmetic: {ast.unparse(node)}")


def _check_power(base: object, exponent: object) -> None:
    """Refuse an integer power whose result would pass :data:`MAX_POWER_BITS`."""
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and base.bit_length() * exponent > MAX_POWER_BITS
    ):
        raise ValueError(f"{base} ** {exponent} is too large to compute")


CALCULATOR = Tool(calculator)
