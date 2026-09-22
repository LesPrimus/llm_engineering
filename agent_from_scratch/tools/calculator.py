"""Arithmetic the model can hand off rather than work out in its head."""

import ast
import operator
from collections.abc import Callable
from typing import Annotated, Any

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

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
    try:
        return str(_evaluate(ast.parse(expression, mode="eval").body))
    except (SyntaxError, ArithmeticError, ValueError) as error:
        # Each of these is the expression's fault — a typo, a division by zero,
        # a number too big to write out — so the model is told which, to fix it.
        # MCP keeps the text of any other exception from the model.
        raise ToolError(str(error)) from error


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
