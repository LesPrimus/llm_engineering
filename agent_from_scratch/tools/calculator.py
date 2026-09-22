"""Arithmetic the model can hand off rather than work out in its head."""

import ast
import operator
from collections.abc import Callable
from typing import Any

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


def calculate(expression: str) -> str:
    """The exact value of an arithmetic expression, written out.

    Raises ``SyntaxError`` when the expression does not parse, ``ValueError``
    when it is not arithmetic or is too large to compute or to write out, and
    ``ArithmeticError`` for a division by zero or a float that overflows.
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
