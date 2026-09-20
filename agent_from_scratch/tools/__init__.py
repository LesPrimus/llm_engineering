"""Tools the agent can call: plain functions, and the schema a model knows them by.

A tool has two readers. The model sees a JSON schema — a name, a description,
the arguments and their types — and answers with a call to it, the arguments
as a JSON string. The agent loop sees a Python function to run on them.
:class:`~agent_from_scratch.tools.base.Tool` derives the one from the other, so
a tool is written as a function and nothing else. What the model should know
about a single argument goes in ``Annotated[..., Field(description=...)]``.

A tool's docstring is sent to the model verbatim, so it is written for the model
to read; notes for us go in comments. One module per tool, exported here.
"""

from .base import Tool
from .calculator import CALCULATOR, calculator

__all__ = [
    "CALCULATOR",
    "Tool",
    "calculator",
]
