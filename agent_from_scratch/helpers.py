import inspect
from collections.abc import Collection
from typing import Any

import docstring_parser
from pydantic import Field, create_model
from pydantic.json_schema import GenerateJsonSchema


class _NoFieldTitles(GenerateJsonSchema):
    """Leave out the titles pydantic derives from field names.

    A title of "Location" above a property already keyed ``location``
    tells the model nothing it cannot see, and it pays for the reading.
    """

    def field_title_should_be_set(self, schema) -> bool:
        return False


def function_to_input_schema(func, exclude: Collection[str] = ()) -> dict:
    """Describe a function's arguments as a JSON schema.

    Parameters named in ``exclude`` are left out: an argument the caller
    supplies itself is not one the model should be asked for.

    Each argument carries the description the docstring gives it. That text
    is what the model reads to decide what to pass, so it belongs in the
    schema rather than only in the prose a human sees.

    The annotations are read by pydantic, so a container, a union or a
    ``Literal`` describes itself as precisely as a bare ``str`` does. An
    argument annotated ``Literal["fast", "slow"]`` reaches the model as an
    enum of the two words it accepts rather than as an open string.
    """
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    descriptions = function_to_argument_descriptions(func)

    fields: dict[str, Any] = {}
    for param in signature.parameters.values():
        if param.name in exclude:
            continue
        # An unannotated argument constrains nothing; saying so beats
        # guessing at a type the function never asked for.
        annotation = (
            Any if param.annotation is inspect.Parameter.empty else param.annotation
        )
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (
            annotation,
            Field(default, description=descriptions.get(param.name)),
        )

    schema = create_model(func.__name__, **fields).model_json_schema(
        schema_generator=_NoFieldTitles
    )
    # The model exists only to be described; its name is not part of the schema.
    schema.pop("title", None)
    return schema


def function_to_description(func) -> str:
    """Take the prose half of a function's docstring.

    The parameter section is left out. Each argument already carries its
    own description in the input schema, and the section documents
    arguments the caller injects as readily as ones the model supplies —
    repeating it here would invite the model to pass what it cannot see.
    """
    docstring = docstring_parser.parse(func.__doc__ or "")
    prose = [docstring.short_description, docstring.long_description]
    return "\n\n".join(part for part in prose if part)


def function_to_argument_descriptions(func) -> dict[str, str]:
    """Take the other half: what the docstring says about each argument.

    An argument the docstring passes over is absent rather than empty, so
    the schema can leave it undescribed instead of describing it as
    nothing.
    """
    return {
        param.arg_name: param.description
        for param in docstring_parser.parse(func.__doc__ or "").params
        if param.description
    }


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
