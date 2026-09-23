import inspect
from collections.abc import Collection

import docstring_parser


def function_to_input_schema(func, exclude: Collection[str] = ()) -> dict:
    """Describe a function's arguments as a JSON schema.

    Parameters named in ``exclude`` are left out: an argument the caller
    supplies itself is not one the model should be asked for.

    Each argument carries the description the docstring gives it. That text
    is what the model reads to decide what to pass, so it belongs in the
    schema rather than only in the prose a human sees.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    descriptions = {
        param.arg_name: param.description
        for param in docstring_parser.parse(func.__doc__ or "").params
        if param.description
    }

    params = [
        param for param in signature.parameters.values() if param.name not in exclude
    ]

    parameters = {}
    for param in params:
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}
        if param.name in descriptions:
            parameters[param.name]["description"] = descriptions[param.name]

    required = [param.name for param in params if param.default == inspect._empty]

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


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


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def function_to_tool_definition(func) -> dict:
    return format_tool_definition(
        func.__name__, function_to_description(func), function_to_input_schema(func)
    )
