import inspect
from collections.abc import Collection


def function_to_input_schema(func, exclude: Collection[str] = ()) -> dict:
    """Describe a function's arguments as a JSON schema.

    Parameters named in ``exclude`` are left out: an argument the caller
    supplies itself is not one the model should be asked for.
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

    required = [param.name for param in params if param.default == inspect._empty]

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
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


def function_to_tool_definition(func) -> dict:
    return format_tool_definition(
        func.__name__, func.__doc__ or "", function_to_input_schema(func)
    )
