"""Native port of ADFNE15 ``setstructfields.m``."""


def setstructfields(structure, new_fields):
    output = dict(structure)
    if not isinstance(new_fields, dict):
        return output
    for key, value in new_fields.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = setstructfields(output[key], value)
        else:
            output[key] = value
    return output
