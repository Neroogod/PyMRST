"""Native port of ADFNE15 ``Field.m``."""


def field(value, name):
    """Select a named field from a mapping, dataclass, or object."""
    if isinstance(value, dict):
        return value[name]
    return getattr(value, name)
