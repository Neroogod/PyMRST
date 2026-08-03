"""Native port of ADFNE15 ``Option.m``."""

from .setstructfields import setstructfields


_ALIASES = {"fc": "facecolor", "ec": "edgecolor", "fa": "facealpha",
            "r": "radius", "lw": "linewidth", "cnt": "center"}


def _pairs(values):
    values = list(values)
    if values and not isinstance(values[0], str):
        values = ["in", *values]
    if len(values) % 2:
        raise ValueError("options require key/value pairs")
    return {str(values[i]): values[i+1] for i in range(0, len(values), 2)}


def option(values=(), *defaults, **default_values):
    """Merge ADFNE key/value arguments over defaults and expand aliases."""
    baseline = _pairs(defaults)
    baseline.update(default_values)
    supplied = dict(values) if isinstance(values, dict) else _pairs(values)
    output = setstructfields(baseline, supplied)
    for short, long in _ALIASES.items():
        if short in output:
            output[long] = output[short]
    return output
