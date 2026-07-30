"""Record internal and external state-function dependencies."""

from __future__ import annotations

from types import SimpleNamespace


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _set(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def add_property_dependence(prop, name, grouping=None):
    """Port MRST ``addPropertyDependence``.

    Plain names without a grouping are appended to ``dependencies``.
    Structured dependencies or explicitly grouped names are appended to
    ``externals``.
    """
    names = list(name) if isinstance(name, (list, tuple)) else name
    dependencies = list(_get(prop, "dependencies"))
    externals = list(_get(prop, "externals"))
    if grouping is None or grouping == "":
        if isinstance(names, (dict, SimpleNamespace)):
            externals.append(names)
        elif isinstance(names, list):
            dependencies.extend(names)
        else:
            dependencies.append(names)
    else:
        externals.append(SimpleNamespace(name=names, grouping=grouping))
    _set(prop, "dependencies", dependencies)
    _set(prop, "externals", externals)
    return prop


addPropertyDependence = add_property_dependence

__all__ = ["add_property_dependence", "addPropertyDependence"]
