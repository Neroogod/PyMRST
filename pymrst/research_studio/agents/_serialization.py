"""Small strict-JSON helpers for persistent agent contracts."""

from __future__ import annotations

from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping


def json_value(value: Any, path: str = "$") -> Any:
    """Return a JSON-compatible copy and reject ambiguous runtime objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"unsupported persistent value {type(value).__name__} at {path}"
    )


def frozen_json_mapping(value: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    """Validate, detach, and recursively freeze a JSON mapping."""
    return _freeze(json_value(value, path))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def require_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> dict[str, Any]:
    data = dict(value)
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return data


__all__ = ["frozen_json_mapping", "json_value", "require_fields"]
