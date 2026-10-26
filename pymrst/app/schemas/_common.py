"""Strict application-schema helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_value(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_value(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported application value {type(value).__name__} at {path}")


def required(value: Mapping[str, Any], fields: set[str], name: str) -> dict[str, Any]:
    data = dict(value)
    if set(data) != fields:
        raise ValueError(
            f"invalid {name} fields: missing={sorted(fields-set(data))}, "
            f"unknown={sorted(set(data)-fields)}"
        )
    return data


def nonempty(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


__all__ = ["json_value", "nonempty", "required", "utc_now"]
