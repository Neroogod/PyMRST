"""Isolated-process entrypoint for the curated example catalog.

The catalog (``example_service.ExampleCatalogService``) is the sole source
of trusted module/function strings. ``request["example_id"]`` — the only
client-controlled input here — is used exclusively as a lookup key into
that catalog, never as, or to construct, an importable path.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Mapping

import numpy as np

from ..services.example_service import ExampleCatalogService

_MAX_DEPTH = 6
_MAX_ITEMS = 50


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Best-effort, never-raising conversion of an arbitrary example return
    value into something ``json.dumps`` can serialize. Unrecognized objects
    fall back to a type name and truncated repr rather than propagating an
    exception — a run that produced a real result should not be reported as
    failed just because its result was hard to summarize."""
    try:
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, np.generic):
            return _json_safe(value.item(), depth)
        if depth >= _MAX_DEPTH:
            return f"<{type(value).__name__}: max summary depth reached>"
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "fc":
                finite = value[np.isfinite(value)]
            else:
                finite = value.reshape(-1)
            return {
                "kind": "array", "shape": list(value.shape), "dtype": str(value.dtype),
                "minimum": float(finite.min()) if finite.size else None,
                "maximum": float(finite.max()) if finite.size else None,
                "mean": float(finite.mean()) if finite.size else None,
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "kind": type(value).__name__,
                **{
                    field.name: _json_safe(getattr(value, field.name), depth + 1)
                    for field in dataclasses.fields(value)
                },
            }
        if isinstance(value, Mapping):
            items = list(value.items())[:_MAX_ITEMS]
            return {str(key): _json_safe(item, depth + 1) for key, item in items}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item, depth + 1) for item in value[:_MAX_ITEMS]]
        return {"kind": type(value).__name__, "repr": repr(value)[:2000]}
    except Exception as error:  # noqa: BLE001 - summarization must never crash the run
        return {"unsummarizable": type(value).__name__, "error": str(error)}


def run_example(context, request: dict) -> dict:
    example_id = str(request.get("example_id", "")).strip()
    definition = ExampleCatalogService().get(example_id)
    context.progress(0.05, f"Loading {definition.title}")
    context.raise_if_cancelled()
    module = importlib.import_module(definition.module)
    function = getattr(module, definition.function)
    context.progress(0.15, f"Running {definition.title}")
    value = function()
    context.raise_if_cancelled()
    context.progress(0.9, "Summarizing result")
    return {
        "example_id": definition.example_id, "title": definition.title,
        "result": _json_safe(value),
    }


__all__ = ["run_example"]
