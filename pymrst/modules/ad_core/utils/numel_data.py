"""MATLAB-like element count for result data containers."""

from __future__ import annotations

from typing import Any

import numpy as np


def numel_data(data: Any) -> int:
    """Port MRST ``numelData`` (an overridable ``numel`` alias)."""
    method = getattr(data, "numel_data", None)
    if callable(method):
        return int(method())
    if isinstance(data, np.ndarray):
        return int(data.size)
    if isinstance(data, (list, tuple, dict, str, bytes)):
        return len(data)
    try:
        return len(data)
    except TypeError:
        return 1


numelData = numel_data

__all__ = ["numel_data", "numelData"]
