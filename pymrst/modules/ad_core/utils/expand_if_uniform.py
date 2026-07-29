"""Reverse uniform matrix compaction into column values."""

from __future__ import annotations

from typing import Any

import numpy as np


def expand_if_uniform(values: Any):
    """Port MRST ``expandIfUniform``.

    Numeric/logical matrices with multiple columns become lists of column
    vectors; all other inputs are returned unchanged.
    """
    if isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] > 1:
        return [values[:, index] for index in range(values.shape[1])]
    return values


expandIfUniform = expand_if_uniform

__all__ = ["expand_if_uniform", "expandIfUniform"]
