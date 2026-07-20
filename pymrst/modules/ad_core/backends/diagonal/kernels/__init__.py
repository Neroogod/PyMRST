"""Optional compiled kernels for diagonal AD backends."""

from __future__ import annotations

try:  # pragma: no cover - exercised only when numba is installed.
    from pymrst.modules.ad_core.backends.diagonal.kernels.numba_kernels import (
        HAS_NUMBA_KERNELS,
        add_row_major,
        left_scale_row_major,
        matmul_left_row_major,
        matmul_left_triplet,
        row_repeat_row_major,
        row_repeat_triplet,
        take_rows_row_major,
        take_rows_triplet,
        vstack_row_major,
        vstack_triplet,
    )
except Exception:  # pragma: no cover - import failure is a supported fallback.
    HAS_NUMBA_KERNELS = False
    add_row_major = None
    left_scale_row_major = None
    matmul_left_row_major = None
    matmul_left_triplet = None
    row_repeat_row_major = None
    row_repeat_triplet = None
    take_rows_row_major = None
    take_rows_triplet = None
    vstack_row_major = None
    vstack_triplet = None


__all__ = [
    "HAS_NUMBA_KERNELS",
    "add_row_major",
    "left_scale_row_major",
    "matmul_left_row_major",
    "matmul_left_triplet",
    "row_repeat_row_major",
    "row_repeat_triplet",
    "take_rows_row_major",
    "take_rows_triplet",
    "vstack_row_major",
    "vstack_triplet",
]
