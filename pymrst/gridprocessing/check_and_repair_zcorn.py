"""Detect and repair corner-depth inversions (MRST ``checkAndRepairZCORN``)."""

from __future__ import annotations

import numpy as np


def check_and_repair_zcorn(
    zcorn,
    cart_dims,
    *,
    active=None,
    return_counts: bool = False,
):
    """Repair invalid vertical ordering in an ECLIPSE ``ZCORN`` array.

    Depth is positive downward. For active cells, a top corner deeper than its
    bottom corner is moved to the bottom. Across two active neighboring layers,
    a lower cell top above the upper cell bottom is moved down to that bottom.
    The returned flat array preserves the input's Fortran ordering.
    """
    dims = np.asarray(cart_dims, dtype=np.int64).ravel()
    if dims.shape != (3,) or np.any(dims <= 0):
        raise ValueError("cart_dims must contain three positive integers")
    values = np.asarray(zcorn)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("zcorn must be numeric")
    expected = int(np.prod(2 * dims))
    if values.size != expected:
        raise ValueError(
            f"expected {expected} ZCORN values for dimensions {tuple(dims)}, "
            f"got {values.size}"
        )
    z = np.asarray(values, dtype=float).reshape(tuple(2 * dims), order="F").copy()

    if active is None:
        act = np.ones(tuple(dims), dtype=bool)
    else:
        raw = np.asarray(active)
        if raw.size != int(np.prod(dims)):
            raise ValueError("active must specify one value per Cartesian cell")
        if not (
            np.issubdtype(raw.dtype, np.number)
            or np.issubdtype(raw.dtype, np.bool_)
        ):
            raise TypeError("active must be numeric or boolean")
        act = raw.astype(bool).reshape(tuple(dims), order="F")
    corner_active = np.repeat(np.repeat(act, 2, axis=0), 2, axis=1)

    top = z[:, :, 0::2]
    bottom = z[:, :, 1::2]
    invalid_top = (top > bottom) & corner_active
    top[invalid_top] = bottom[invalid_top]
    top_below_bottom = int(np.count_nonzero(invalid_top))

    upper_bottom = z[:, :, 1:-1:2]
    lower_top = z[:, :, 2::2]
    active_face = corner_active[:, :, :-1] & corner_active[:, :, 1:]
    invalid_interface = (upper_bottom > lower_top) & active_face
    lower_top[invalid_interface] = upper_bottom[invalid_interface]
    bottom_below_lower_top = int(np.count_nonzero(invalid_interface))

    repaired = z.ravel(order="F")
    if return_counts:
        return repaired, {
            "top_below_bottom": top_below_bottom,
            "bottom_below_lower_top": bottom_below_lower_top,
        }
    return repaired


checkAndRepairZCORN = check_and_repair_zcorn

__all__ = ["check_and_repair_zcorn", "checkAndRepairZCORN"]
