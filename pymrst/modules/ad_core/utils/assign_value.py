"""Replace values without changing an automatic-differentiation Jacobian."""

from __future__ import annotations

import numpy as np

from pymrst.utils.adi import value


def assign_value(x, replacement, indices):
    """Return ``x`` with selected numerical values replaced.

    AD objects retain unchanged Jacobian blocks. Python indices are zero-based.
    """
    replacement = np.asarray(value(replacement), dtype=float)
    if hasattr(x, "val") and hasattr(x, "jac"):
        out = x.copy()
        out.val[indices] = replacement
        return out
    out = np.asarray(x).copy()
    out[indices] = replacement
    return out


assignValue = assign_value

__all__ = ["assign_value", "assignValue"]
