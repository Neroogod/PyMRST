"""MRST-level equation concatenation entry point."""

from __future__ import annotations

import numpy as np

from pymrst.utils.adi import combine_equations as _combine_ad_equations


def combine_equations(*equations):
    """Vertically concatenate numeric equations or assemble AD equations."""
    if any(hasattr(equation, "jac") for equation in equations):
        return _combine_ad_equations(*equations)
    return np.concatenate(
        [np.atleast_1d(np.asarray(equation, dtype=float)).ravel()
         for equation in equations]
    )


combineEquations = combine_equations

__all__ = ["combine_equations", "combineEquations"]
