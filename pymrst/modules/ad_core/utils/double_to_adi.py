"""Convert numerical values to the automatic-differentiation type of a sample."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def double_to_adi(values, sample):
    """Return ``values`` with zero Jacobians shaped like ``sample``."""
    array = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    if not hasattr(sample, "jac"):
        return np.asarray(values, dtype=float)
    converter = getattr(sample, "convert_double", None)
    if converter is None:
        converter = getattr(sample, "convertDouble", None)
    if converter is not None:
        return converter(array)
    zeros = [
        sp.csr_array((array.size, block.shape[1]), dtype=float)
        for block in sample.jac
    ]
    return type(sample)(array, zeros)


double2ADI = double_to_adi

__all__ = ["double_to_adi", "double2ADI"]
