"""MRST ``evaluateFunctionCellSubsetReg`` regional evaluator."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD
from pymrst.modules.ad_core.backends.diagonal.utils import double2GenericAD
from pymrst.utils.adi.adi import ADI


def _constant_ad(values, sample):
    if isinstance(sample, GenericAD):
        return double2GenericAD(values, sample)
    if isinstance(sample, ADI):
        return ADI(
            np.asarray(values, dtype=float),
            [sp.csr_array((len(values), block.shape[1])) for block in sample.jac],
        )
    return np.asarray(values, dtype=float)


def evaluate_function_cell_subset_reg(prop, fn, regions, *args):
    """Evaluate region functions on their cell subsets and scatter the result."""
    if not isinstance(fn, (list, tuple)):
        return fn(*args)
    regions = np.asarray(regions, dtype=np.int64).ravel()
    if regions.size == 0:
        raise ValueError("regions cannot be empty")
    nc = regions.size
    sample = next((arg for arg in args if isinstance(arg, (GenericAD, ADI))), None)
    out = _constant_ad(np.zeros(nc), sample) if sample is not None else np.zeros(nc)
    offset = int(regions.min()) - 1 if regions.min() > 1 else 0
    for index, function in enumerate(fn, 1):
        active = regions == index + offset
        if not np.any(active):
            continue
        selected = [
            arg[active] if hasattr(arg, "__len__") and len(arg) == nc else arg
            for arg in args
        ]
        out[active] = function(*selected)
    return out


evaluateFunctionCellSubsetReg = evaluate_function_cell_subset_reg

__all__ = [
    "evaluate_function_cell_subset_reg",
    "evaluateFunctionCellSubsetReg",
]
