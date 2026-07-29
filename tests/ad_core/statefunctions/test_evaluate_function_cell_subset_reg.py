"""Tests for MRST ``evaluateFunctionCellSubsetReg``."""

import numpy as np

from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    init_variables_diagonal,
)
from pymrst.modules.ad_core.statefunctions.flowprops.hysteresis import (
    evaluateFunctionCellSubsetReg,
)


def test_regions_and_offset_numbering():
    x = np.arange(1.0, 7.0)
    funcs = [lambda v: 2 * v, lambda v: 10 + v]
    assert np.array_equal(
        evaluateFunctionCellSubsetReg(None, funcs, [1, 2, 1, 2, 1, 2], x),
        [2, 12, 6, 14, 10, 16],
    )
    assert np.array_equal(
        evaluateFunctionCellSubsetReg(None, funcs, [4, 5, 4, 5, 4, 5], x),
        [2, 12, 6, 14, 10, 16],
    )


def test_ad_jacobian_is_scattered_with_values():
    (x,) = init_variables_diagonal(np.arange(1.0, 7.0))
    funcs = [lambda v: 2 * v, lambda v: v * v]
    out = evaluateFunctionCellSubsetReg(
        None, funcs, np.array([1, 2, 1, 2, 1, 2]), x
    )
    assert np.array_equal(out.val, [2, 4, 6, 16, 10, 36])
    assert np.array_equal(
        out.to_adi().jac[0].diagonal(), [2, 4, 2, 8, 2, 12]
    )
