"""Parity tests for MRST diagonal-backend grid operators.

Covered symbols: ``twoPointGradient``, ``faceAverage``, ``singlePointUpwind``,
``discreteDivergence``, and ``discreteDivergenceDiagonalJac``.
"""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    init_variables_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.operators import (
    discreteDivergence,
    faceAverage,
    singlePointUpwind,
    twoPointGradient,
)
from pymrst.modules.ad_core.backends.diagonal.operators.discrete_divergence_diagonal_jac import (
    discreteDivergenceDiagonalJac,
)


N = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
OPT = SimpleNamespace(N=N, nc=4)


def test_numeric_operator_values_match_mrst_definitions():
    cells = np.array([1.0, 4.0, 9.0, 16.0])
    flux = np.array([2.0, -3.0, 5.0])
    assert np.array_equal(twoPointGradient(N, cells), [3.0, 5.0, 7.0])
    assert np.array_equal(faceAverage(N, cells), [2.5, 6.5, 12.5])
    assert np.array_equal(
        singlePointUpwind([True, False, True], N, cells),
        [1.0, 9.0, 9.0],
    )
    assert np.array_equal(
        discreteDivergence(None, flux, OPT),
        [2.0, -5.0, 8.0, -5.0],
    )
    assert np.array_equal(
        discreteDivergence(np.ones(4), flux, OPT),
        [3.0, -4.0, 9.0, -4.0],
    )


def test_generic_ad_values_and_jacobians_match_sparse_operators():
    (cells,) = init_variables_diagonal(np.array([1.0, 4.0, 9.0, 16.0]))
    grad = twoPointGradient(N, cells)
    assert np.array_equal(grad.val, [3.0, 5.0, 7.0])
    assert np.array_equal(
        grad.to_adi().jac[0].toarray(),
        [
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
        ],
    )

    avg = faceAverage(N, cells)
    assert np.array_equal(
        avg.to_adi().jac[0].toarray(),
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ],
    )

    upwind = singlePointUpwind([True, False, True], N, cells)
    assert np.array_equal(
        upwind.to_adi().jac[0].toarray(),
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    )


def test_discrete_divergence_combines_accumulation_and_flux_jacobians():
    acc, flux = init_variables_diagonal(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([2.0, -3.0, 5.0]),
    )
    result = discreteDivergence(acc, flux, OPT)
    assert np.array_equal(result.val, [12.0, 15.0, 38.0, 35.0])
    jac = result.to_adi().jac
    assert np.array_equal(jac[0].toarray(), np.eye(4))
    assert np.array_equal(
        jac[1].toarray(),
        [
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
    )


def test_low_level_divergence_jacobian_entry_point():
    (flux,) = init_variables_diagonal(np.array([2.0, -3.0, 5.0]))
    block = discreteDivergenceDiagonalJac(None, flux.jac[0], OPT)
    assert np.array_equal(
        block.to_sparse().toarray(),
        [
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
    )
