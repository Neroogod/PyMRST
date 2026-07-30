"""Tests for the runnable MRST ``MPFAvsTPFAwithADExample`` counterpart."""

import numpy as np

from pymrst.modules.ad_core.examples import mpfa_vs_tpfa_with_ad_example


def test_tpfa_and_mpfa_are_conservative_but_genuinely_different():
    result = mpfa_vs_tpfa_with_ad_example()

    assert result.tpfa_gradient.shape == result.mpfa_gradient.shape
    assert result.tpfa_jacobian.shape == result.mpfa_jacobian.shape
    assert result.tpfa_constant_error < 1e-14
    assert result.mpfa_constant_error < 1e-14
    assert result.gradient_difference > 1e-8
    assert result.jacobian_difference > 1e-8


def test_ad_jacobians_reproduce_the_linear_operator_action():
    result = mpfa_vs_tpfa_with_ad_example()
    # With independent pressure variables, applying the Jacobian to the
    # pressure vector reproduces each linear pressure-gradient operation.
    # Recover that vector from the known 4x3 cell-centroid layout.
    x = np.tile(np.arange(4, dtype=float) + 0.5, 3)
    y = np.repeat(np.arange(3, dtype=float) + 0.5, 4)
    pressure = 2.0 * x - 0.75 * y
    assert np.allclose(result.tpfa_jacobian @ pressure, result.tpfa_gradient)
    assert np.allclose(result.mpfa_jacobian @ pressure, result.mpfa_gradient)
