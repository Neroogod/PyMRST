"""Tests for the runnable MRST ``compositeModelTutorial`` counterpart."""

import numpy as np

from pymrst.modules.ad_core.examples import composite_model_tutorial


def test_composite_tutorial_is_conservative_and_couples_jacobian_blocks():
    result = composite_model_tutorial()

    assert result.residual_without_coupling.shape == (16,)
    assert result.residual_with_coupling.shape == (16,)
    assert result.jacobian_shape == (16, 16)
    assert result.off_diagonal_nonzeros == 16
    assert result.conservation_error < 1e-14
    assert not np.array_equal(
        result.residual_without_coupling, result.residual_with_coupling
    )
    assert set(result.layout) == {"LayerA", "LayerB"}


def test_zero_pressure_difference_has_no_coupling_residual():
    result = composite_model_tutorial(pressure_a=1e7, pressure_b=1e7)

    assert np.allclose(
        result.residual_without_coupling, result.residual_with_coupling
    )
    # Derivatives remain coupled even at zero transfer.
    assert result.off_diagonal_nonzeros == 16
