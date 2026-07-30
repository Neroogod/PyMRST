"""Tests for MRST ``differentRegionFunctionsExample`` behavior."""

import numpy as np

from pymrst.modules.ad_core.examples import (
    different_region_functions_example,
)


def _check(result):
    saturation = np.array([0.2, 0.4, 0.6, 0.8])
    expected_water = np.array([0.04, 0.16, 0.6, 0.8])
    oil_saturation = 1.0 - saturation
    expected_oil = np.array([
        oil_saturation[0] ** 2,
        oil_saturation[1] ** 2,
        oil_saturation[2],
        oil_saturation[3],
    ])
    assert np.array_equal(result.regions, [1, 1, 2, 2])
    assert np.allclose(result.water, expected_water)
    assert np.allclose(result.oil, expected_oil)
    assert np.allclose(
        np.diag(result.water_jacobian), [0.4, 0.8, 1.0, 1.0]
    )
    assert np.allclose(
        np.diag(result.oil_jacobian), [-1.6, -1.2, -1.0, -1.0]
    )


def test_regions_are_loaded_from_rock():
    _check(different_region_functions_example(via_rock=True))


def test_regions_can_be_set_directly_on_state_function():
    _check(different_region_functions_example(via_rock=False))
