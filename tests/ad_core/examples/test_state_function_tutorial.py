"""Tests for the runnable MRST ``stateFunctionTutorial`` counterpart."""

import numpy as np
import pytest

from pymrst.modules.ad_core.examples import state_function_tutorial


def test_tutorial_exposes_graph_cache_values_and_ad_derivatives():
    saturation = np.array([0.2, 0.4, 0.6, 0.8])
    result = state_function_tutorial(saturation)

    assert result.grouping_names == (
        "PVTPropertyFunctions",
        "FlowPropertyFunctions",
        "FlowDiscretization",
    )
    assert "RelativePermeability" in result.flow_properties
    assert "Mobility" in result.flow_properties
    assert ("self", "RelativePermeability") in result.mobility_dependencies
    assert ("PVTPropertyFunctions", "Viscosity") in result.mobility_dependencies
    assert np.allclose(result.relative_permeability[0], saturation**2)
    assert np.allclose(result.relative_permeability[1], (1.0 - saturation) ** 2)
    assert np.allclose(result.mobility[0], saturation**2)
    assert np.allclose(result.mobility[1], (1.0 - saturation) ** 2 / 2.0)
    assert np.allclose(
        np.diag(result.mobility_saturation_jacobians[0]), 2.0 * saturation
    )
    assert np.allclose(
        np.diag(result.mobility_saturation_jacobians[1]),
        -(1.0 - saturation),
    )
    assert set(result.cached_flow_properties) >= {
        "Saturations",
        "RelativePermeability",
        "Mobility",
    }


def test_tutorial_rejects_empty_state():
    with pytest.raises(ValueError):
        state_function_tutorial([])
