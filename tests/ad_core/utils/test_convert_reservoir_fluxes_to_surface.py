"""Tests for MRST ``convertReservoirFluxesToSurface``."""

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils.convert_reservoir_fluxes_to_surface import (
    convert_reservoir_fluxes_to_surface,
)
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params.rock import make_rock
from pymrst.utils.units import DARCY, MILLI


def _model():
    grid = compute_geometry(cart_grid([4, 1, 1], [4.0, 1.0, 1.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    return SimpleNamespace(operators=setup_operators_tpfa(grid, rock))


def test_surface_flux_uses_upstream_bfactor_for_each_phase():
    model = _model()
    state = {
        "flux": np.array([[2.0, -5.0], [3.0, -6.0], [4.0, -7.0]]),
        "bfactor": np.array([
            [1.1, 0.7], [1.2, 0.8], [1.3, 0.9], [1.4, 1.0],
        ]),
        "upstreamFlag": np.array([
            [True, False], [True, False], [True, False],
        ]),
    }
    converted = convert_reservoir_fluxes_to_surface(model, state)[0]
    expected = np.array([
        [2.0 * 1.1, -5.0 * 0.8],
        [3.0 * 1.2, -6.0 * 0.9],
        [4.0 * 1.3, -7.0 * 1.0],
    ])
    assert np.allclose(converted["surfaceFlux"], expected)
    assert "surfaceFlux" not in state


def test_single_and_sequence_inputs_follow_mrst_list_return_contract():
    model = _model()
    state = {
        "flux": np.ones((3, 1)),
        "bfactor": np.ones((4, 1)),
        "upstreamFlag": np.ones((3, 1), dtype=bool),
    }
    single = convert_reservoir_fluxes_to_surface(model, state)
    sequence = convert_reservoir_fluxes_to_surface(model, [state, state])
    assert isinstance(single, list) and len(single) == 1
    assert len(sequence) == 2
    assert all(np.allclose(item["surfaceFlux"], 1.0) for item in sequence)


def test_missing_bfactor_reports_extra_state_output_requirement():
    with pytest.raises(ValueError, match="extraStateOutput"):
        convert_reservoir_fluxes_to_surface(
            _model(),
            {
                "flux": np.ones((3, 1)),
                "upstreamFlag": np.ones((3, 1), dtype=bool),
            },
        )
