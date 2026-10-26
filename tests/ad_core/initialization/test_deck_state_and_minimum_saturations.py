"""Validation for MRST ``initStateDeck`` and ``getMinimumPhaseSaturations``."""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.initialization.deck import init_state_deck_ad
from pymrst.modules.ad_core.initialization.equilibrium import (
    EquilibriumRegion,
    _connate_saturations,
)


def test_explicit_deck_state_copies_phase_fields_and_live_oil_status():
    model = SimpleNamespace(
        G=SimpleNamespace(cells=SimpleNamespace(num=3)),
        n_phases=3,
        fluid=SimpleNamespace(disgas=True),
    )
    deck = {
        "PRESSURE": [100.0, 110.0, 120.0],
        "SWAT": [0.2, 0.3, 0.4],
        "SGAS": [0.0, 0.1, 0.0],
        "RS": [5.0, 6.0, 7.0],
    }
    state = init_state_deck_ad(model, deck)

    assert np.array_equal(state.pressure, deck["PRESSURE"])
    assert np.array_equal(state.sW, deck["SWAT"])
    assert np.allclose(state.sO, [0.8, 0.6, 0.6])
    assert np.array_equal(state.sG, deck["SGAS"])
    assert np.array_equal(state.rs, deck["RS"])
    assert np.array_equal(state.status, [1, 3, 1])
    state.pressure[0] = -1.0
    assert deck["PRESSURE"][0] == 100.0


def test_minimum_phase_saturations_use_region_then_fluid_connate():
    model = SimpleNamespace(
        n_phases=3,
        fluid=SimpleNamespace(swcon=None),
        raw_fluid=SimpleNamespace(swcon=0.12),
    )
    region = EquilibriumRegion(0.0, 1.0, [1.0, 2.0])
    assert np.array_equal(_connate_saturations(model, region), [0.12, 0.0, 0.0])

    region.s_min = np.array([0.2, 0.05, 0.01])
    assert np.array_equal(
        _connate_saturations(model, region), region.s_min
    )
