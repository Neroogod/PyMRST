"""Tests for MRST ``getMinMaxPhaseSaturationsFromRelPerm``."""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.initialization import (
    getMinMaxPhaseSaturationsFromRelPerm,
)


def test_three_phase_endpoints_and_region_selection():
    fluid = SimpleNamespace(
        krW=[
            lambda s: np.clip((s - 0.2) / 0.7, 0, 1),
            lambda s: np.clip((s - 0.1) / 0.8, 0, 1),
        ],
        krOW=lambda s: np.clip((s - 0.15) / 0.75, 0, 1),
        krOG=lambda s: np.clip((s - 0.25) / 0.65, 0, 1),
        krG=lambda s: np.clip((s - 0.05) / 0.8, 0, 1),
    )
    model = SimpleNamespace(
        fluid=fluid, water=True, oil=True, gas=True,
        get_phase_index=lambda phase: {"W": 0, "O": 1, "G": 2}[phase],
    )
    s_min, s_max = getMinMaxPhaseSaturationsFromRelPerm(
        model, tol=0.05, region=2
    )
    assert np.allclose(s_min, [0.1, 0.25, 0.05])
    assert np.allclose(s_max, [0.9, 0.9, 0.85])
