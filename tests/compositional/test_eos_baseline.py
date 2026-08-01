from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.compositional.eos import flash, simple_mixture, wilson_k
from pymrst.modules.compositional.models.eos.equation_of_state_model import (
    EquationOfStateModel,
)


def test_pr_model_surface_and_physical_cubic_roots():
    mixture = simple_mixture()
    model = EquationOfStateModel(fluid=mixture, eosname="PR")
    assert model.shortname() == "pr"
    assert model.getNumberOfComponents() == len(mixture.names)
    roots = model.solveCubicEOS(np.array([0.2]), np.array([0.02]))
    finite = roots[np.isfinite(roots)]
    assert finite.size >= 1
    assert np.all(finite > 0.02)


def test_flash_closes_phase_and_overall_compositions():
    mixture = simple_mixture()
    pressure = np.array([5.0e6, 1.0e7])
    temperature = np.array([300.0, 330.0])
    z = np.tile(np.full(len(mixture.names), 1.0 / len(mixture.names)), (2, 1))
    result = flash(mixture, pressure, temperature, z)
    assert np.all((result.V >= 0.0) & (result.V <= 1.0))
    assert np.allclose(result.x.sum(axis=1), 1.0)
    assert np.allclose(result.y.sum(axis=1), 1.0)
    reconstructed = (1.0 - result.V[:, None]) * result.x + result.V[:, None] * result.y
    assert np.allclose(reconstructed, z, atol=1.0e-8)


def test_wilson_values_are_positive_and_vectorized():
    mixture = simple_mixture()
    values = wilson_k(mixture, np.array([5.0e6, 1.0e7]), np.array([300.0, 330.0]))
    assert values.shape == (2, len(mixture.names))
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_zudkevitch_joffe_refusal_matches_mrst_2024a():
    with pytest.raises(NotImplementedError, match="Zudkevitch-Joffe"):
        EquationOfStateModel(fluid=simple_mixture(), eosname="ZJ")
