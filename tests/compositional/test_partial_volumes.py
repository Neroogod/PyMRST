from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymrst.modules.compositional.utils import getPartialVolumes
from pymrst.modules.compositional.fluids import get_benchmark_mixture
from pymrst.modules.compositional.models.eos import EquationOfStateModel
from pymrst.utils.adi import init_variables_adi
from pymrst.utils.units import BARSA


def _model():
    mixture = SimpleNamespace(molarMass=np.array([0.02, 0.10]), names=("C1", "C5"))
    eos = SimpleNamespace(CompositionalMixture=mixture)
    return SimpleNamespace(EOSModel=eos, water=False)


def test_mixed_phase_cells_use_their_own_mrst_strategies():
    state = {
        "pressure": np.array([1.0e7, 1.1e7, 1.2e7]),
        "components": np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]),
        "L": np.array([1.0, 0.5, 0.0]),
        "rho": np.array([500.0, 100.0, 250.0]),
    }
    V, dVdp = getPartialVolumes(
        _model(),
        state,
        singlePhaseStrategy="analytical",
        twoPhaseStrategy="numerical",
        singlePhaseDifferentiation="none",
        twoPhaseDifferentiation="none",
    )
    expected_single = np.array(
        [[1.0 / 500.0 / 0.02, 1.0 / 500.0 / 0.10],
         [1.0 / 250.0 / 0.02, 1.0 / 250.0 / 0.10]]
    )
    assert np.allclose(V[[0, 2]], expected_single)
    assert np.allclose(V[1], 1.0)
    assert np.allclose(dVdp, 0.0)


def test_mixed_strategy_subset_preserves_requested_cell_order():
    state = {
        "pressure": np.arange(4, dtype=float) + 1.0e7,
        "components": np.full((4, 2), 0.5),
        "L": np.array([0.0, 0.5, 1.0, 0.5]),
        "rho": np.array([100.0, 200.0, 400.0, 800.0]),
    }
    V, _ = getPartialVolumes(
        _model(),
        state,
        subset=np.array([True, False, True, False]),
        singlePhaseStrategy="analytical",
        twoPhaseStrategy="numerical",
        singlePhaseDifferentiation="none",
        twoPhaseDifferentiation="none",
    )
    assert V.shape == (2, 2)
    assert np.allclose(V[:, 0], [1.0 / 100.0 / 0.02, 1.0 / 400.0 / 0.02])


def test_numerical_weights_support_reorder_and_well_variable_removal():
    pressure, composition, well = init_variables_adi(
        np.array([1.0, 2.0]), np.array([0.4, 0.6]), np.array([0.0])
    )
    acc = [2.0 * pressure + composition + well, pressure + 3.0 * composition + 2.0 * well]
    state = {
        "pressure": np.array([1.0e7, 1.1e7]),
        "components": np.full((2, 2), 0.5),
        "L": np.ones(2),
    }
    V, _ = getPartialVolumes(
        _model(),
        state,
        acc=acc,
        singlePhaseStrategy="numerical",
        singlePhaseDifferentiation="none",
        reorder=np.arange(5),
        wellVarIndices=[4],
    )
    cell_jacobian = np.array(
        [[2.0, 0.0, 1.0, 0.0],
         [0.0, 2.0, 0.0, 1.0],
         [1.0, 0.0, 3.0, 0.0],
         [0.0, 1.0, 0.0, 3.0]]
    )
    rhs = np.array([1.0 / BARSA, 1.0 / BARSA, 0.0, 0.0])
    expected = np.linalg.solve(cell_jacobian.T, rhs).reshape(2, 2).T
    assert np.allclose(V, expected)


def test_numerical_derivative_requires_iteration_after_first():
    state = {
        "pressure": np.array([2.0, 3.0]),
        "components": np.full((2, 2), 0.5),
        "L": np.ones(2),
        "w": np.array([[2.0, 1.0], [1.0, 3.0]]),
        "w_p": np.array([1.0, 1.0]),
    }
    _, first = getPartialVolumes(
        _model(), state, singlePhaseDifferentiation="numerical", iteration=1
    )
    _, later = getPartialVolumes(
        _model(), state, singlePhaseDifferentiation="numerical", iteration=2
    )
    assert np.allclose(first, 0.0)
    assert np.any(np.abs(later) > 0.0)


def test_analytical_eos_pressure_derivative_is_native_ad_and_epsilon_independent():
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    model = SimpleNamespace(EOSModel=eos, water=False, T=350.0)
    state = {
        "pressure": np.array([5.0e6, 5.5e6]),
        "T": np.full(2, 350.0),
        "components": np.tile([0.5, 0.3, 0.2], (2, 1)),
        "L": np.ones(2),
    }
    epsilon = 500.0
    values, derivative = getPartialVolumes(
        model,
        state,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="analytical",
        pressureEpsilon=epsilon,
    )
    values_other, derivative_other = getPartialVolumes(
        model,
        state,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="analytical",
        pressureEpsilon=1.0e5,
    )
    assert np.all(np.isfinite(derivative))
    assert np.any(np.abs(derivative) > 0.0)
    assert np.allclose(values, values_other, rtol=0.0, atol=0.0)
    assert np.allclose(derivative, derivative_other, rtol=0.0, atol=0.0)
    assert np.all(derivative > 0.0)


def test_explicit_density_has_no_fabricated_analytical_derivative():
    state = {
        "pressure": np.array([1.0e7]),
        "components": np.array([[0.5, 0.5]]),
        "L": np.ones(1),
        "rho": np.array([500.0]),
    }
    _, derivative = getPartialVolumes(
        _model(),
        state,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="analytical",
    )
    assert np.allclose(derivative, 0.0)


def test_water_analytical_weights_are_water_first_and_density_based():
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    model = SimpleNamespace(
        EOSModel=eos,
        water=True,
        water_rho_s=997.0,
        water_molar_mass=0.01801528,
        T=350.0,
    )
    state = {
        "pressure": np.array([5.0e6, 5.5e6]),
        "T": np.full(2, 350.0),
        "components": np.tile([0.5, 0.3, 0.2], (2, 1)),
        "L": np.ones(2),
    }
    values, derivative = getPartialVolumes(
        model,
        state,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="analytical",
    )
    assert values.shape == (2, 4)
    assert np.allclose(values[:, 0], 1.0 / 997.0)
    assert np.allclose(derivative[:, 0], 0.0)
    assert np.all(np.isfinite(values[:, 1:]))
    assert np.any(np.abs(derivative[:, 1:]) > 0.0)


def test_water_eos_weights_use_water_first_mass_and_molar_units():
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    model = SimpleNamespace(
        EOSModel=eos,
        water=True,
        water_rho_s=1000.0,
        water_molar_mass=0.018,
        T=350.0,
    )
    state = {
        "pressure": np.array([5.0e6]),
        "T": np.array([350.0]),
        "components": np.array([[0.5, 0.3, 0.2]]),
        "L": np.array([0.5]),
    }
    mass, _ = getPartialVolumes(
        model,
        state,
        twoPhaseStrategy="eos",
        twoPhaseDifferentiation="none",
        units="mass",
    )
    moles, _ = getPartialVolumes(
        model,
        state,
        twoPhaseStrategy="eos",
        twoPhaseDifferentiation="none",
        units="moles",
    )
    assert mass.shape == moles.shape == (1, 4)
    assert np.isclose(mass[0, 0], 1.0 / 1000.0)
    assert np.isclose(moles[0, 0], 0.018 / 1000.0)
