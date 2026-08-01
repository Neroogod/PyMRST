from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.compositional.models.natvars import PressureReducedLinearSystem
from pymrst.utils.adi import init_variables_adi


def test_component_rows_collapse_to_weighted_pressure_equation():
    matrix = sp.csr_matrix(
        [[2.0, 0.0, 0.0, 0.0],
         [0.0, 3.0, 0.0, 0.0],
         [5.0, 0.0, 0.0, 0.0],
         [0.0, 7.0, 0.0, 0.0]]
    )
    rhs = np.array([1.0, 2.0, 3.0, 4.0])
    state = SimpleNamespace(
        pressure=np.array([100.0, 110.0]),
        components=np.full((2, 2), 0.5),
        L=np.ones(2),
    )
    model = SimpleNamespace(
        water=False,
        singlePhaseStrategy="numerical",
        twoPhaseStrategy="numerical",
        singlePhaseDifferentiation="none",
        twoPhaseDifferentiation="none",
    )
    system = PressureReducedLinearSystem(
        A=matrix,
        b=rhs,
        keepNum=4,
        accumulationTerms=[np.ones(2), np.ones(2)],
        model=model,
        state=state,
    ).assembleSystem()
    assert system.A.shape == (2, 2)
    assert np.allclose(system.A.toarray(), [[7.0, 0.0], [0.0, 10.0]])
    assert np.allclose(system.b, [4.0, 6.0])
    assert np.allclose(system.w, 1.0)


def test_plain_reduced_system_fallback_is_preserved_without_context():
    matrix = sp.csr_matrix([[4.0, 1.0], [2.0, 3.0]])
    system = PressureReducedLinearSystem(
        A=matrix, b=np.array([1.0, 2.0]), keepNum=1
    ).assembleSystem()
    assert system.assembled
    assert np.allclose(system.A.toarray(), [[4.0 - 2.0 / 3.0]])
    assert np.allclose(system.b, [1.0 - 2.0 / 3.0])


class _State:
    def __init__(self):
        self.pressure = np.array([10.0, 20.0])
        self.z = np.full((2, 2), 0.5)
        self.x = np.full((2, 2), 0.5)
        self.y = np.full((2, 2), 0.5)
        self.sO = np.full(2, 0.5)
        self.sG = np.full(2, 0.5)
        self.L = np.full(2, 0.5)

    def copy(self):
        other = _State()
        for name, item in self.__dict__.items():
            setattr(other, name, np.array(item, copy=True))
        return other


class _FlashModel:
    water = False
    nonlinearTolerance = 1.0e-3
    EOSModel = SimpleNamespace(getComponentNames=lambda: ("C1", "C2"))

    def __init__(self, base_pressure):
        self.base_pressure = np.asarray(base_pressure, dtype=float)

    def getFlag(self, state):
        active = np.ones(state.pressure.size, dtype=bool)
        return ~active, ~active, active

    def _sync_phase_state_from_flash(self, state):
        dp = state.pressure - self.base_pressure
        state.x[:, 0] += 0.1 * dp
        state.y[:, 0] += 0.2 * dp
        state.sO += 0.3 * dp
        state.sG -= 0.3 * dp
        state.L += 0.05 * dp
        return state


def test_pressure_solution_reconstructs_natural_variable_blocks():
    state = _State()
    model = _FlashModel(state.pressure)
    primaries = init_variables_adi(
        state.pressure,
        state.x[:, 0],
        state.sO,
        state.y[:, 0],
        state.sG,
    )
    system = PressureReducedLinearSystem(
        model=model,
        state=state,
        equations=[primaries[0]],
        types=["cell"],
        primaryVariables=["pressure", "v_C1", "sl", "w_C1", "sv"],
    )
    update, report = system.processResultAfterSolve(
        np.array([1.0, 2.0]), {"ok": True}
    )
    expected = np.concatenate(
        [
            [1.0, 2.0],
            [0.1, 0.2],
            [0.3, 0.6],
            [0.2, 0.4],
            [-0.3, -0.6],
        ]
    )
    assert np.allclose(update, expected)
    assert report == {"ok": True}
