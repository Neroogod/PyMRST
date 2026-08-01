from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver
from pymrst.modules.ad_core.statefunctions.state_function import State


def _well(*, status=True, control="rate", value=0.0, compi=(0.0, 0.0, 1.0)):
    return SimpleNamespace(
        status=status, type=control, val=value, compi=np.asarray(compi),
    )


def test_changed_injector_is_seeded_from_target_and_composition():
    state = State(
        qWs=np.zeros(2), qOs=np.zeros(2), qGs=np.zeros(2),
    )
    old = [_well(value=0.0), _well(value=-2.0)]
    new = [_well(value=4.5), _well(value=-3.0)]

    seeded = NonLinearSolver._seed_injector_control_state(state, old, new)

    np.testing.assert_allclose(seeded["qWs"], [0.0, 0.0])
    np.testing.assert_allclose(seeded["qOs"], [0.0, 0.0])
    np.testing.assert_allclose(seeded["qGs"], [4.5, 0.0])
    # The input and changed producer are both preserved.
    np.testing.assert_allclose(state["qGs"], [0.0, 0.0])


def test_unchanged_or_shut_injector_does_not_copy_state():
    state = State(qWs=np.zeros(1), qOs=np.zeros(1), qGs=np.zeros(1))
    well = _well(status=False, value=4.5)
    assert NonLinearSolver._seed_injector_control_state(
        state, [well], [well],
    ) is state


def test_linear_failure_is_returned_for_timestep_cutting():
    class Model:
        @staticmethod
        def get_equations(state0, state, dt):
            return []

        @staticmethod
        def get_convergence(equations, dt):
            return False, {"CNV": 1.0}

    solver = NonLinearSolver(max_iterations=2)

    def fail(*args):
        raise RuntimeError("iterative solve did not converge")

    solver._solve_linear = fail
    state = State(x=np.ones(1))
    returned, report = solver.solve_ministep(Model(), state, state.copy(), 1.0)

    assert returned["x"][0] == 1.0
    assert not report.converged
    assert np.isinf(report.residuals[-1]["linear_failure"])


def test_reassembly_consumes_stale_well_limit_switch_flag():
    class Reassembled(Exception):
        pass

    class Dummy:
        facility = SimpleNamespace(within_limits=False)

        def init_state_ad(self, state):
            assert self.facility.within_limits
            raise Reassembled

    with pytest.raises(Reassembled):
        GenericBlackOilModel.get_equations(Dummy(), State(), State(), 1.0)


def test_cpr_amg_rebuild_interval_must_be_positive():
    assert CPRSolver(amg_rebuild_interval=3).amg_rebuild_interval == 3
    with pytest.raises(ValueError, match="positive"):
        CPRSolver(amg_rebuild_interval=0)
