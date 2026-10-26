"""One-call single-step solve (MRST `standaloneSolveAD`).

The shortest path from a state to the next one: given a model, an initial state
and a timestep, advance a single step and return the new state and its report.
MRST wraps this as a one-line `simpleSchedule` of one step run through
`simulateScheduleAD`; here it is a thin call onto the nonlinear solver's
`solve_timestep`, which already does the ministep and any timestep cutting. Its
value is convenience - a test or a script that just wants "solve one step" need
not assemble a schedule.

The model carries its own driving forces (wells, bc, sources), exactly as in the
rest of the port, so none are passed here.
"""

from typing import Optional, Tuple

from pymrst.modules.ad_core.simulators.nonlinear_solver import (
    NonLinearSolver,
    StepReport,
)
from pymrst.modules.ad_core.statefunctions.state_function import State


def standalone_solve(state0: State, model, dt: float,
                     solver: Optional[NonLinearSolver] = None
                     ) -> Tuple[State, StepReport]:
    """Advance `state0` by `dt` under `model` and return `(state, report)`."""
    if solver is None:
        solver = NonLinearSolver()
    return solver.solve_timestep(model, state0, dt)


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
standaloneSolveAD = standalone_solve

if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model():
        g = compute_geometry(cart_grid([6, 6, 1], [300.0, 300.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 3.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        inj = add_well(None, g, rock, np.array([0]), well_type="rate",
                       val=2e-3, compi=np.array([1.0, 0.0]), sign=1, name="I")[0]
        prod = add_well(None, g, rock, np.array([g.cells.num - 1]),
                        well_type="bhp", val=150 * BARSA,
                        compi=np.array([0.0, 1.0]), sign=-1, name="P")[0]
        return GenericBlackOilModelAD(g, rock, fluid, [inj, prod], tol_well=1e-8)

    def test_single_step_advances_and_reports_converged():
        model = _model()
        nc = model.G.cells.num
        st = State(pressure=np.full(nc, 200.0 * BARSA), sW=np.full(nc, 0.2),
                   qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -1e-3]),
                   bhp=np.array([200.0 * BARSA, 150.0 * BARSA]))
        state, report = standalone_solve(st, model, 30.0 * DAY)
        assert report.converged, "one standalone step should converge"
        # the producer is on bhp control, so its bhp is exactly the target
        assert abs(float(state["bhp"][1]) - 150.0 * BARSA) < 1e-3
        # injection raised the average pressure above the 150 bar producer
        assert float(np.mean(state["pressure"])) > 150.0 * BARSA
        print("ok test_single_step_advances_and_reports_converged")

    def test_matches_solve_timestep_directly():
        """It is exactly the solver's own single step - same state out."""
        model = _model()
        nc = model.G.cells.num
        st = State(pressure=np.full(nc, 200.0 * BARSA), sW=np.full(nc, 0.2),
                   qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -1e-3]),
                   bhp=np.array([200.0 * BARSA, 150.0 * BARSA]))
        s1, _ = standalone_solve(st, model, 30.0 * DAY)
        s2, _ = NonLinearSolver().solve_timestep(model, st, 30.0 * DAY)
        assert np.allclose(s1["pressure"], s2["pressure"])
        print("ok test_matches_solve_timestep_directly")

    test_single_step_advances_and_reports_converged()
    test_matches_solve_timestep_directly()
    print("\nAll tests passed!")
