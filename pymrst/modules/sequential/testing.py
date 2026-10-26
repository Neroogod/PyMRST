"""Small sequential cases for the self-tests of `pymrst.modules.sequential`.

A 1D two-phase waterflood on a `GenericBlackOilModel` (water injected at a
fixed rate at the left, 0 bar at the right), its sequential split, and a
transport state after one pressure solve, so state functions and models can
be exercised without a full run.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def toy_case(n: int = 10, kr_exponent: float = 2.0) -> SimpleNamespace:
    """The fully implicit model, its sequential split, forces and a state."""
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.sequential.utils.get_sequential_model_from_fi import (
        get_sequential_model_from_fi,
    )
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.params.wells_and_bc import fluxside, pside
    from pymrst.solvers.init_res_sol import init_res_sol
    from pymrst.utils.units import BARSA, DARCY, DAY

    G = compute_geometry(cart_grid([n], [float(n)]))
    rock = make_rock(G, DARCY, 0.3)
    fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 2e-3], rho=[1000, 800],
                                  n=[kr_exponent, kr_exponent])
    model = GenericBlackOilModel(G, rock, fluid, gas=False)
    rate = np.sum(pore_volume(G, rock))/(10*DAY)
    bc = fluxside(None, G, "xmin", rate, sat=np.array([1.0, 0.0]))
    bc = pside(bc, G, "xmax", 0*BARSA, sat=np.array([0.0, 1.0]))
    schedule = simple_schedule(np.full(3, DAY), bc=bc)
    state0 = init_res_sol(G, 50*BARSA, np.array([0.2, 0.8]))
    seq = get_sequential_model_from_fi(model)
    forces = seq.get_driving_forces(schedule.controls[0])
    seq = seq.validate_model(forces)
    return SimpleNamespace(G=G, model=model, seq=seq, schedule=schedule, forces=forces,
                           state0=seq.validate_state(state0))


def toy_transport_state(case: SimpleNamespace, sw: float = 0.4) -> SimpleNamespace:
    """A transport model, and a state with a total flux from a pressure
    solve, saturations `sw` and AD caches initialised."""
    from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
    from pymrst.utils.units import DAY
    state, _, _ = case.seq.pressureNonLinearSolver.solve_timestep(
        case.state0, DAY, case.seq.pressureModel, **case.forces)
    state["s"] = np.column_stack([np.full(case.G.cells.num, sw),
                                  np.full(case.G.cells.num, 1 - sw)])
    tmodel = case.seq.transportModel
    ad_state, _, _ = tmodel.get_state_ad(copy_state(state))
    return SimpleNamespace(model=tmodel, parent=tmodel.parentModel, state=state,
                           ad_state=ad_state)


__all__ = ["toy_case", "toy_transport_state"]


if __name__ == "__main__":
    def test_the_case_builds_a_sequential_model():
        case = toy_case()
        assert type(case.seq).__name__ == "SequentialPressureTransportModel"
        t = toy_transport_state(case)
        assert np.asarray(t.state["flux"]).shape == (case.G.faces.num, 2)
        print("ok test_the_case_builds_a_sequential_model")

    test_the_case_builds_a_sequential_model()
    print("\nAll tests passed!")
