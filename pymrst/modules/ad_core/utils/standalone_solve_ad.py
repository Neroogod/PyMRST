"""MRST ``ad-core/utils/standaloneSolveAD.m``: one time step of `dt` for a
model, the driving forces given by name (`W`, `bc`, `src`, ...), anything
else passed on to `simulateScheduleAD`. Returns the new state and the
report."""

from typing import Any, Tuple


def standalone_solve_ad(state0: Any, model: Any, dt: float, **kwargs: Any) -> Tuple[Any, Any]:
    """MRST `[state, report] = standaloneSolveAD(state0, model, dt, ...)`."""
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    forces = model.get_valid_driving_forces()
    extra = {}
    for name, val in kwargs.items():
        match = next((k for k in forces if k.lower() == name.lower()), None)
        if match is None:
            extra[name] = val
        else:
            forces[match] = val
    schedule = simple_schedule(np.atleast_1d(float(dt)))
    for name, val in forces.items():
        setattr(schedule.controls[0], name, val)
    _, states, report = simulate_schedule_ad(state0, model, schedule, **extra)
    return states[0], report


standaloneSolveAD = standalone_solve_ad
__all__ = ["standalone_solve_ad", "standaloneSolveAD"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_source
    from pymrst.utils.units import BARSA, DAY

    def test_one_step_with_a_source():
        G = compute_geometry(cart_grid([3, 1, 1]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800],
                                      c=[1e-9, 1e-9])
        model = GenericBlackOilModel(G, make_rock(G, 1e-13, 0.3), fluid, gas=False)
        state0 = {"pressure": np.full(3, 100*BARSA), "s": np.tile([0.0, 1.0], (3, 1))}
        src = add_source(None, np.array([0]), 1e-6, sat=np.array([1.0, 0.0]))
        state, report = standalone_solve_ad(state0, model, DAY, src=src)
        assert state["s"][0, 0] > 0 and state["pressure"][0] > 100*BARSA
        assert len(report["ControlstepReports"]) == 1
        print("ok test_one_step_with_a_source")

    test_one_step_with_a_source()
    print("\nAll tests passed!")
