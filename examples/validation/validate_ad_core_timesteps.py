"""Validate MRST's time-step selectors and `rampupTimesteps`: the ports in
`ad_core/timesteps/` (one file each).

The reference is MRST 2024a (`dump_ad_core_timesteps_reference.m`): the
Simple (ramp-up, hard limit), IterationCount, StateChange (on the water
saturation of a black-oil model's states) and Factor (reset on control
change) selectors, each driven through the same eight ministeps - control
changes and repeated controls, a failed step followed by a cut, step reports
of varying iteration counts - and `rampupTimesteps` in three cases. Every
time step chosen is compared.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-14


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def main() -> int:
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_core.timesteps.factor_time_step_selector import FactorTimeStepSelector
    from pymrst.modules.ad_core.timesteps.iteration_count_time_step_selector import (
        IterationCountTimeStepSelector,
    )
    from pymrst.modules.ad_core.timesteps.rampup_timesteps import rampup_timesteps
    from pymrst.modules.ad_core.timesteps.simple_time_step_selector import SimpleTimeStepSelector
    from pymrst.modules.ad_core.timesteps.state_change_time_step_selector import (
        StateChangeTimeStepSelector,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.utils.units import CENTI, DAY, POISE

    ref = loadmat(HERE/"ref_ad_core_timesteps.mat", squeeze_me=True)
    G = compute_geometry(cart_grid([3, 1, 1]))
    fluid = init_simple_adi_fluid(phases="WO", mu=np.array([1, 1])*CENTI*POISE, rho=[1000, 800],
                                  n=[2, 2])
    model = GenericBlackOilModel(G, make_rock(G, 1e-13, 0.3), fluid, gas=False).validate_model()

    class Solver:
        maxIterations = 25

    its, conv, ctrl, sw = ref["its"], ref["conv"].astype(bool), ref["ctrl"], ref["sw"]
    selectors = [
        ("SimpleTimeStepSelector", SimpleTimeStepSelector(firstRampupStepRelative=0.1,
                                                          firstRampupStep=2*DAY, maxTimestep=40*DAY)),
        ("IterationCountTimeStepSelector", IterationCountTimeStepSelector(
            targetIterationCount=8, firstRampupStepRelative=0.1)),
        ("StateChangeTimeStepSelector", StateChangeTimeStepSelector(
            targetProps=["s"], targetChangeAbs=[0.2], targetIterationCount=np.inf)),
        ("FactorTimeStepSelector", FactorTimeStepSelector(maxRelativeAdjustment=1.5,
                                                          resetOnControlsChanged=True)),
    ]
    ok = True
    for k, (name, sel) in enumerate(selectors):
        dt_prev, state_prev, dt = None, None, 30*DAY
        dts = []
        for i in range(its.size):
            if i == 0 or ctrl[i] != ctrl[i - 1] or (i + 1) % 3 == 0:
                sel.new_control_step({"controlId": int(ctrl[i])})
            state = {"s": np.column_stack([np.full(3, sw[i]), np.full(3, 1 - sw[i])]),
                     "pressure": np.full(3, 1e7)}
            if i > 0 and not conv[i - 1]:
                dt = sel.cut_timestep(dt_prev, dt, model, Solver, state_prev, state)
            else:
                dt = sel.pick_timestep(dt_prev, dt, model, Solver, state_prev, state)
            dts.append(dt)
            sel.store_timestep({"Converged": bool(conv[i]), "Iterations": int(its[i]), "Timestep": dt})
            if conv[i]:
                dt_prev, state_prev = dt, state
        err = np.abs(np.array(dts) - ref["dts"][k]).max()/np.abs(ref["dts"][k]).max()
        ok &= report(err < RTOL, f"{name}: the eight time steps chosen, rel. diff {err:.1e}")
    for tag, args, key in (("100 days in 10-day steps", (100*DAY, 10*DAY), "ramp1"),
                           ("3 days, ramp of 4", (3*DAY, 10*DAY, 4), "ramp2"),
                           ("105 days, ramp of 3", (105*DAY, 10*DAY, 3), "ramp3")):
        dT = rampup_timesteps(*args)
        expected = np.atleast_1d(ref[key])
        same = dT.shape == expected.shape and np.allclose(dT, expected, rtol=RTOL, atol=0)
        ok &= report(same, f"rampupTimesteps, {tag}: {dT.size} steps")
    print("\nThe time-step selectors match MRST" if ok
          else "\nThe time-step selectors do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
