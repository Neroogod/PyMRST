"""Schedule simulator (MRST ``simulateScheduleAD.m`` layout)."""

from typing import List, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import OWState, TwoPhaseOilWaterModelAD


def simulate_schedule_ad(
    state0: OWState,
    model: TwoPhaseOilWaterModelAD,
    timesteps: Sequence[float],
    max_cuts: int = 6,
    verbose: bool = False,
) -> Tuple[List[OWState], np.ndarray]:
    """Advance through report steps, cutting failed nonlinear substeps."""
    state = state0
    states: List[OWState] = []
    its_per_step = np.zeros(len(timesteps), dtype=np.int64)

    for i, dt_report in enumerate(timesteps):
        t_local, its = 0.0, 0
        dt, cuts = float(dt_report), 0
        while t_local < dt_report - 1e-12 * dt_report:
            dt = min(dt, dt_report - t_local)
            candidate, ok, nit = model.step(state, dt)
            its += nit
            if ok:
                state = candidate
                t_local += dt
                dt = min(2.0 * dt, dt_report - t_local) if cuts else dt
            else:
                cuts += 1
                assert cuts <= max_cuts, (
                    f"step {i}: Newton failed after {max_cuts} timestep cuts"
                )
                dt /= 2.0
                if verbose:
                    print(f"  step {i}: cutting dt to {dt:.3e}")
        its_per_step[i] = its
        states.append(state)
        if verbose:
            print(f"step {i + 1}/{len(timesteps)}: {its} iterations")
    return states, its_per_step


__all__ = ["simulate_schedule_ad"]
