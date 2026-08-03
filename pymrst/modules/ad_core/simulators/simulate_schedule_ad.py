"""Schedule simulation driver.

Ports MRST's ``ad-core/simulators/simulateScheduleAD.m``.

Runs a sequence of report steps through a model's backward-Euler `step`,
halving substeps on Newton failure (up to max_cuts, as MRST's
NonLinearSolver) and returning the state after every report step.
"""

from typing import List, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import OWState, TwoPhaseOilWaterModelAD


def simulate_schedule_ad(
    state0: OWState,
    model: TwoPhaseOilWaterModelAD,
    timesteps: Sequence[float],
    max_cuts: int = 6,
    verbose: bool = False,
    *,
    observer=None,
) -> Tuple[List[OWState], np.ndarray]:
    """Advance through the report steps; returns (states, iterations).

    `observer` is an optional, dependency-free observation hook, matching the
    one `NonLinearSolver` carries. It receives copied scalar metadata only and
    cannot participate in any decision this loop makes; its exceptions are
    swallowed so a reporting fault never becomes a simulation fault.

    It exists because this driver keeps its own Newton loop rather than
    delegating to `NonLinearSolver`, so the canonical hook does not reach it.
    Event names match the canonical solver's, so a handler written for one
    reads the other without special cases. The default is `None`, and with it
    this function behaves exactly as before.
    """
    def _observe(event, **payload):
        if observer is None:
            return
        try:
            observer(event, payload)
        except Exception:
            pass

    state = state0
    states: List[OWState] = []
    its_per_step = np.zeros(len(timesteps), dtype=np.int64)

    for i, dt_report in enumerate(timesteps):
        _observe("report_step_started", report_step=int(i), dt=float(dt_report))
        t_local, its = 0.0, 0
        dt, cuts = float(dt_report), 0
        while t_local < dt_report - 1e-12 * dt_report:
            dt = min(dt, dt_report - t_local)
            _observe("ministep_started", report_step=int(i), dt=float(dt))
            candidate, ok, nit = model.step(state, dt)
            its += nit
            _observe("step_accepted" if ok else "step_rejected",
                     report_step=int(i), dt=float(dt), iterations=int(nit))
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
                _observe("timestep_cut", report_step=int(i), cut=int(cuts),
                         dt=float(dt))
                if verbose:
                    print(f"  step {i}: cutting dt to {dt:.3e}")
        its_per_step[i] = its
        states.append(state)
        if verbose:
            print(f"step {i + 1}/{len(timesteps)}: {its} iterations")
    return states, its_per_step


__all__ = ["simulate_schedule_ad"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def test_schedule_runs_and_conserves():
        """An 6-step waterflood runs; produced+injected volumes reconcile."""
        g = compute_geometry(cart_grid([10, 10, 1], [200.0, 200.0, 10.0]))
        rock = make_rock(g, 200 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=2e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([99]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = TwoPhaseOilWaterModelAD(g, rock, fluid, w)
        st0 = model.initial_state(200 * BARSA, 0.1)

        dts = np.full(6, 20.0) * DAY
        states, its = simulate_schedule_ad(st0, model, dts)
        assert len(states) == 6 and np.all(its > 0)

        f, pv = fluid, model.pv
        w_of = lambda s: np.sum(pv * np.exp(1e-6 / BARSA *
                                            (s.pressure - 200 * BARSA)) * s.sW)
        injected = sum(s.qWs[0] for s in states) * 20 * DAY
        produced = -sum(s.qWs[1] for s in states) * 20 * DAY
        assert np.isclose(w_of(states[-1]) - w_of(st0) + produced, injected,
                          rtol=1e-6)
        assert states[-1].qOs[1] < 0, "producer takes oil"
        assert np.all(np.diff([s.sW[55] for s in states]) >= -1e-12)
        print("ok test_schedule_runs_and_conserves")

    test_schedule_runs_and_conserves()
    print("\nAll tests passed!")
