"""Phase 356: structured schedules and time-varying controls (AC20).

Until now a schedule in this port was a bare list of timesteps, and the wells
were fixed for the whole run. AC20 adds MRST's structured schedule - timesteps
grouped into control periods, each with its own wells / bc / sources - and wires
the solver to run it, so a well can change its control partway through.

There is no MATLAB reference here, and deliberately so:

- the **editing operations** (`simple_schedule`, `compress_schedule`,
  `refine_schedule`, `combine_schedules`, `subset_schedule`) are validated by
  *invariants*, which is what they are - total time is preserved, `compress`
  leaves one step per control period, `refine` never lengthens a step, and
  `compress(refine(s))` recovers the control history of `s`. These live in the
  module's own self-tests; a representative round-trip is repeated here.
- the **coupling into the solver** is validated by *behaviour*: a schedule whose
  producer tightens its bottom-hole pressure halfway through must actually
  produce that tighter bhp - and, because a harder drawdown lifts more oil, must
  produce more oil after the change than before. If the control were not being
  applied, the bhp would never move.

`getInitializationRegionsBlackOil`-style deck schedule parsing
(`convert_deck_schedule_to_mrst_ad`) is out of this phase: it needs a general deck
SCHEDULE reader and an SPE9 deck, neither of which is present, and is noted as
deferred in `ad-core_implementation.md`.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_core.utils.schedule import (
    Control,
    Schedule,
    compress_schedule,
    refine_schedule_ad as refine_schedule,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def _check(name: str, ok: bool) -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if ok else 1


def _wells(g, rock, prod_bhp):
    inj = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                   compi=np.array([1.0, 0.0]), sign=1, name="I")[0]
    prod = add_well(None, g, rock, np.array([g.cells.num - 1]),
                    well_type="bhp", val=prod_bhp, compi=np.array([0.0, 1.0]),
                    sign=-1, name="P")[0]
    return [inj, prod]


def main() -> int:
    failed = 0

    # -- the round-trip invariant, repeated as a headline check ---------------
    s = Schedule(step_val=np.array([10., 10., 10., 10., 20., 20., 20.]) * DAY,
                 step_control=np.array([0, 0, 0, 0, 1, 1, 1]),
                 controls=[Control(W=["a"]), Control(W=["b"])])
    base = compress_schedule(s)
    back = compress_schedule(refine_schedule(s, 3.0 * DAY))
    print("Schedule editing invariants:")
    failed += _check("compress leaves one step per control period",
                     base.n_steps == 2 and list(base.step_control) == [0, 1])
    failed += _check("total time preserved through refine + compress",
                     np.isclose(back.total_time, s.total_time))
    failed += _check("compress(refine(s)) recovers the control periods",
                     list(back.step_control) == list(base.step_control)
                     and np.allclose(back.step_val, base.step_val))

    # -- the coupling: a mid-run control change must take effect ---------------
    g = compute_geometry(cart_grid([8, 8, 1], [400.0, 400.0, 20.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 3.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)

    model = GenericBlackOilModelAD(g, rock, fluid, _wells(g, rock, 150 * BARSA),
                                 tol_well=1e-8)
    st = State(pressure=np.full(g.cells.num, 200.0 * BARSA),
               sW=np.full(g.cells.num, 0.2),
               qWs=np.array([2e-3, 0.0]), qOs=np.array([0.0, -1e-3]),
               bhp=np.array([200.0 * BARSA, 150.0 * BARSA]))

    # control 0: produce at 150 bar; control 1: tighten to 120 bar at step 2
    schedule = Schedule(
        step_val=np.array([30., 30., 30., 30.]) * DAY,
        step_control=np.array([0, 0, 1, 1]),
        controls=[Control(W=_wells(g, rock, 150 * BARSA)),
                  Control(W=_wells(g, rock, 120 * BARSA))],
    )
    states, rep = NonLinearSolver().simulate_schedule(model, st, schedule)

    print("\nTime-varying control (producer bhp 150 -> 120 bar at step 2):")
    failed += _check("schedule runs to completion", rep.converged
                     and len(states) == 4)
    bhp = [float(s["bhp"][1]) / BARSA for s in states]
    failed += _check(f"bhp holds 150 before the change ({bhp[0]:.0f}, {bhp[1]:.0f})",
                     abs(bhp[0] - 150) < 1e-6 and abs(bhp[1] - 150) < 1e-6)
    failed += _check(f"bhp switches to 120 after it ({bhp[2]:.0f}, {bhp[3]:.0f})",
                     abs(bhp[2] - 120) < 1e-6 and abs(bhp[3] - 120) < 1e-6)
    q_before = abs(float(states[1]["qOs"][1]))
    q_after = abs(float(states[2]["qOs"][1]))
    failed += _check(f"the tighter bhp lifts more oil "
                     f"({q_before:.3e} -> {q_after:.3e})", q_after > q_before)

    print("\nStructured schedules and time-varying controls work!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
