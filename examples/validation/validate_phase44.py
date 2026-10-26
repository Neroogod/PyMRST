"""Phase 44: adaptive timestep selection, against MATLAB ad-core.

A timestep selector is a *controller*, and the honest way to test a controller
is to drive it with a known input and record what it does. Running a simulation
and checking the answer would not do it: the physics is already validated, and
a selector that is subtly wrong does not produce a wrong answer - it produces
the *right* answer inefficiently, or it oscillates, and neither shows up in the
states. What it changes is the sequence of timesteps, so that is what to
compare.

`dump_phase44_reference.m` drives MRST's selectors with a fixed synthetic
history - "this step took 3 Newton iterations, this one took 12" - and records
every timestep they pick. This replays exactly the same drive through PyMRST's.
The control law is then pinned bit for bit, independently of any model.

Four cases:

- **A.** `IterationCountTimeStepSelector`, everything converging. This is the
  two-point branch, the one that does the real work and the one easiest to get
  wrong: it is a second-order controller, and it uses `dt` and iteration counts
  from *both* of the last two steps.
- **B.** The same with failed steps salted in, which forces the one-point
  `restart` branch. A controller must not learn from a step it did not choose -
  a step that was cut by the physics carries no information about the
  controller's own response - and this is where that logic lives.
- **C.** `StateChangeTimeStepSelector`, bounding the relative change in
  pressure. Cao's rule.
- **D.** `rampup_timesteps`.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_core.simulators.nonlinear_solver import StepReport
from pymrst.modules.ad_core.timesteps import (
    IterationCountTimeStepSelectorAD as IterationCountTimeStepSelector,
    StateChangeTimeStepSelectorAD as StateChangeTimeStepSelector,
    rampup_timesteps,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.units import BARSA, DAY

HERE = Path(__file__).parent
REF = HERE / "ref_p44_timesteps.mat"

N_CELLS = 9          # the 3x3x1 grid the reference used
MAX_ITERATIONS = 12  # NonLinearSolver('maxIterations', 12)


class _Solver:
    """The selectors ask a solver for exactly one thing."""
    max_iterations = MAX_ITERATIONS


def _report(iterations, dt, converged=True) -> StepReport:
    return StepReport(converged=bool(converged), iterations=int(iterations),
                      dt=float(dt))


def case_a(its) -> np.ndarray:
    """Iteration count, every step converging."""
    sel = IterationCountTimeStepSelector(
        target_iteration_count=5, iteration_offset=5,
        max_timestep=100 * DAY, min_timestep=0.01 * DAY)
    sel.new_control_step(1)
    dt, dt_prev, out = 10 * DAY, None, []
    for n in its:
        dt = sel.pick_timestep(dt_prev, dt, None, _Solver(), None, None)
        out.append(dt)
        sel.store_timestep(_report(n, dt))
        dt_prev = dt
    return np.array(out)


def case_b(its, converged) -> np.ndarray:
    """Iteration count, with failures - the `restart` branch."""
    sel = IterationCountTimeStepSelector(
        target_iteration_count=5, iteration_offset=5,
        max_timestep=100 * DAY, min_timestep=0.01 * DAY)
    sel.new_control_step(1)
    dt, dt_prev, out = 10 * DAY, None, []
    for k, n in enumerate(its):
        pick = (sel.cut_timestep if k > 0 and not converged[k - 1]
                else sel.pick_timestep)
        dt = pick(dt_prev, dt, None, _Solver(), None, None)
        out.append(dt)
        sel.store_timestep(_report(n, dt, converged[k]))
        if converged[k]:
            dt_prev = dt
    return np.array(out)


def case_c(moves) -> np.ndarray:
    """State change, bounding the relative movement of the pressure."""
    sel = StateChangeTimeStepSelector(
        target_props=["pressure"], target_change_rel=[0.05], relax_factor=1.0,
        target_iteration_count=5, iteration_offset=5,
        max_timestep=100 * DAY, min_timestep=0.01 * DAY)
    sel.new_control_step(1)
    dt, dt_prev, out = 10 * DAY, None, []
    p0 = np.full(N_CELLS, 200.0 * BARSA)
    for m in moves:
        prev = State(pressure=p0)
        curr = State(pressure=p0 * (1.0 + m))
        dt = sel.pick_timestep(dt_prev, dt, None, _Solver(), prev, curr)
        out.append(dt)
        sel.store_timestep(_report(5, dt))
        dt_prev = dt
    return np.array(out)


def compare(name, py, ml, rtol=1e-12) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    if py.shape != ml.shape:
        print(f"  FAIL {name}: {py.size} steps, MATLAB has {ml.size}")
        return 1
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p44_timesteps.mat missing (run dump_phase44_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    failed = 0

    its = np.atleast_1d(ref["its_A"]).astype(int)
    conv = np.atleast_1d(ref["conv_B"]).astype(bool)
    moves = np.atleast_1d(ref["moves"]).astype(float)

    a = case_a(its)
    b = case_b(its, conv)
    c = case_c(moves)

    print("A. IterationCountTimeStepSelector, all steps converging:")
    failed += compare("timesteps picked", a, ref["A_dt"])
    print(f"     Newton its fed in : {[int(i) for i in its]}")
    print(f"     dt chosen (days)  : {[round(float(x) / DAY, 3) for x in a]}")

    print("\nB. ...with failed steps (the one-point `restart` branch):")
    failed += compare("timesteps picked", b, ref["B_dt"])

    print("\nC. StateChangeTimeStepSelector (relative pressure change, "
          "target 5%):")
    failed += compare("timesteps picked", c, ref["C_dt"])
    print(f"     pressure moves    : {[f'{m:.1%}' for m in moves]}")
    print(f"     dt chosen (days)  : {[round(float(x) / DAY, 3) for x in c]}")

    print("\nD. rampup_timesteps:")
    for tag, time, dt, n in (("D_year", 365 * DAY, 30 * DAY, 8),
                             ("D_short", 100 * DAY, 30 * DAY, 5),
                             ("D_tiny", 10 * DAY, 30 * DAY, 8)):
        failed += compare(f"{tag}", rampup_timesteps(time, dt, n), ref[tag])

    # The check that no reference can make for us: the controller must actually
    # *do* something, and in the right direction. If it simply held dt constant
    # it would still match MATLAB - MATLAB would have held the same constant -
    # and every comparison above would pass while the selector was useless.
    #
    # Mind the indexing, which caught me out: `a[k]` is the dt chosen *for*
    # step k, and `its[k]` is what that step then cost. The controller cannot
    # respond to a step until it has taken it, so the response to `its[k]` is
    # `a[k + 1]`.
    easy, hard = int(np.argmin(its)), int(np.argmax(its))
    varied = float(a.max() / a.min())
    responds = a[easy + 1] > a[hard + 1]
    print(f"\n  {'ok  ' if varied > 2 else 'FAIL'} the controller moves: dt "
          f"spans {varied:.1f}x across the run")
    print(f"  {'ok  ' if responds else 'FAIL'} and in the right direction: "
          f"after the easiest step ({its[easy]} its) dt grew to "
          f"{a[easy + 1] / DAY:.2f} d; after the hardest ({its[hard]} its) it "
          f"was cut to {a[hard + 1] / DAY:.2f} d")
    failed += (0 if varied > 2 else 1) + (0 if responds else 1)

    print("\nTimestep selection matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
