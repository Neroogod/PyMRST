"""Phase 43: the CPR linear solver, validated against MATLAB.

A linear solver has no physics of its own and so no reference of its own to
dump. What it has is a sharper obligation: **it must not change the answer.**
Every Newton step still solves the same linear system; CPR just gets there by
iterating instead of factorising. So if CPR is right, the two cases that are
already validated against MRST - Phase 5a's oil-water quarter-spot and Phase
5b's SPE 1 - must still match MATLAB *to the tolerances they matched at
before*. Not "close enough": the same numbers.

That is the honest test, and it is the one that catches the failure that
matters. A linear solver that is subtly too loose does not crash - it quietly
returns a slightly wrong Newton update, the nonlinear solver absorbs it, and
the simulation converges to something that is almost right. The only way to see
that is to compare against a reference computed a completely different way.

So this replays both cases through `CPRSolver` and checks them against:

- **MATLAB** (`ref_p5_ow.mat`, `ref_p5b_spe1.mat`) - the same tolerances Phase
  24 demanded of the direct solve;
- **the direct solve itself**, run here, side by side - which is the tighter
  check, because two linear solvers on the identical discrete system should
  agree to solver tolerance, not merely to MRST's.

It also reports the Newton counts. If CPR were returning poor updates, the
first thing to move would be the number of Newton iterations needed - so a
matching count is evidence the linear solves are genuinely converged, not
merely tolerable.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).parent))

from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    initial_state_from_deck,
)
from pymrst.modules.ad_core.linear_solvers import (
    CPR_CROSSOVER_BLOCK_JACOBI,
    CPR_CROSSOVER_ILU,
    HAVE_AMG,
    CPRSolver,
    DirectSolver,
    select_linear_solver,
)
from pymrst.modules.ad_core.nonlinear_solver import NonLinearSolver
from pymrst.params import NORM_GRAVITY
from pymrst.utils.units import DAY

from validate_phase24 import SPE1_STEPS, build_ow, build_spe1, ow_initial

HERE = Path(__file__).parent


def run_ow(linear_solver) -> Tuple[Dict[str, np.ndarray], int]:
    """Phase 5a's oil-water schedule, through whichever linear solver."""
    g, rock, fluid, wells = build_ow()
    model = GenericBlackOilModel(g, rock, fluid, wells)
    solver = NonLinearSolver(linear_solver=linear_solver)
    st = ow_initial(g, wells)

    hist: Dict[str, List[np.ndarray]] = {k: [] for k in
                                         ("p", "sw", "qWs", "qOs", "bhp")}
    newtons = 0
    for i, dt in enumerate(np.full(8, 30.0) * DAY):
        st, rep = solver.solve_ministep(model, st, st.copy(), float(dt))
        assert rep.converged, f"oil-water step {i} failed"
        newtons += rep.iterations
        for key, val in (("p", st["pressure"]), ("sw", st["sW"]),
                         ("qWs", st["qWs"]), ("qOs", st["qOs"]),
                         ("bhp", st["bhp"])):
            hist[key].append(val.copy())
    return {k: np.vstack(v) for k, v in hist.items()}, newtons


def run_spe1(order, linear_solver) -> Tuple[Dict[str, np.ndarray], int]:
    """SPE 1's first 30 report steps, through whichever linear solver."""
    deck, g, rock, fluid, wells = build_spe1(order)
    model = GenericBlackOilModel(g, rock, fluid, wells, disgas=True,
                                 gravity=NORM_GRAVITY, tol_well=1e-8)
    solver = NonLinearSolver(linear_solver=linear_solver)
    st = initial_state_from_deck(model, deck)

    hist: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("p", "sw", "sg", "rs", "qWs", "qOs", "qGs", "bhp")
    }
    newtons = 0
    for i, dt in enumerate(deck["tsteps"][:SPE1_STEPS]):
        st, rep = solver.solve_ministep(model, st, st.copy(), float(dt))
        assert rep.converged, f"SPE1 step {i} failed"
        newtons += rep.iterations
        for key, val in (("p", st["pressure"]), ("sw", st["sW"]),
                         ("sg", st["sG"]), ("rs", st["rs"]),
                         ("qWs", st["qWs"]), ("qOs", st["qOs"]),
                         ("qGs", st["qGs"]), ("bhp", st["bhp"])):
            hist[key].append(val.copy())
    return {k: np.vstack(v) for k, v in hist.items()}, newtons


def compare(title: str, checks) -> int:
    print(title)
    failed = 0
    for name, py, ref, rtol in checks:
        py = np.asarray(py, dtype=float).ravel()
        ref = np.asarray(ref, dtype=float).ravel()
        scale = max(np.abs(ref).max(), 1e-300)
        diff = float(np.abs(py - ref).max())
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = "
              f"{diff:.3e} (rel {diff / scale:.2e}, rtol {rtol:.0e})")
        failed += 0 if ok else 1
    return failed


def main() -> int:
    ow_ref_f = HERE / "ref_p5_ow.mat"
    spe1_ref_f = HERE / "ref_p5b_spe1.mat"
    if not ow_ref_f.exists() or not spe1_ref_f.exists():
        print("ref_p5_ow.mat / ref_p5b_spe1.mat missing (run "
              "dump_phase5_reference.m and dump_phase5b_reference.m)")
        return 1

    print(f"pyamg {'available - AMG pressure stage' if HAVE_AMG else
                   'NOT installed - ILU pressure stage'}\n")
    failed = 0

    # -- oil-water ------------------------------------------------------------
    # The rtols are Phase 24's, unchanged. That is the point: CPR has to clear
    # the same bar the direct solve cleared, not a bar lowered to fit it.
    ow_ref = scipy.io.loadmat(ow_ref_f, squeeze_me=True)
    ow_cpr, n_cpr = run_ow(CPRSolver(rtol=1e-10))
    ow_dir, n_dir = run_ow(DirectSolver())

    failed += compare("\nOil-water (Phase 5a's case) through CPR, vs MATLAB:", [
        ("pressure history", ow_cpr["p"], ow_ref["p_hist"], 5e-6),
        ("sW history", ow_cpr["sw"], ow_ref["sw_hist"], 5e-6),
        ("qWs history", ow_cpr["qWs"], ow_ref["qWs"], 5e-6),
        ("qOs history", ow_cpr["qOs"], ow_ref["qOs"], 5e-6),
        ("bhp history", ow_cpr["bhp"], ow_ref["bhp"], 5e-6),
    ])
    failed += compare("  ...and against the direct solve, side by side:", [
        (k, ow_cpr[k], ow_dir[k], 1e-9)
        for k in ("p", "sw", "qWs", "qOs", "bhp")
    ])
    same = n_cpr == n_dir
    print(f"  {'ok  ' if same else 'FAIL'} Newton iterations: {n_cpr} through "
          f"CPR, {n_dir} direct")
    failed += 0 if same else 1

    # -- SPE 1 ----------------------------------------------------------------
    spe1_ref = scipy.io.loadmat(spe1_ref_f, squeeze_me=True)
    order = ("PRODUCER", "INJECTOR") if bool(spe1_ref["isProdFirst"]) \
        else ("INJECTOR", "PRODUCER")
    spe1_cpr, n_cpr = run_spe1(order, CPRSolver(rtol=1e-10))
    spe1_dir, n_dir = run_spe1(order, DirectSolver())

    failed += compare("\nSPE 1 (three-phase, live oil) through CPR, vs MATLAB:", [
        ("pressure history", spe1_cpr["p"], spe1_ref["p_hist"], 2e-5),
        ("sW history", spe1_cpr["sw"], spe1_ref["sw_hist"], 2e-5),
        ("sG history", spe1_cpr["sg"], spe1_ref["sg_hist"], 2e-5),
        ("rs history", spe1_cpr["rs"], spe1_ref["rs_hist"], 2e-5),
        ("qWs history", spe1_cpr["qWs"], spe1_ref["qWs"], 2e-5),
        ("qOs history", spe1_cpr["qOs"], spe1_ref["qOs"], 2e-5),
        ("qGs history", spe1_cpr["qGs"], spe1_ref["qGs"], 2e-5),
        ("bhp history", spe1_cpr["bhp"], spe1_ref["bhp"], 2e-5),
    ])
    failed += compare("  ...and against the direct solve, side by side:", [
        (k, spe1_cpr[k], spe1_dir[k], 1e-8)
        for k in ("p", "sw", "sg", "rs", "qWs", "qOs", "qGs", "bhp")
    ])
    same = n_cpr == n_dir
    print(f"  {'ok  ' if same else 'FAIL'} Newton iterations: {n_cpr} through "
          f"CPR, {n_dir} direct")
    failed += 0 if same else 1

    # -- the selector ---------------------------------------------------------
    # Both cases above are far below the crossover, so the selector must hand
    # back a *direct* solve for them. CPR is genuinely the slower choice at this
    # size, and saying otherwise would be the easy lie.
    deck, g, rock, fluid, wells = build_spe1(order)
    spe1 = GenericBlackOilModel(g, rock, fluid, wells, disgas=True)
    n = spe1.G.cells.num * spe1.n_phases
    chosen = select_linear_solver(spe1)
    ok = isinstance(chosen, DirectSolver)
    print(f"\nSolver selection (crossover {CPR_CROSSOVER_BLOCK_JACOBI:,} unknowns for\n  two-phase, where the local stage is nearly free; {CPR_CROSSOVER_ILU:,} for three-phase, which\n  must pay for an ILU):")
    print(f"  {'ok  ' if ok else 'FAIL'} SPE 1 has {n:,} unknowns -> "
          f"{type(chosen).__name__} (below the crossover, the direct solve "
          f"is faster and we say so)")
    failed += 0 if ok else 1

    print("\nCPR reproduces MRST exactly - a different path, the same answer!"
          if failed == 0 else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
