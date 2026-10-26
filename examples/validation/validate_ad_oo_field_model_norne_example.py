"""Validate MRST fieldModelNorneExample: the port in
`ad_blackoil/examples/field_model_norne_example.py`.

References from MRST 2024a running the script's model and settings with
useMex false (MRST's MEX does not build here):

* The model check (`dump_field_model_norne_first_iteration_reference.m`):
  - the equilibrated, SWATINIT-scaled initial state, to `RTOL_INIT` (one or
    two ulps in the equilibrated pressure remain from the ODE integration);
  - the residuals of the first linearised system and the state after the
    first Newton iteration (direct solver), both from MRST's own initial
    state, to `RTOL_SYSTEM` and `RTOL_UPDATE`.
* The first two report steps with a direct solver
  (`dump_field_model_norne_example_reference.m`, `direct`), from MRST's
  initial state. Newton updates leave cells with sO < -1e-4 and no
  historical oil, so `RvMax`'s VAPPARS factor is negative and its 0.5 power
  complex in MATLAB; MRST iterates on with a complex `rv` and Jacobian. The
  port takes the real part of MATLAB's power (see `rv_max.py`). In report
  step 1 that happens in one cell and leaves MRST's Newton path intact:
  iterations and ministeps identical, pressures and bhps to `RTOL_DIRECT`.
  In step 2 it happens in one cell after the first iteration and six after
  the second, and from the third iteration MATLAB's complex Jacobian gives
  a different real update (CNV 0.432 vs 0.442, then growing): the ministeps
  are identical, the iterations are reported (8 vs MRST's 9), and pressures
  and bhps are required to `RTOL_COMPLEX`. `rs` and `rv` in the few cells
  near the bubble and dew points differ at the solver's CNV 1e-2 level and
  are reported.
* The example as it runs (`iterative`): the first five report steps with
  CPR/GMRES-ILU, whose GMRES stops at 1e-4 and so carries the two
  implementations' rounding as well. The well curves are required within
  `RTOL_WELLS` of the largest rate or bhp; the cell states are reported.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
RTOL_INIT = 1e-12
RTOL_SYSTEM = 1e-10
RTOL_UPDATE = 1e-9
RTOL_WELLS = 1e-4
RTOL_DIRECT = 1e-5
RTOL_COMPLEX = 1e-4
ITERATIVE_STEPS = 5


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def model_check() -> bool:
    from pymrst.modules.ad_blackoil.examples.field_model_norne_example import (
        setup_field_model_norne_example,
    )
    from pymrst.modules.ad_core.ad_oo.physical_model import copy_state
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    from pymrst.utils.adi import value

    ref = loadmat(HERE/"ref_field_model_norne_first_iteration.mat", squeeze_me=True)
    ok = True
    out = setup_field_model_norne_example(1)
    for key, rkey in (("pressure", "p0"), ("s", "s0"), ("rs", "rs0"), ("rv", "rv0"),
                      ("pcowScale", "pcowScale")):
        err = _rel(out.state0[key], ref[rkey])
        ok &= report(err < RTOL_INIT, f"initial {key} rel. diff {err:.1e}")
    model, nls = out.model, out.nls
    forces = model.get_driving_forces(out.schedule.controls[0])
    dt = float(out.schedule.step_val[0])
    s0 = copy_state(out.state0)
    for key, rkey in (("pressure", "p0"), ("s", "s0"), ("rs", "rs0"), ("rv", "rv0"),
                      ("pcowScale", "pcowScale")):
        s0[key] = np.array(ref[rkey], dtype=float)
    model = model.validate_model(forces)
    state = model.validate_state(s0)
    start = copy_state(state)
    model, state = model.prepare_report_step(state, start, dt, forces)
    model, state = model.prepare_timestep(state, start, dt, forces)
    problem, _ = model.get_equations(start, copy_state(state), dt, forces, iteration=1)
    names = [str(n) for n in np.atleast_1d(ref["eqnames"])]
    for name, eq in zip(problem.equation_names, problem.equations):
        expected = np.atleast_1d(np.asarray(ref["res"][names.index(name)], dtype=float))
        err = _rel(np.asarray(value(eq)).ravel(), expected.ravel())
        ok &= report(err < RTOL_SYSTEM, f"first system: {name} rel. diff {err:.1e}")
    state1, _ = model.step_function(state, start, dt, forces, BackslashSolverAD(), nls, 1)
    for key, rkey in (("pressure", "p1"), ("s", "s1"), ("rs", "rs1"), ("rv", "rv1")):
        err = _rel(value(state1[key]), ref[rkey])
        ok &= report(err < RTOL_UPDATE, f"first Newton update: {key} rel. diff {err:.1e}")
    err = _rel([w.bhp for w in state1["wellSol"]], ref["bhp1"])
    ok &= report(err < RTOL_UPDATE, f"first Newton update: bhp rel. diff {err:.1e}")
    return ok


def direct_run() -> bool:
    from pymrst.modules.ad_blackoil.examples.field_model_norne_example import (
        setup_field_model_norne_example,
    )
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD

    ref = loadmat(HERE/"ref_field_model_norne_first_iteration.mat", squeeze_me=True)
    direct = loadmat(HERE/"ref_field_model_norne_example.mat", squeeze_me=True,
                     struct_as_record=False)["direct"]
    n = int(np.size(direct.its))
    out = setup_field_model_norne_example(n)
    for key, rkey in (("pressure", "p0"), ("s", "s0"), ("rs", "rs0"), ("rv", "rv0"),
                      ("pcowScale", "pcowScale")):
        out.state0[key] = np.array(ref[rkey], dtype=float)
    out.nls.LinearSolver = BackslashSolverAD()
    ws, states, rep = simulate_schedule_ad(out.state0, out.model, out.schedule,
                                           NonLinearSolver=out.nls)
    reports = rep["ControlstepReports"]
    its = [int(r["Iterations"]) for r in reports]
    nmini = [len(r["StepReports"]) for r in reports]
    its_ref = np.atleast_1d(direct.its).astype(int).tolist()
    nmini_ref = np.atleast_1d(direct.nmini).astype(int).tolist()
    ok = report(nmini == nmini_ref, f"ministeps {nmini} (MRST {nmini_ref})")
    ok &= report(its[0] == its_ref[0], f"step 1: Newton iterations {its[0]} (MRST {its_ref[0]})")
    print(f"info step 2: Newton iterations {its[1]} (MRST {its_ref[1]})")
    for k, rtol in ((0, RTOL_DIRECT), (1, RTOL_COMPLEX)):
        err = _rel(states[k]["pressure"], direct.p[k])
        ok &= report(err < rtol, f"step {k + 1}: pressure rel. diff {err:.1e}")
        err = _rel([w.bhp for w in ws[k]], direct.bhp[k])
        ok &= report(err < rtol, f"step {k + 1}: well bhp rel. diff {err:.1e}")
    for key in ("rs", "rv"):
        print(f"info {key} rel. diff {_rel([s[key] for s in states], getattr(direct, key)):.1e}")
    return ok


def as_run() -> bool:
    from pymrst.modules.ad_blackoil.examples.field_model_norne_example import (
        field_model_norne_example,
    )
    ref = loadmat(HERE/"ref_field_model_norne_example.mat", squeeze_me=True,
                  struct_as_record=False)
    it = ref["iterative"]
    ex = field_model_norne_example(plot=False, nsteps=ITERATIVE_STEPS)
    ok = report(type(ex.nls.LinearSolver).__name__ == str(ref["lsolver"]),
                f"linear solver {type(ex.nls.LinearSolver).__name__}")
    ws = ex.wellSols
    rates = {k: np.array([[getattr(w, k) for w in row] for row in ws]) for k in
             ("qWs", "qOs", "qGs")}
    bhp = np.array([[w.bhp for w in row] for row in ws])
    err = _rel(bhp, it.bhp)
    ok &= report(err < RTOL_WELLS, f"well bhp rel. diff {err:.1e}")
    for k, v in rates.items():
        scale = np.abs(np.asarray(getattr(it, k), dtype=float)).max()
        err = float(np.abs(v - np.asarray(getattr(it, k), dtype=float)).max()/scale)
        ok &= report(err < RTOL_WELLS, f"well {k} rel. diff {err:.1e}")
    for key in ("p", "sw", "sg", "rs", "rv"):
        py = np.array([np.asarray(s["pressure" if key == "p" else key]) if key in ("p", "rs", "rv")
                       else np.asarray(s["s"])[:, 0 if key == "sw" else 2] for s in ex.states])
        print(f"info {key}: max abs diff {np.abs(py - np.asarray(getattr(it, key))).max():.2e}")
    return ok


def main() -> int:
    with isolated_gravity():
        print("Model check (initial state, first system, first Newton update):")
        ok = model_check()
        print("\nFirst two report steps, direct linear solver, from MRST's initial state:")
        ok &= direct_run()
        print(f"\nThe example as it runs, first {ITERATIVE_STEPS} report steps (CPR, GMRES-ILU):")
        ok &= as_run()
    print("\nfieldModelNorneExample matches MRST" if ok
          else "\nfieldModelNorneExample does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
