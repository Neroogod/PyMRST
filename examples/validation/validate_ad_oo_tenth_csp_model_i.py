"""Validate MRST tenthCSP_Model_I: the port in
`ad_blackoil/examples/spe10/tenth_csp_model_i.py`.

The reference is MRST 2024a running the script without its figures
(`dump_tenth_csp_model_i_reference.m`). Compared:
- the setup: the Model I rock, the SGOF relative permeabilities as the
  model evaluates them, the rock compressibility multiplier, the two wells'
  cells and well indices, and the equilibrated initial state - to
  `EXACT_RTOL`;
- report steps 1-2: Newton iterations identical, states to `EXACT_STEP_RTOL`;
- report step 3 from MRST's own step-2 state, to `RESTART_RTOL`. The two
  runs part in step 3 at the last bit, for a reason found in the linear
  solve: in the oil-only cells the gas equation's residual is exactly zero,
  and the gas-saturation increment the direct solver returns there is
  round-off, +-1e-15. MATLAB's sign of it caps `sG` to exactly 0; SciPy's
  can leave 1.4e-15, which puts `sO` just below 1, on the other side of the
  SGOF table's last knot - a different slope of `krO`, so a different
  Jacobian in a thousand cells. Fluids and rock are nearly incompressible,
  so the pressure level moves by up to 1 Pa while the residuals agree to
  six digits. From MRST's step-2 state (its exact zeros) the port's step 3
  is MRST's;
- the run, all 131 report steps: ministeps and well controls identical at
  every step, the Newton iterations reported (they differ by one in a few
  steps), pressures, gas saturations and well curves to `RUN_RTOL`, the
  level that knife edge leaves.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
EXACT_RTOL = 1e-14
EXACT_STEP_RTOL = 1e-12
RESTART_RTOL = 1e-9
RUN_RTOL = 1e-4


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def restart_step_3(out, ref) -> bool:
    """Step 3 from MRST's step-2 state: pressure, saturations and well
    unknowns as MRST left them."""
    from copy import deepcopy

    from pymrst.modules.ad_blackoil.examples.spe10.tenth_csp_model_i import TIMESTEP_DAYS
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.utils.units import DAY

    state = deepcopy(out.states[1])
    state["pressure"] = np.array(ref["p"][1], dtype=float)
    state["s"] = np.array(ref["s2"], dtype=float)
    for i, w in enumerate(state["wellSol"]):
        w.bhp, w.qOs, w.qGs = (float(ref[k][1, i]) for k in ("bhp", "qOs", "qGs"))
    schedule = deepcopy(out.schedule)
    schedule.step_val = np.array([TIMESTEP_DAYS[2]*DAY])
    schedule.step_control = np.array([0])
    with isolated_gravity():
        ws, states, rep = simulate_schedule_ad(state, out.model, schedule)
    ok = report(rep["ControlstepReports"][0]["Iterations"] == ref["its"][2],
                "step 3 from MRST's step-2 state: Newton iterations identical")
    for name, actual, expected in (("pressure", states[0]["pressure"], ref["p"][2]),
                                   ("gas saturation", np.asarray(states[0]["s"])[:, 1], ref["sg"][2]),
                                   ("bhp", [w.bhp for w in ws[0]], ref["bhp"][2])):
        err = _rel(actual, expected)
        ok &= report(err < RESTART_RTOL, f"step 3 from MRST's step-2 state: {name} rel. diff {err:.1e}")
    return ok


def main() -> int:
    from pymrst.modules.ad_blackoil.examples.spe10.tenth_csp_model_i import tenth_csp_model_i
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA

    ref = loadmat(HERE/"ref_tenth_csp_model_i.mat", squeeze_me=True)
    with isolated_gravity():
        out = tenth_csp_model_i(plot=False)
    ok = True
    s = np.linspace(0, 1, 101)
    f = out.fluid
    for name, actual, expected in (
            ("perm", out.rock.perm, ref["perm"]), ("poro", out.rock.poro, ref["poro"]),
            ("krG", value(f.krG(s)), ref["krG"]), ("krO", value(f.krO(s)), ref["krO"]),
            ("pvMultR", value(f.pvMultR(np.arange(50, 201, 10)*BARSA)), ref["pvMultR"]),
            ("WI injector", out.W[0].WI, ref["WI_I"]), ("WI producer", out.W[1].WI, ref["WI_P"]),
            ("initial pressure", out.state0["pressure"], ref["p0"]),
            ("initial saturations", out.state0["s"], ref["s0"])):
        err = _rel(np.reshape(actual, np.shape(expected)), expected)
        ok &= report(err < EXACT_RTOL, f"{name} rel. diff {err:.1e}")
    cells = [np.asarray(w.cells).ravel() + 1 for w in out.W]
    ok &= report(np.array_equal(cells[0], ref["cells_I"]) and np.array_equal(cells[1], ref["cells_P"]),
                 "well cells (MRST's are 1-based)")
    reports = out.report["ControlstepReports"]
    its = np.array([r["Iterations"] for r in reports])
    nmini = np.array([len(r["StepReports"]) for r in reports])
    p = np.array([st["pressure"] for st in out.states])
    sg = np.array([np.asarray(st["s"])[:, 1] for st in out.states])
    ok &= report(np.array_equal(its[:2], ref["its"][:2]), "steps 1-2: Newton iterations identical")
    for name, actual, expected in (("pressure", p[:2], ref["p"][:2]), ("gas saturation", sg[:2], ref["sg"][:2])):
        err = _rel(actual, expected)
        ok &= report(err < EXACT_STEP_RTOL, f"steps 1-2: {name} rel. diff {err:.1e}")
    ok &= restart_step_3(out, ref)
    ok &= report(np.array_equal(nmini, ref["nmini"]), f"ministeps identical at all {nmini.size} steps")
    types = [",".join(w.type for w in ws) for ws in out.wellSols]
    ok &= report(types == [str(t) for t in ref["types"]], "well controls identical at every step")
    print(f"info Newton iterations {its.sum()} (MRST {ref['its'].sum()}), equal in "
          f"{np.count_nonzero(its == ref['its'])} of {its.size} steps")
    well = {k: np.array([[getattr(w, k) for w in ws] for ws in out.wellSols]) for k in ("bhp", "qOs", "qGs")}
    for name, actual, expected in (("pressure", p, ref["p"]), ("gas saturation", sg, ref["sg"]),
                                   ("bhp", well["bhp"], ref["bhp"]), ("qOs", well["qOs"], ref["qOs"]),
                                   ("qGs", well["qGs"], ref["qGs"])):
        err = _rel(actual, expected)
        ok &= report(err < RUN_RTOL, f"run: {name} rel. diff {err:.1e}")
    print("\ntenthCSP_Model_I matches MRST" if ok else "\ntenthCSP_Model_I does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
