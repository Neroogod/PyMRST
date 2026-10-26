"""Validate MRST eggExample: the port in `ad_blackoil/examples/egg/egg_example.py`.

Two references from MRST 2024a, both on realization 0:

* `ref_egg_example_direct.mat` (`dump_egg_example_direct_reference.m`): the
  first three report steps with a direct linear solver (`BackslashSolverAD`)
  in place of the example's GMRES-ILU. This is the model check: well curves,
  pressures and saturations to `RTOL_DIRECT`, Newton iterations exactly.
* `ref_egg_example.mat` (`dump_egg_example_reference.m`): the example as it
  runs, all 123 steps with GMRES-ILU at the CPR default tolerance. MATLAB's
  `gmres` and SciPy's stop at different iterates inside that tolerance, so
  the two runs drift apart by the linear tolerance, not by round-off: the
  initial state is compared exactly, the well curves to `RTOL_ITERATIVE`
  (relative to each curve's largest magnitude), and the Newton iterations
  are reported. The Python side of this run is `run_egg_example_python.py`
  (`py_egg_example.npz`).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
RTOL_DIRECT = 1e-9
RTOL_ITERATIVE = 1e-3
DIRECT_STEPS = 3


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def run_direct():
    """The example's first `DIRECT_STEPS` steps with `BackslashSolverAD`."""
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad,
    )
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    from pymrst.modules.ad_core.utils.get_well_output import get_well_output
    from pymrst.modules.ad_core.utils.schedule import Schedule
    from pymrst.modules.test_suite.dataset_functions.egg.setup_egg import setup_egg

    G, rock, fluid, deck, _ = setup_egg(0)
    state, model, schedule, nonlinear = init_eclipse_problem_ad(deck, G=G,
                                                                TimestepStrategy="none")
    nonlinear.LinearSolver = BackslashSolverAD()
    schedule = Schedule(step_val=schedule.step_val[:DIRECT_STEPS],
                        step_control=schedule.step_control[:DIRECT_STEPS],
                        controls=schedule.controls)
    wellSols, states, rep = simulate_schedule_ad(state, model, schedule,
                                                 NonLinearSolver=nonlinear)
    W = schedule.controls[0].W
    inj = np.flatnonzero([w.sign > 0 for w in W])
    prod = np.flatnonzero([w.sign < 0 for w in W])
    return dict(bhp=get_well_output(wellSols, "bhp", inj)[0][:, :, 0],
                orat=get_well_output(wellSols, "qOs", prod)[0][:, :, 0],
                wrat=get_well_output(wellSols, "qWs", prod)[0][:, :, 0],
                p=np.array([s["pressure"] for s in states]),
                sw=np.array([np.asarray(s["s"])[:, 0] for s in states]),
                its=np.asarray(rep["Iterations"]).astype(int))


def main() -> int:
    ok = True
    print("First three steps, direct linear solver:")
    ref = loadmat(HERE/"ref_egg_example_direct.mat")
    with isolated_gravity():
        out = run_direct()
    its = out.pop("its")
    ok &= report(np.array_equal(its, np.ravel(ref["its"]).astype(int)),
                 f"Newton iterations {its.tolist()} (MRST {np.ravel(ref['its']).tolist()})")
    # No water reaches the producers in three steps: their water rates are
    # round-off (1e-125), so they are measured against the oil rates.
    scale = {"wrat": np.abs(np.asarray(ref["orat"], dtype=float)).max()}
    for key, val in out.items():
        expected = np.asarray(ref[key], dtype=float).reshape(np.shape(val))
        err = float(np.abs(val - expected).max()
                    / scale.get(key, max(np.abs(expected).max(), 1e-300)))
        ok &= report(err < RTOL_DIRECT, f"{key} rel. diff {err:.1e}")

    print("\nThe example as it runs (123 steps, GMRES-ILU):")
    ref = loadmat(HERE/"ref_egg_example.mat")
    py = np.load(HERE/"py_egg_example.npz")
    ok &= report(str(py["lsolver"]) == str(np.ravel(ref["lsolver"])[0]),
                 f"linear solver {py['lsolver']}")
    ok &= report(np.array_equal(py["p0"], np.ravel(ref["p0"]))
                 and np.array_equal(py["s0"], ref["s0"]),
                 "equilibrated initial pressure and saturations identical")
    for key in ("bhp", "orat", "wrat"):
        err = _rel(py[key], ref[key])
        ok &= report(err < RTOL_ITERATIVE, f"{key} rel. diff {err:.1e}")
    its_py, its_ref = py["its"].ravel(), np.ravel(ref["its"])
    print(f"info Newton iterations {its_py.sum()} (MRST {its_ref.sum()}), "
          f"equal in {(its_py == its_ref).sum()} of {its_ref.size} steps; "
          f"wall time {float(py['walltime']):.0f} s (MRST {float(np.ravel(ref['walltime'])[0]):.0f} s)")
    print("\neggExample matches MRST" if ok else "\neggExample does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
