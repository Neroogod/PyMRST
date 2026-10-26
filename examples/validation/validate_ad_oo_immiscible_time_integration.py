"""Validate MRST immiscibleTimeIntegrationExample: the port in
`ad_blackoil/examples/immiscible_time_integration_example.py`.

The reference is MRST 2024a's runs of the example's models
(`dump_immiscible_time_integration_reference.m`): fully implicit, explicit,
adaptive implicit, explicit at CFL 5, sequential implicit and sequential
explicit. ("Sequential-AIM" is set up with 'explicit' in MRST, so it repeats
the sequential-explicit run and is not compared separately.) The port's runs
come from the example itself, through its packed problems, in a temporary
output directory. Checked per run: Newton iterations per report step,
ministeps, and saturation and pressure at every report step; also AIM's
fraction of implicit cells and the fully implicit run's saturation CFL.

Fully implicit, explicit, AIM and sequential implicit match to round-off.
Two runs amplify round-off, which was confirmed by perturbing the port's own
initial saturation by 1e-14 and getting the same divergence from itself:
- explicit at CFL 5 is unstable by design. It matches to 1e-15 for the first
  `UNSTABLE_STEPS` steps, then grows tenfold per step to O(0.1). Its
  ministeps and iterations are compared exactly, its states only that far;
- sequential explicit picks its ministeps from CFL estimates of the state,
  so a 1e-15 difference grows to about 1e-8 in saturation and 5e-5 in the
  ministep lengths. It is checked at `SE_RTOL`, and its ministep count
  exactly.
"""

import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.immiscible_time_integration_example import (
    immiscible_time_integration_example,
)
from pymrst.modules.ad_core.utils.get_report_ministeps import get_report_ministeps
from pymrst.params.gravity.gravity import isolated_gravity
from pymrst.utils.settings_manager.mrst_settings import override_setting

HERE = Path(__file__).resolve().parent
RTOL = 1e-12
UNSTABLE_STEPS = 9
SE_RTOL = 1e-6
SE_MINISTEP_RTOL = 1e-3


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.abs(actual - expected).max()/max(np.abs(expected).max(), 1e-300))


def main() -> int:
    ref = loadmat(HERE/"ref_immiscible_time_integration.mat")
    with tempfile.TemporaryDirectory() as tmp, isolated_gravity():
        with override_setting("outputDirectory", tmp):
            out = immiscible_time_integration_example(plot=False, sequential=True)
    # The example's runs, in the reference's order: FIM, EXPL, AIM,
    # EXPL_unstable, SI, SE.
    states = list(out.states) + [out.states_all[3], out.states_all[4]]
    reports = list(out.reports) + [out.reports_all[3], out.reports_all[4]]
    labels = ["fully implicit", "explicit", "AIM", "explicit CFL 5", "sequential implicit",
              "sequential explicit"]
    ok = True
    for k, (label, run, rep) in enumerate(zip(labels, states, reports)):
        # Packed outputs give the control-step reports, as MRST's do.
        its = np.array([r["Iterations"] for r in rep], dtype=int)
        ok &= report(np.array_equal(its, np.ravel(ref["its"][0, k]).astype(int)),
                     f"{label}: Newton iterations identical over {its.size} steps")
        mini, mini_ref = get_report_ministeps(rep), np.ravel(ref["mini"][0, k])
        same = mini.size == mini_ref.size
        ok &= report(same, f"{label}: {mini_ref.size} ministeps")
        mtol = SE_MINISTEP_RTOL if k == 5 else RTOL
        if same:
            err = relative(mini, mini_ref)
            ok &= report(err < mtol, f"{label}: ministep lengths, rel. diff {err:.1e}")
        sw = np.array([np.asarray(st["s"])[:, 0] for st in run])
        p = np.array([st["pressure"] for st in run])
        sw_ref, p_ref = np.asarray(ref["sw"][0, k]), np.asarray(ref["p"][0, k])
        if k == 3:
            sw, sw_ref = sw[:UNSTABLE_STEPS], sw_ref[:UNSTABLE_STEPS]
            p, p_ref = p[:UNSTABLE_STEPS], p_ref[:UNSTABLE_STEPS]
            label = f"{label}, first {UNSTABLE_STEPS} steps"
        tol = SE_RTOL if k == 5 else RTOL
        ok &= report(np.abs(sw - sw_ref).max() < tol,
                     f"{label}: water saturation, max diff {np.abs(sw - sw_ref).max():.1e}")
        err = relative(p, p_ref)
        ok &= report(err < tol, f"{label}: pressure, rel. diff {err:.1e}")
    ok &= report(np.array_equal(out.imp_frac, np.ravel(ref["imp_frac"])),
                 "AIM: fraction of implicit cells at every step")
    err = relative(out.cfl, np.asarray(ref["cfl"]))
    ok &= report(err < RTOL, f"fully implicit saturation CFL, rel. diff {err:.1e}")
    print("\nimmiscibleTimeIntegrationExample matches MRST" if ok
          else "\nimmiscibleTimeIntegrationExample does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
