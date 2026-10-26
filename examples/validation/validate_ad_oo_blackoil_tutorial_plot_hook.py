"""Validate MRST blackoilTutorialPlotHook: the port in
`ad_blackoil/examples/blackoil_tutorial_plot_hook.py`.

The reference is MRST 2024a's run without the after-step function
(`dump_blackoil_tutorial_plot_hook_reference.m`); the function only draws,
which the port's self-test checks (same states with and without it). SPE1
on the AD-OO stack matches MRST to round-off, so every report step's
pressure, saturations, `rs`, well rates and BHPs are checked to `RTOL`, and
the Newton iterations and linear solver exactly.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_plot_hook import (
    blackoil_tutorial_plot_hook,
)
from pymrst.modules.ad_core.utils.well_sol_to_vector import well_sol_to_vector
from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
RTOL = 1e-10


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def main() -> int:
    ref = loadmat(HERE/"ref_blackoil_tutorial_plot_hook.mat")
    with isolated_gravity():
        out = blackoil_tutorial_plot_hook(plot=False)
    qWs, qOs, qGs, bhp = well_sol_to_vector(out.wellSols)
    actual = dict(p=np.array([s["pressure"] for s in out.states]),
                  sat=np.array([np.asarray(s["s"]).ravel(order="F") for s in out.states]),
                  rs=np.array([s["rs"] for s in out.states]),
                  qWs=qWs, qOs=qOs, qGs=qGs, bhp=bhp)
    ok = report(type(out.linsolve).__name__ == str(np.ravel(ref["lsolver"])[0]),
                f"linear solver {type(out.linsolve).__name__}")
    ok &= report(np.array_equal(np.asarray(out.report["Iterations"]).astype(int),
                                np.ravel(ref["its"]).astype(int)),
                 f"Newton iterations identical over {len(out.states)} steps")
    for key, value in actual.items():
        expected = np.asarray(ref[key], dtype=float)
        err = np.abs(value - expected).max()/max(np.abs(expected).max(), 1e-300)
        ok &= report(err < RTOL, f"{key} rel. diff {err:.1e}")
    print("\nblackoilTutorialPlotHook matches MRST" if ok
          else "\nblackoilTutorialPlotHook does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
