"""Validate MRST blackoilTutorialGravSeg: the port in
`ad_blackoil/examples/blackoil_tutorial_grav_seg.py`, on the AD-OO
TwoPhaseOilWaterModel.

The reference is MRST 2024a's own run (`dump_ad_oo_grav_seg_reference.m`).

The first report step's second Newton system is numerically singular in
both codes (condition number ~1e17 here, MATLAB warns `RCOND = 7e-17`), so
which update it produces is decided by each linear solver's round-off and
the two Newton paths through that step differ. Both converge to the same
solution within the nonlinear tolerance, and every later step takes MRST's
iteration count exactly. The checks say so: step 1's count is excluded and
the state tolerance is the convergence tolerance, not round-off.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_grav_seg import (
    blackoil_tutorial_grav_seg,
)
from pymrst.params.gravity.gravity import isolated_gravity
from pymrst.utils.units import BARSA

HERE = Path(__file__).resolve().parent
PRESSURE_TOL_BAR = 1e-6
SATURATION_TOL = 1e-5


def run_case():
    """The example as published, without its figures."""
    out = blackoil_tutorial_grav_seg(plot=False)
    p = np.array([st["pressure"] for st in out.states])
    s_w = np.array([np.asarray(st["s"])[:, 0] for st in out.states])
    return p, s_w, np.asarray(out.report["Iterations"])


def main() -> int:
    path = HERE/"ref_ad_oo_grav_seg.mat"
    if not path.exists():
        print(f"{path.name} missing (run dump_ad_oo_grav_seg_reference.m)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    with isolated_gravity():
        p, s_w, its = run_case()
    dp = np.abs(p - ref["p"]).max()/BARSA
    ds = np.abs(s_w - ref["sw"]).max()
    same_its = np.array_equal(its[1:], np.ravel(ref["its"])[1:])
    results = [(dp < PRESSURE_TOL_BAR, f"pressure max |diff| {dp:.2e} bar"),
               (ds < SATURATION_TOL, f"sW max |diff| {ds:.2e}"),
               (same_its, "Newton iterations identical after the singular step 1")]
    for ok, text in results:
        print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return int(not all(ok for ok, _ in results))


if __name__ == "__main__":
    raise SystemExit(main())
