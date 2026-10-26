"""Validate MRST blackoilTutorialOnePhase: the port in
`ad_blackoil/examples/blackoil_tutorial_one_phase.py`, on the AD-OO WaterModel.

The reference is MRST 2024a's own run (`dump_ad_oo_one_phase_reference.m`):
equilibrated pressure, every report-step pressure field and the producer's
rate, for the open (constant-pressure south boundary) and closed
compartments, plus the Newton iteration counts. The AD-OO stack -
`WaterModel`, `equationsWater`, `SimpleWell`, `FacilityModel`,
`NonLinearSolver`, `simulateScheduleAD` - must reproduce all of it.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_one_phase import (
    blackoil_tutorial_one_phase,
)
from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
RTOL = 1e-12


def run_case():
    """`(p0, p_open, p_closed, q_open, q_closed, its_open, its_closed)` from
    the example as published, without its figures."""
    out = blackoil_tutorial_one_phase(plot=False)
    p = [np.array([st["pressure"] for st in states]) for states in (out.states1, out.states2)]
    q = [np.array([w[0].qWs for w in ws]).ravel() for ws in (out.wellSols1, out.wellSols2)]
    its = [out.report1["Iterations"], out.report2["Iterations"]]
    return (out.state["pressure"], *p, *q, *its)


def main() -> int:
    path = HERE/"ref_ad_oo_one_phase.mat"
    if not path.exists():
        print(f"{path.name} missing (run dump_ad_oo_one_phase_reference.m)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    with isolated_gravity():
        p0, p1, p2, q1, q2, its1, its2 = run_case()
    checks = [("equilibrated pressure", p0, ref["p0"]),
              ("open: pressure", p1, ref["p1"]), ("closed: pressure", p2, ref["p2"]),
              ("open: qWs", q1, ref["q1"]), ("closed: qWs", q2, ref["q2"])]
    failed = False
    for name, actual, expected in checks:
        err = np.abs(actual - expected).max()/np.abs(expected).max()
        ok = err < RTOL
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: rel {err:.2e}")
    for name, actual, expected in (("open", its1, ref["its1"]), ("closed", its2, ref["its2"])):
        ok = np.array_equal(np.ravel(actual), np.ravel(expected))
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: Newton iterations identical")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
