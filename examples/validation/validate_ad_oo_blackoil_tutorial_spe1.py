"""Validate MRST blackoilTutorialSPE1 through the AD-OO port.

The tutorial runs end to end on the ported stack: `setupSPE1`,
`selectModelFromDeck` (a `GenericBlackOilModel` with `GenericFacilityModel`),
`convertDeckScheduleToMRST` (`processWells`, `makeScheduleConsistent`) and
`simulateScheduleAD` with a line-searching `NonLinearSolver`. The reference is
MRST 2024a's own run (`dump_blackoil_tutorial_spe1_reference.m`). Checked:
- every step's pressure, saturations and `rs`;
- both wells' rates and BHP;
- the well order and control modes;
- every step's Newton iteration count;
- the vectors the tutorial compares with Eclipse 100: the GOR and BHP read
  from `SPE1_smry.mat`, and the simulated GOR.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    blackoil_tutorial_spe1,
)
from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent
RTOL = 1e-10


def main() -> int:
    path = HERE/"ref_blackoil_tutorial_spe1.mat"
    if not path.exists():
        print(f"{path.name} missing (run dump_blackoil_tutorial_spe1_reference.m)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    with isolated_gravity():
        out = blackoil_tutorial_spe1(plot=False)
    states = out.states
    run = dict(
        p=np.array([s["pressure"] for s in states]),
        rs=np.array([s["rs"] for s in states]),
        sw=np.array([np.asarray(s["s"])[:, 0] for s in states]),
        sg=np.array([np.asarray(s["s"])[:, 2] for s in states]),
        qWs=out.qWs, qOs=out.qOs, qGs=out.qGs, bhp=out.bhp, T=out.T,
        Tcomp=out.Tcomp, gor_ecl=out.gor_ecl, gor_mrst=out.gor_mrst,
        bhp_prod_ecl=out.bhp_prod_ecl, bhp_inj_ecl=out.bhp_inj_ecl)
    failed = False
    for key, actual in run.items():
        expected = np.asarray(ref[key], dtype=float)
        scale = max(np.nanmax(np.abs(expected)), 1e-300)
        err = np.nanmax(np.abs(actual - expected))/scale
        ok = actual.shape == expected.shape and err < RTOL
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {key}: rel {err:.2e}")
    checks = {
        "Newton iterations identical": np.array_equal(
            np.asarray(out.report["Iterations"]).astype(int), np.ravel(ref["its"]).astype(int)),
        "well order identical": [w.name for w in out.wellSols[0]] == list(ref["names"]),
        "control modes identical": [",".join(w.type for w in ws) for ws in out.wellSols]
                                   == list(ref["types"]),
        "injector and producer identified": (out.inj + 1, out.prod + 1)
                                            == (int(ref["inj"]), int(ref["prod"])),
    }
    for name, ok in checks.items():
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
