"""Phase 20: upscaled VE relperm and pc with a capillary fringe (co2lab-ve).

Against MRST's `addVERelpermCapillaryFringe` ('P-scaled table', its default)
and `free_sg`, on a mix of drainage and imbibition states and cells of
different thickness.

MRST's other branch, 'S table', cannot be compared: it raises before it can
return anything (`truncate_at_aquifer_bottom` indexes a misspelled variable,
`trunctate`). PyMRST's `make_ve_table_h` is that model with the typo fixed,
and so has no reference to compare against.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.modules.co2lab.ve_fluids import (              # noqa: E402
    VECapillaryFringeFluid,
    free_sg,
    make_ve_table_p,
    upper_cap_pressure,
)

REF = Path(__file__).with_name("ref_p20_ve_fluids.mat")
TOL = 1e-10


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase20_reference.m in MATLAB)")
        return 0

    r = scipy.io.loadmat(REF, squeeze_me=True)
    g = lambda k: np.asarray(r[k], float).ravel()      # noqa: E731

    srw, srg = float(r["srw"]), float(r["srg"])
    entry = float(r["entry"])
    H = g("H")
    sg, sgmax, p = g("sg"), g("sGmax"), g("p")

    def inv_pc_3d(pp):
        return 1.0 - np.clip(np.asarray(pp, float) / entry, 0.0, 1.0) * (1 - srw)

    def kr_3d(s):
        return np.asarray(s, float) ** 2

    # Build the table from PyMRST's own upper_cap_pressure rather than the
    # dumped number: it is the port of determine_upper_cap_press_limit, and
    # the table has to be sampled the way the MRST fluid samples it.
    p_max = upper_cap_pressure(inv_pc_3d, srw, float(H.max()), 1000.0 - 700.0)
    assert np.isclose(p_max, float(r["Pmax"]), rtol=1e-12), (
        f"upper_cap_pressure gives {p_max:.1f}, MATLAB {float(r['Pmax']):.1f}")
    table = make_ve_table_p(inv_pc_3d, kr_3d, p_max, samples=2000)
    fluid = VECapillaryFringeFluid(
        rhoWS=1000.0, rhoGS=700.0, mu_w=1e-3, mu_g=6e-5,
        srw=srw, srg=srg, H=H, table=table)

    checks = [
        ("the upscaled table: SP", table.SP, g("tab_SP")),
        ("the upscaled table: krP", table.krP, g("tab_krP")),
        ("free_sg (drainage + imbibition)",
         free_sg(sg, sgmax, srw, srg), g("fsg")),
        ("krG (capillary fringe)", fluid.kr_g(sg, p, sg_max=sgmax), g("krG")),
        ("pcWG (capillary fringe)", fluid.pc(sg, p, sg_max=sgmax), g("pcWG")),
        ("krW", fluid.kr_w(1.0 - sg, sg_max=sgmax), g("krW")),
    ]

    print(f"Upscaled VE fluid over {sg.size} cells "
          f"(H = {H.min():.0f}-{H.max():.0f} m, drainage and imbibition)\n")

    ok = True
    for name, mine, theirs in checks:
        mine = np.asarray(mine, float).ravel()
        scale = max(float(np.abs(theirs).max()), 1e-30)
        rel = np.abs(mine - theirs).max() / scale
        good = rel <= TOL
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:33s} "
              f"max rel diff = {rel:.2e}  (tol {TOL:.0e})")

    print(f"\n  (MRST krG = {np.round(g('krG'), 5)})")
    print("\nUpscaled VE fluid matches MATLAB!" if ok
          else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
