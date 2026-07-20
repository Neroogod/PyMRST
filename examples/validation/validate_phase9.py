"""Compare PyMRST's vertical-equilibrium port against MATLAB co2lab-ve.

Two parts, matching dump_phase9_reference.m:

1. topSurfaceGrid + averageRock: the collapse of a 3D columnar aquifer
   onto its top surface (thickness H, caprock depth z, areal footprint,
   upscaled rock) and the resulting VE transmissibilities (K*H, harmonic).
2. makeVEFluid('sharp_interface_simple'): the upscaled krG/krW and the
   capillary pressure, on a sweep of upscaled saturations during both
   drainage (sGmax = sG) and imbibition (sGmax held fixed, which exercises
   the residually trapped CO2 zone in krW).
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.co2lab.ve import (
    VEFluid,
    VEModel,
    average_rock,
    top_surface_grid,
)
from pymrst.params.rock import make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import DARCY, MILLI

HERE = Path(__file__).parent


def main() -> int:
    path = HERE / "ref_p9_ve.mat"
    if not path.exists():
        print("ref_p9_ve.mat missing (run dump_phase9_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    failed = 0

    def chk(name, a, b, rtol):
        nonlocal failed
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        scale = max(np.abs(b).max(), 1e-300)
        diff = float(np.abs(a - b).max())
        ok = a.shape == b.shape and diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name:26s} rel diff = "
              f"{diff / scale:.2e}  (tol {rtol:.0e})")
        failed += 0 if ok else 1

    # --- 1. top-surface grid and upscaled rock --------------------------
    g3 = cart_grid([50, 1, 10], [2000.0, 100.0, 20.0])
    x = g3.nodes.coords[:, 0]
    g3.nodes.coords[:, 2] += 1000.0 - 0.04 * x
    g3 = compute_geometry(g3)
    rock3 = make_rock(g3, 400 * MILLI * DARCY, 0.2)

    ts = top_surface_grid(g3)
    rock2 = average_rock(rock3, ts)

    assert ts.g.cells.num == int(ref["ncells_ve"])
    chk("column thickness H", ts.H, ref["H_ve"], 1e-12)
    chk("caprock depth z", ts.z, ref["z_ve"], 1e-12)
    chk("areal footprint", ts.area, ref["vol_ve"], 1e-12)
    chk("upscaled porosity", rock2.poro, ref["poro_ve"], 1e-12)
    chk("upscaled permeability", rock2.perm[:, 0], ref["perm_ve"], 1e-12)

    f = VEFluid(rhoWS=float(ref["rhoW"]), rhoGS=float(ref["rhoG"]),
                mu_w=8e-4, mu_g=6e-5,
                srw=float(ref["srw"]), srg=float(ref["srg"]),
                krW_max=1.0, krG_max=1.0)
    model = VEModel(ts, rock2, f)

    chk("VE pore volume", model.pv, ref["pv_ve"], 1e-12)
    n_ml = np.atleast_2d(np.asarray(ref["N_ve"], np.int64)) - 1
    same_n = np.array_equal(model.N, n_ml)
    print(f"  {'ok  ' if same_n else 'FAIL'} "
          f"{'VE face neighbours':26s} {'identical' if same_n else 'DIFFER'}")
    failed += 0 if same_n else 1
    chk("VE transmissibility (K*H)", model.T, ref["T_ve"], 1e-11)

    # --- 2. sharp-interface upscaled fluid ------------------------------
    sg_q = np.asarray(ref["sg_q"], float).ravel()
    H = ts.H[0]
    Hv = np.full(sg_q.size, H)

    h, hmax = f.heights(sg_q, sg_q, Hv)               # drainage: sGmax = sG
    chk("upscaled krG (drainage)", value(f.kr_g(h, Hv)), ref["krG_q"], 1e-12)
    chk("upscaled krW (drainage)", value(f.kr_w(h, hmax, Hv)),
        ref["krW_q"], 1e-12)
    chk("VE capillary pressure",
        value(f.pc(h, np.full(sg_q.size, 1e7))), ref["pc_q"], 1e-12)

    sgmax_fix = float(ref["sgmax_fix"])
    sg_imb = np.minimum(sg_q, sgmax_fix)
    h_i, hmax_i = f.heights(sg_imb, np.full(sg_q.size, sgmax_fix), Hv)
    chk("upscaled krW (imbibition)", value(f.kr_w(h_i, hmax_i, Hv)),
        ref["krW_imb"], 1e-12)

    print("\nVertical equilibrium matches MATLAB co2lab-ve!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
