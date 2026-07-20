"""Compare PyMRST's CO2 module against MATLAB co2lab.

1. The sampled CO2 property tables (density and viscosity) interpolated
   at 851 (p, T) points.
2. A 3D CO2 injection into a brine aquifer run with co2lab's
   TwoPhaseWaterGasModel: pressures, gas saturations, and well responses.

Note on bhp: the reservoir state matches to round-off, but the reported
bottom-hole pressure differs by ~4e-4 relative. bhp = p_cell + drawdown +
connection pressure drop, and MRST computes that last term inside
SimpleWell.updateConnectionPressureDrop -- which this MRST install has
locally patched (the "OMO modification" reading fluid.rhoSinj). PyMRST
uses the reservoir CO2 density at the perforation. Since the well is
rate-controlled, bhp is a diagnostic output and does not feed back into
the reservoir solution, which is why pressures and saturations still
agree to 1e-13.

The aquifer is sealed, so 0.1 Mt/yr drives the pressure from 125 bar to
610 bar over the 8 steps -- a numerically demanding test, not a realistic
storage scenario.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.co2lab import CO2BrineModel, co2_brine_fluid, co2_props
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.units import DARCY, DAY, MILLI

HERE = Path(__file__).parent


def main() -> int:
    path = HERE / "ref_p8_co2.mat"
    if not path.exists():
        print("ref_p8_co2.mat missing (run dump_phase8_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    failed = 0

    def chk(name, a, b, rtol):
        nonlocal failed
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        scale = max(np.abs(b).max(), 1e-300)
        diff = float(np.abs(a - b).max())
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name:22s} rel diff = "
              f"{diff / scale:.2e}  (tol {rtol:.0e})")
        failed += 0 if ok else 1

    # --- 1. property tables ---------------------------------------------
    co2 = co2_props()
    pq = np.asarray(ref["pq"], float).ravel()
    tq = np.asarray(ref["tq"], float).ravel()
    PQ, TQ = np.meshgrid(pq, tq, indexing="ij")
    chk("CO2 density table", co2.rho(PQ.ravel(order="F"),
                                     TQ.ravel(order="F")),
        ref["rho_tab"], 1e-12)
    chk("CO2 viscosity table", co2.mu(PQ.ravel(order="F"),
                                      TQ.ravel(order="F")),
        ref["mu_tab"], 1e-12)

    # --- 2. 3D CO2 injection --------------------------------------------
    nx, ny, nz = 20, 1, 10
    g = cart_grid([nx, ny, nz], [2000.0, 100.0, 50.0])
    g.nodes.coords[:, 2] += 1200.0
    g = compute_geometry(g)
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)

    t_res = float(ref["T_res"])
    fluid = co2_brine_fluid(temperature=t_res, rhoWS=1020.0, srw=0.2,
                            src=0.05, rho_gs=float(ref["rhoGS"]))

    inj = 0 + nx * ny * (nz - 1)
    w = add_well(None, g, rock, np.array([inj]), well_type="rate",
                 val=float(ref["qGs"]), radius=0.15,
                 compi=np.array([0.0, 1.0]), sign=1,
                 gravity=np.array([0.0, 0.0, NORM_GRAVITY]))
    chk("well index", w[0].WI, np.atleast_1d(ref["WI"]), 1e-12)

    model = CO2BrineModel(g, rock, fluid, w, tol=1e-11)
    p0 = 1e5 + 1020.0 * NORM_GRAVITY * g.cells.centroids[:, 2]
    st = model.initial_state(p0, np.zeros(g.cells.num))

    dts = np.asarray(ref["dts"], float).ravel()
    p_hist, sg_hist, qgs, bhp = [], [], [], []
    for i, dt in enumerate(dts):
        st, ok, its = model.step(st, dt)
        assert ok, f"step {i} failed after {its} iterations"
        p_hist.append(st.pressure.copy())
        sg_hist.append(st.sG.copy())
        qgs.append(st.qGs[0])
        bhp.append(st.bhp[0])

    chk("pressure", np.vstack(p_hist), ref["p_hist"], 1e-6)
    chk("gas saturation sG", np.vstack(sg_hist), ref["sg_hist"], 1e-5)
    chk("well gas rate qGs", np.asarray(qgs), ref["qGs_hist"], 1e-8)
    chk("well bhp", np.asarray(bhp), ref["bhp_hist"], 1e-3)

    print("\nCO2 storage matches MATLAB co2lab!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
