"""Compare PyMRST's flow diagnostics against MATLAB's diagnostics module.

A heterogeneous quarter five-spot: forward and backward time-of-flight,
well tracer partitions, and the derived F-Phi curve, Lorenz coefficient
and sweep efficiency.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.diagnostics import (
    compute_diagnostics,
    compute_f_phi,
    compute_sweep,
    lorenz_coefficient,
)
from pymrst.modules.incomp import incomp_tpfa, init_single_fluid
from pymrst.params import add_well, make_rock, pore_volume
from pymrst.solvers import compute_trans, init_state
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

HERE = Path(__file__).parent


def main() -> int:
    path = HERE / "ref_p11_diagnostics.mat"
    if not path.exists():
        print("ref_p11_diagnostics.mat missing (run dump_phase11_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    failed = 0

    def chk(name, a, b, rtol):
        nonlocal failed
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        scale = max(np.abs(b).max(), 1e-300)
        diff = float(np.abs(a - b).max()) if a.shape == b.shape else np.inf
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name:24s} rel diff = "
              f"{diff / scale:.2e}  (tol {rtol:.0e})")
        failed += 0 if ok else 1

    n = 32
    g = compute_geometry(cart_grid([n, n, 1], [500.0, 500.0, 10.0]))
    kvals = (10 + (7 * np.arange(g.cells.num)) % 100) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.2)
    fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
    t = compute_trans(g, rock)

    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=1e-3,
                 radius=0.1, compi=np.array([1.0]))
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=100 * BARSA, radius=0.1, compi=np.array([1.0]))
    st = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, wells=w)

    chk("pressure", st.pressure, ref["p"], 1e-9)
    chk("flux", st.flux, ref["flux"], 1e-9)

    d = compute_diagnostics(g, rock, st, w)
    chk("forward time-of-flight", d.tof_forward, ref["tof_f"], 1e-9)
    chk("backward time-of-flight", d.tof_backward, ref["tof_b"], 1e-9)
    chk("injector tracer", d.tracer_inj[:, 0], ref["tr_inj"], 1e-9)
    chk("producer tracer", d.tracer_prod[:, 0], ref["tr_prd"], 1e-9)

    pv = pore_volume(g, rock)
    f, phi = compute_f_phi(pv, d.total_tof)
    chk("flow capacity F", f, ref["F"], 1e-9)
    chk("storage capacity Phi", phi, ref["Phi"], 1e-9)

    lc = lorenz_coefficient(f, phi)
    lc_ml = float(ref["Lc"])
    ok = abs(lc - lc_ml) <= 1e-9 * abs(lc_ml)
    print(f"  {'ok  ' if ok else 'FAIL'} {'Lorenz coefficient':24s} "
          f"{lc:.6f} vs {lc_ml:.6f}")
    failed += 0 if ok else 1

    ev, td = compute_sweep(f, phi)
    chk("sweep efficiency Ev", ev, ref["Ev"], 1e-9)
    chk("dimensionless time tD", td, ref["tD"], 1e-9)

    print("\nFlow diagnostics match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
