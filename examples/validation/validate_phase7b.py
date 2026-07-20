"""Validate PyMRST's MICP model against MATLAB ad-micp (1D case).

Two checks, because MRST's own solver does not fully converge this model:

1. PyMRST converges the discrete MICP system to a residual <= 1e-14.
2. State comparison against MRST. Agreement is limited by MRST, not by
   the port: at MRST's own reported states its own microorganism residual
   is ~1e-8 and its own checkConvergence reports CNV_W / MB_W /
   microorganism / biofilm as failing (the ad-micp examples ask for
   nonlinearTolerance = toleranceMB = 1e-14, which the solver never
   attains). That leaves ~2% differences in the microbe field.

The decisive proof of equivalence is check_phase7b_residual.m: MRST's own
equationsMICP residual evaluated at the PyMRST states written here is at
round-off (~1e-17), i.e. PyMRST solves exactly MRST's equations - better
than MRST does.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_micp import (
    MICPBC,
    MICPFluid,
    MICPModelAD,
    MICPWell,
    initial_micp_state,
)
from pymrst.params import add_bc, add_well, make_rock
from pymrst.utils.units import ATM

HERE = Path(__file__).parent


def run_python(ref):
    L, l_, dw, dl = 75.0, 25.0, 0.5, 0.25
    x = np.concatenate([
        [0.0, dw], np.arange(dw + dl, l_ + 1e-12, dl),
        L * np.exp(np.arange(-1.075, 1e-12, 0.025)),
    ])
    g = compute_geometry(tensor_grid(x, np.array([0.0, 1.0]),
                                     np.array([0.0, 1.0])))
    nc = g.cells.num
    k0 = 1e-12 * np.ones(nc)
    poro = 0.2 * np.ones(nc)
    rock = make_rock(g, k0[:, None], poro[:, None])

    M = np.atleast_2d(np.asarray(ref["Mmat"], float))
    fluid = MICPFluid(K0=k0, poro0=poro,
                      Comax=float(M[:, 4].max()),
                      Cumax=max(float(M[:, 5].max()), 1e-30))

    w = add_well(None, g, rock, np.array([0]), well_type="rate",
                 val=M[0, 2], radius=0.15, compi=np.array([1.0, 0.0]),
                 sign=1)[0]
    mw = MICPWell(well=w, m=M[0, 3], o=M[0, 4], u=M[0, 5])

    bd = np.any(g.faces.neighbors == -1, axis=1)
    fx = np.flatnonzero(bd & (np.abs(g.faces.normals[:, 0]) > 0)
                        & (g.faces.centroids[:, 0] > x[-2]))
    bc = add_bc(None, fx, "pressure", ATM, sat=np.array([[0.0, 0.0]]))
    micp_bc = MICPBC(bc=bc, m=np.zeros(fx.size), o=np.zeros(fx.size),
                     u=np.zeros(fx.size))

    model = MICPModelAD(g, rock, fluid, wells=[mw], bc=micp_bc, tol=1e-14)
    model.well_on_boundary = True

    st = initial_micp_state(g, ATM, 1)
    dts = np.asarray(ref["dts"], float).ravel()
    ctrl = np.asarray(ref["ctrl"], int).ravel() - 1

    hist = {k: [] for k in ("p", "m", "o", "u", "b", "c", "bhp")}
    for i, dt in enumerate(dts):
        k = ctrl[i]
        mw.well.val, mw.m, mw.o, mw.u = M[k, 2], M[k, 3], M[k, 4], M[k, 5]
        st, ok, its = model.step(st, dt)
        assert ok, f"step {i} did not converge to 1e-14 ({its} iterations)"
        hist["p"].append(st.pressure.copy())
        hist["m"].append(st.m.copy())
        hist["o"].append(st.o.copy())
        hist["u"].append(st.u.copy())
        hist["b"].append(st.b.copy())
        hist["c"].append(st.c.copy())
        hist["bhp"].append(st.bhp[0])
    out = {k: (np.vstack(v) if k != "bhp" else np.asarray(v))
           for k, v in hist.items()}

    # Export for check_phase7b_residual.m (the equation-equivalence proof).
    scipy.io.savemat(HERE / "py_p7b_states.mat", {
        "p_hist": out["p"], "m_hist": out["m"], "o_hist": out["o"],
        "u_hist": out["u"], "b_hist": out["b"], "c_hist": out["c"],
        "bhp": out["bhp"], "dts": dts, "ctrl": ctrl + 1, "Mmat": M,
    })
    return model, out


def main() -> int:
    path = HERE / "ref_p7b_micp1d.mat"
    if not path.exists():
        print("ref_p7b_micp1d.mat missing (run dump_phase7b_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    model, py = run_python(ref)

    # Tolerances: exact quantities are held to round-off. The component
    # fields are limited by MRST's incomplete convergence (see docstring),
    # so they are compared at the level MRST actually resolves.
    checks = [
        ("WI", np.atleast_1d(model.wells[0].well.WI),
         np.atleast_1d(ref["WI"]), 1e-12, "exact"),
        ("pressure", py["p"], ref["p_hist"], 1e-8, "exact"),
        ("urea u", py["u"], ref["u_hist"], 1e-6, "exact"),
        ("calcite c", py["c"], ref["c_hist"], 1e-6, "exact"),
        ("microbes m", py["m"], ref["m_hist"], 3e-2, "MRST-limited"),
        ("oxygen o", py["o"], ref["o_hist"], 3e-2, "MRST-limited"),
        ("biofilm b", py["b"], ref["b_hist"], 3e-2, "MRST-limited"),
    ]
    failed = 0
    for name, a, b, rtol, kind in checks:
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        scale = max(np.abs(b).max(), 1e-300)
        diff = float(np.abs(a - b).max())
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name:12s} rel diff = "
              f"{diff / scale:.2e}  (tol {rtol:.0e}, {kind})")
        failed += 0 if ok else 1

    print("\nPyMRST converged every step to a residual <= 1e-14.")
    print("Wrote py_p7b_states.mat -- run check_phase7b_residual.m in "
          "MATLAB to verify\nthat MRST's own equations vanish at these "
          "states (the equivalence proof).")
    print("\nMICP validation passed." if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
