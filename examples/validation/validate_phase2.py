"""Compare PyMRST Phase 2 solver results against MATLAB MRST exports.

Rebuilds each case from dump_phase2_reference.m with PyMRST and compares
half-face transmissibilities, cell pressures, face fluxes/pressures, and
well solutions. Run dump_phase2_reference.m in MATLAB first.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import add_bc, add_source, add_well, gravity_vector, make_rock
from pymrst.modules.incomp import incomp_tpfa_ad as incomp_tpfa, init_single_fluid
from pymrst.solvers import compute_trans, init_res_sol, init_state
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

HERE = Path(__file__).parent
FLUID = init_single_fluid(1 * CENTI * POISE, 1000.0)


def solve_qfs() -> Dict[str, np.ndarray]:
    g = compute_geometry(cart_grid([32, 32, 1], [500.0, 500.0, 10.0]))
    kvals = (10 + (7 * np.arange(g.cells.num)) % 100) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.2)
    ht = compute_trans(g, rock)
    src = add_source(None, np.array([0]), 1e-3)
    src = add_source(src, np.array([g.cells.num - 1]), -1e-3)
    st = incomp_tpfa(init_res_sol(g, 0.0), g, ht, FLUID, src=src)
    return {"p": st.pressure, "flux": st.flux, "fp": st.facePressure,
            "hT": ht}


def solve_bc() -> Dict[str, np.ndarray]:
    g = compute_geometry(cart_grid([20, 10, 5], [200.0, 100.0, 50.0]))
    kvals = (1 + (3 * np.arange(g.cells.num)) % 50) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.25)
    ht = compute_trans(g, rock)
    bd = np.any(g.faces.neighbors == -1, axis=1)
    fc = g.faces.centroids
    bc = add_bc(None, np.flatnonzero(bd & (fc[:, 0] < 1e-10)),
                "pressure", 200 * BARSA)
    bc = add_bc(bc, np.flatnonzero(bd & (fc[:, 0] > 200 - 1e-10)),
                "pressure", 100 * BARSA)
    bc = add_bc(bc, np.flatnonzero(bd & (fc[:, 1] < 1e-10)), "flux", 1e-6)
    st = incomp_tpfa(init_res_sol(g, 0.0), g, ht, FLUID, bc=bc)
    return {"p": st.pressure, "flux": st.flux, "fp": st.facePressure,
            "hT": ht}


def solve_wells() -> Dict[str, np.ndarray]:
    g = compute_geometry(cart_grid([10, 10, 3], [1000.0, 1000.0, 30.0]))
    kvals = np.tile(np.array([100.0, 100.0, 10.0]) * MILLI * DARCY,
                    (g.cells.num, 1))
    rock = make_rock(g, kvals, 0.3)
    ht = compute_trans(g, rock)
    gvec = gravity_vector(3)
    w = add_well(None, g, rock, np.array([0, 100, 200]), well_type="rate",
                 val=5e-3, radius=0.1, compi=np.array([1.0]), gravity=gvec)
    w = add_well(w, g, rock, np.array([99, 199, 299]), well_type="bhp",
                 val=100 * BARSA, radius=0.1, compi=np.array([1.0]),
                 gravity=gvec)
    st = incomp_tpfa(init_state(g, w, 0.0), g, ht, FLUID, wells=w,
                     gravity=gvec)
    return {
        "p": st.pressure, "flux": st.flux, "hT": ht,
        "wi1": w[0].WI, "wi2": w[1].WI, "dz1": w[0].dZ, "dz2": w[1].dZ,
        "wflux1": st.wellSol[0].flux, "wflux2": st.wellSol[1].flux,
        "bhp1": np.array([st.wellSol[0].pressure]),
        "bhp2": np.array([st.wellSol[1].pressure]),
    }


def solve_gravity() -> Dict[str, np.ndarray]:
    g = compute_geometry(cart_grid([1, 1, 30], [1.0, 1.0, 300.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
    ht = compute_trans(g, rock)
    bd = np.any(g.faces.neighbors == -1, axis=1)
    top = np.flatnonzero(bd & (g.faces.centroids[:, 2] < 1e-10))
    bc = add_bc(None, top, "pressure", 100 * BARSA)
    st = incomp_tpfa(init_res_sol(g, 0.0), g, ht, FLUID, bc=bc,
                     gravity=gravity_vector(3))
    return {"p": st.pressure, "flux": st.flux, "hT": ht}


CASES = {
    "ref_p2_qfs.mat": solve_qfs,
    "ref_p2_bc.mat": solve_bc,
    "ref_p2_wells.mat": solve_wells,
    "ref_p2_gravity.mat": solve_gravity,
}

# Pressures ~1e7 Pa, fluxes ~1e-3 m^3/s. ATOL_FLOOR absorbs pure round-off
# noise in fields that are physically zero (e.g. hydrostatic fluxes ~1e-19).
RTOL, ATOL_FLOOR = 1e-9, 1e-15


def compare_case(name: str) -> Tuple[int, int]:
    ref = scipy.io.loadmat(HERE / name, squeeze_me=True)
    py = CASES[name]()
    failed = 0
    for key, pv in py.items():
        mv = np.asarray(ref[key], dtype=float).ravel()
        pv = np.asarray(pv, dtype=float).ravel()
        scale = max(np.abs(mv).max(), 1e-300)
        # atol scaled by field magnitude: fields like the source-driven QFS
        # pressure cross zero, where element-wise rtol alone is meaningless.
        if not np.allclose(pv, mv, rtol=RTOL, atol=RTOL * scale + ATOL_FLOOR):
            worst = float(np.max(np.abs(pv - mv)))
            print(f"  FAIL {key}: max abs diff = {worst:.3e} "
                  f"(scale {scale:.3e})")
            failed += 1
    status = "FAILED" if failed else "matches MATLAB"
    print(f"{name}: {len(py)} arrays checked, {status}")
    return len(py), failed


def main() -> int:
    total_failed, missing = 0, 0
    for name in CASES:
        if not (HERE / name).exists():
            print(f"{name}: missing (run dump_phase2_reference.m in MATLAB)")
            missing += 1
            continue
        _, failed = compare_case(name)
        total_failed += failed
    if missing == len(CASES):
        print("\nNo reference files found - nothing validated.")
        return 1
    print("\nAll compared arrays match!" if total_failed == 0
          else f"\n{total_failed} array comparisons FAILED")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
