"""Phase 70: VEM_linElastTEDFM solve wrapper vs MATLAB MRST."""

from pathlib import Path
import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.vem import (
    ElasticBC,
    calculate_stress_vem,
    create_augmented_grid,
    enu2c_2d,
    vem_lin_elast_tedfm,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p70_vem_tedfm_solve.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def _dense(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x, dtype=float)


def compare(name, py, ml, rtol=1e-10, atol=1e-12) -> int:
    py = _dense(py).ravel()
    ml = _dense(ml).ravel()
    scale = max(float(np.max(np.abs(ml))), 1.0)
    diff = float(np.max(np.abs(py - ml)))
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase70_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    E = float(ref["E"])
    nu = float(ref["nu"])
    nx = int(ref["nx"])
    ny = int(ref["ny"])
    L = np.asarray(ref["L"], dtype=float).ravel()

    g = create_augmented_grid(compute_geometry(cart_grid([nx, ny], L)))
    g.Matrix = g
    C = enu2c_2d(np.full(g.cells.num, E), np.full(g.cells.num, nu))

    coords = g.nodes.coords[:, :2]
    bottom_nodes = np.flatnonzero(coords[:, 1] < 1e-9)
    left_nodes = np.flatnonzero(coords[:, 0] < 1e-9)
    disp_nodes = np.concatenate([bottom_nodes, left_nodes])
    disp_mask = np.vstack([
        np.tile([False, True], (bottom_nodes.size, 1)),
        np.tile([True, False], (left_nodes.size, 1)),
    ])
    disp_vals = np.zeros((disp_nodes.size, g.griddim))

    bd = np.any(g.faces.neighbors == -1, axis=1)
    top_faces = np.flatnonzero(bd & (g.faces.centroids[:, 1] > L[1] - 1e-9))
    force = np.tile([0.0, -1e6], (top_faces.size, 1))
    el_bc = ElasticBC(
        disp_nodes=disp_nodes,
        disp_mask=disp_mask,
        disp_vals=disp_vals,
        force_faces=top_faces,
        force=force,
    )

    uu, extra = vem_lin_elast_tedfm(
        g, C, el_bc, load=lambda x: np.zeros_like(x)
    )
    stress, strain = calculate_stress_vem(g, uu, C)
    disc = extra["disc"]

    failures = 0
    print("Phase 70: VEM_linElastTEDFM solve vs MATLAB MRST")
    failures += compare("C", C, _arr(ref, "C"), rtol=1e-14)
    failures += compare("uu", uu, _arr(ref, "uu"), rtol=1e-10)
    failures += compare("stress", stress, _arr(ref, "stress"), rtol=1e-10)
    failures += compare("strain", strain, _arr(ref, "strain"), rtol=1e-10)
    failures += compare("S", extra["S"], ref["S"], rtol=1e-10)
    failures += compare("A", extra["A"], ref["A"], rtol=1e-10)
    failures += compare("rhs", extra["rhs"], _arr(ref, "rhs"), rtol=1e-10)
    failures += compare("isdirdofs", disc["isdirdofs"].astype(float),
                        _arr(ref, "isdirdofs"), rtol=0, atol=0)
    failures += compare("V_dir", disc["V_dir"], _arr(ref, "V_dir"),
                        rtol=1e-12)
    failures += compare("ovol_div", disc["ovol_div"], ref["ovol_div"],
                        rtol=1e-12)
    failures += compare("gradP", disc["gradP"], ref["gradP"], rtol=1e-12)
    failures += compare("div", disc["div"], ref["div"], rtol=1e-12)
    failures += compare("divrhs", disc["divrhs"], _arr(ref, "divrhs"),
                        rtol=1e-12)

    if failures:
        print(f"\nVEM_linElastTEDFM validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST VEM_linElastTEDFM solve wrapper.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
