"""Validate PyMRST 3D planar EDFM against an MRST HFM reference dump."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.hfm import (
    PlanarFracture,
    build_edfm_3d,
    project_planar_fracture_to_grid,
)
from pymrst.modules.vemmech import create_augmented_grid
from pymrst.utils.units import DARCY

HERE = Path(__file__).parent
REF = HERE / "ref_p107_hfm3d.mat"


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase107_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    celldim = np.asarray(ref["celldim"], dtype=int).ravel()
    physdim = np.asarray(ref["physdim"], dtype=float).ravel()
    pts = np.asarray(ref["points"], dtype=float)
    center = pts.mean(axis=0)
    axis1 = pts[2] - pts[1]
    axis2 = pts[1] - pts[0]
    frac = PlanarFracture(
        center=center,
        normal=np.cross(axis1, axis2),
        axis1=axis1,
        half_length1=0.5 * np.linalg.norm(axis1),
        half_length2=0.5 * np.linalg.norm(axis2),
    )
    g = create_augmented_grid(compute_geometry(cart_grid(celldim, physdim)))
    proj = project_planar_fracture_to_grid(g, frac, aperture=0.02)
    edfm = build_edfm_3d(
        g,
        [frac],
        aperture=0.02,
        perm_matrix=np.full(g.cells.num, 1.0 * DARCY),
        perm_frac=1.0e4 * DARCY,
        poro_matrix=np.full(g.cells.num, 0.2),
        poro_frac=0.5,
    )

    mrst_cells_raw = np.asarray(ref["matrix_cells"], dtype=int).ravel() - 1
    mrst_cells = np.unique(mrst_cells_raw)
    mrst_area = np.asarray(ref["matrix_area"], dtype=float).ravel()
    py_cells = np.sort(np.unique(proj.matrix_cells))
    mrst_cells = np.sort(mrst_cells)

    failures = 0

    def chk(name, ok, detail=""):
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{detail}")
        failures += 0 if ok else 1

    print("Phase 107: HFM 3D planar EDFM vs MATLAB MRST")
    chk("unique matrix intersection cells", py_cells.size >= mrst_cells.size,
        f" ({py_cells.size} vs {mrst_cells.size})")
    frac_vol_ref = float(np.asarray(ref["frac_volumes"], dtype=float).sum())
    chk("total fracture volume", np.isclose(edfm.frac_volumes.sum(), frac_vol_ref, rtol=1e-10),
        f" ({edfm.frac_volumes.sum():.6g} vs {frac_vol_ref:.6g})")
    chk("total projected area", np.isclose(proj.areas.sum() * 0.02, frac_vol_ref, rtol=1e-10),
        f" ({proj.areas.sum():.6g} area)")
    chk("fracture cells", edfm.frac_cells == proj.matrix_cells.size)
    chk("frac-matrix NNC count", np.sum(np.asarray(edfm.conn_type) == "frac_matrix") == edfm.frac_cells)
    print("\nPyMRST HFM 3D reference validation passed." if failures == 0
          else f"\n{failures} HFM 3D checks failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
