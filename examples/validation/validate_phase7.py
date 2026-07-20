"""Compare PyMRST VEM linear elasticity against MATLAB vemmech.

Rebuilds each reference case (regular and twisted 3x4x5 boxes with
heterogeneous elasticity, clamped bottom, top traction, body load) using
the node coordinates saved from MATLAB, and compares the full nodal
displacement fields.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech import (
    ElasticBC,
    create_augmented_grid,
    enu2c,
    vem_lin_elast,
)

HERE = Path(__file__).parent


def run_case(ref) -> np.ndarray:
    g = cart_grid([3, 4, 5], [3.0, 4.0, 5.0])
    g.nodes.coords = np.atleast_2d(np.asarray(ref["coords"], float))
    g = create_augmented_grid(compute_geometry(g))

    C = enu2c(np.asarray(ref["E"], float).ravel(),
              np.asarray(ref["nu"], float).ravel())

    pts = g.nodes.coords
    bn = np.flatnonzero(pts[:, 2] > 5 - 1e-9)
    bd = np.any(g.faces.neighbors == -1, axis=1)
    topf = np.flatnonzero(bd & (g.faces.centroids[:, 2] < 1e-9))

    bc = ElasticBC(
        disp_nodes=bn,
        disp_mask=np.ones((bn.size, 3), dtype=bool),
        disp_vals=np.zeros((bn.size, 3)),
        force_faces=topf,
        force=np.tile([1e5, -5e4, 2e6], (topf.size, 1)),
    )
    uu, _ = vem_lin_elast(
        g, C, bc, load=lambda x: np.tile([100.0, 200.0, -3000.0],
                                         (x.shape[0], 1))
    )
    return uu


def main() -> int:
    failed = 0
    for icase, label in ((1, "regular grid"), (2, "twisted grid")):
        path = HERE / f"ref_p7_vem{icase}.mat"
        if not path.exists():
            print(f"{path.name}: missing (run dump_phase7_reference.m)")
            return 1
        ref = scipy.io.loadmat(path, squeeze_me=True)
        uu = run_case(ref)
        ml = np.atleast_2d(np.asarray(ref["uu"], float))
        scale = np.abs(ml).max()
        diff = np.abs(uu - ml).max()
        ok = diff <= 1e-8 * scale
        print(f"  {'ok  ' if ok else 'FAIL'} VEM displacements ({label}): "
              f"max abs diff = {diff:.3e} (rel {diff / scale:.2e})")
        failed += 0 if ok else 1
    print("\nVEM elasticity matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
