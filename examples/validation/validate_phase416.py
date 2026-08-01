"""Phase 416: 3-D k=2 incompVEM standard BC + Neumann against MRST.

Pins the PyMRST 3-D k=2 standard boundary-condition path - per-face-constant
Dirichlet on left/right, a prescribed Neumann flux on the bottom, natural
no-flow elsewhere - against MRST's `incompVEM` on a planar-faced grid. This
validates the Neumann RHS at the face-moment dof and Dirichlet imposition over
the node/edge/face dofs.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.types import BoundaryCondition
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p416_incompvem_3d_k2_bc.mat")


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _rel(a, b):
    return np.max(np.abs(a - b)), max(np.max(np.abs(b)), 1e-300)


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase416_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_vem_geometry(compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0])))
    perm = np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0) * DARCY
    S = compute_virtual_ip(g, make_rock(g, perm, 0.2), 2)

    fc = np.asarray(g.faces.centroids)
    tol = 1e-6
    left = np.flatnonzero(np.abs(fc[:, 0] - fc[:, 0].min()) < tol)
    right = np.flatnonzero(np.abs(fc[:, 0] - fc[:, 0].max()) < tol)
    bottom = np.flatnonzero(np.abs(fc[:, 2] - fc[:, 2].min()) < tol)
    bc = BoundaryCondition()
    bc.face = np.concatenate([left, right, bottom])
    bc.type = ["pressure"] * left.size + ["pressure"] * right.size \
        + ["flux"] * bottom.size
    bc.value = np.concatenate([np.ones(left.size), np.zeros(right.size),
                               np.full(bottom.size, 0.3 * DARCY)])

    state = incomp_vem(g, S, bc=bc)

    failed = 0
    print("incompVEM (k=2, 3-D planar, standard BC: Dirichlet + Neumann) vs MRST:")
    for name, a, key in (("node pressures", state["node_pressure"], "nodePressure"),
                         ("edge pressures", state["edge_pressure"], "edgePressure"),
                         ("face pressures", state["face_pressure"], "facePressure"),
                         ("cell pressures", state["pressure"], "cellPressure")):
        err, scale = _rel(a, ref[key].ravel())
        failed += _check(f"{name} match", err < 1e-9 * scale,
                         f"max|d| = {err:.3e} (scale {scale:.2e})")
    err, scale = _rel(state["flux"], ref["flux"].ravel())
    failed += _check("all face fluxes match (incl. boundary)", err < 1e-9 * scale,
                     f"max|dq| = {err:.3e} (scale {scale:.2e})")

    print("\n3-D k=2 standard-BC incompVEM matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
