"""Phase 422: 3-D k=1 incompVEM standard BC + Neumann against MRST.

Pins PyMRST's 3-D k=1 prescribed-Neumann path (flux distributed to the face
nodes by the face P0 projection) against MRST's `incompVEM`: per-face-constant
Dirichlet on left/right, a prescribed Neumann flux on the bottom, natural
no-flow elsewhere, on a planar-faced grid where 3-D k=1 is exact.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip, incomp_vem
from pymrst.modules.vem.utils.vem_geometry import compute_vem_geometry
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc.types import BoundaryCondition
from pymrst.utils.units import DARCY

REF = Path(__file__).with_name("ref_p422_incompvem_3d_k1_bc.mat")


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _rel(a, b):
    return np.max(np.abs(a - b)), max(np.max(np.abs(b)), 1e-300)


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase422_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    p_ref = np.asarray(ref["nodePressure"], dtype=float).ravel()
    flux_ref = np.asarray(ref["flux"], dtype=float).ravel()

    g = compute_vem_geometry(compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0])))
    perm = np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0) * DARCY
    S = compute_virtual_ip(g, make_rock(g, perm, 0.2), 1)

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
    print("incompVEM (k=1, 3-D planar, Dirichlet + prescribed Neumann) vs MRST:")
    perr, ps = _rel(state["node_pressure"], p_ref)
    failed += _check("node pressures match", perr < 1e-9 * ps,
                     f"max|dp| = {perr:.3e} (scale {ps:.2e})")
    ferr, fs = _rel(state["flux"], flux_ref)
    failed += _check("all face fluxes match (incl. Neumann)", ferr < 1e-9 * fs,
                     f"max|dq| = {ferr:.3e} (scale {fs:.2e})")

    print("\n3-D k=1 Neumann incompVEM matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
