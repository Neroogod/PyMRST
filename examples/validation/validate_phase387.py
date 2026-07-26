"""Phase 387: MPFA-O assembly and the non-K-orthogonal linear patch test.

The patch test is deliberately discriminating: both schemes solve the same
Dirichlet problem on a smoothly twisted grid with a full permeability tensor.
MPFA must reproduce the imposed linear pressure to roundoff; TPFA must not.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.mpfa import compute_multipoint_trans, incomp_mpfa
from pymrst.params.rock import make_rock
from pymrst.solvers.compute_trans import compute_trans


def _twisted_grid(n=5, amplitude=0.25):
    length = float(n)
    g = cart_grid([n, n], [length, length])
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y = c[:, 0].copy(), c[:, 1].copy()
    c[:, 0] += amplitude*np.sin(np.pi*x/length)*np.sin(np.pi*y/length)
    c[:, 1] += 0.7*amplitude*np.sin(2*np.pi*x/length)*np.sin(np.pi*y/length)
    g.nodes.coords = c
    return compute_geometry(g)


def _linear(points):
    return 3.0*points[:, 0] + 2.0*points[:, 1] + 5.0


def _tpfa_dirichlet(g, rock, boundary_pressure):
    """Small direct TPFA assembly used only for the cross-scheme patch test."""
    half_t = compute_trans(g, rock)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    internal = np.all(neigh >= 0, axis=1)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    cellno = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))

    inv_t = np.bincount(cell_faces, weights=1.0/half_t,
                        minlength=g.faces.num)
    face_t = np.zeros(g.faces.num)
    face_t[internal] = 1.0/inv_t[internal]

    A = sp.lil_array((g.cells.num, g.cells.num))
    rhs = np.zeros(g.cells.num)
    for face, (left, right) in enumerate(neigh):
        if left >= 0 and right >= 0:
            trans = face_t[face]
            A[left, left] += trans
            A[left, right] -= trans
            A[right, right] += trans
            A[right, left] -= trans
        else:
            cell = left if left >= 0 else right
            row = np.flatnonzero((cell_faces == face) & (cellno == cell))[0]
            trans = half_t[row]
            A[cell, cell] += trans
            rhs[cell] += trans*boundary_pressure[face]
    return spsolve(sp.csr_array(A), rhs)


def main():
    g = _twisted_grid()
    # [kxx, kxy, kyy]: positive-definite, homogeneous full tensor.
    perm = np.tile([2.0, 0.6, 1.0], (g.cells.num, 1))*1e-13
    rock = make_rock(g, perm, 0.2)
    boundary = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
    bc = {int(f): float(_linear(g.faces.centroids[f:f + 1])[0])
          for f in boundary}
    exact = _linear(np.asarray(g.cells.centroids))

    operators = compute_multipoint_trans(g, rock)
    mpfa = incomp_mpfa(g, operators, bc)
    tpfa = _tpfa_dirichlet(g, rock, bc)
    mpfa_error = float(np.max(np.abs(mpfa - exact)))
    tpfa_error = float(np.max(np.abs(tpfa - exact)))

    constant_bc = {int(f): 100.0 for f in boundary}
    constant = incomp_mpfa(g, operators, constant_bc)
    constant_error = float(np.max(np.abs(constant - 100.0)))

    print("Phase 387: MPFA on a non-K-orthogonal grid")
    print(f"  MPFA linear patch error: {mpfa_error:.3e}")
    print(f"  TPFA linear patch error: {tpfa_error:.3e}")
    print(f"  MPFA constant patch error: {constant_error:.3e}")

    failed = (mpfa_error >= 1e-10 or tpfa_error <= 1e-3
              or constant_error >= 1e-10)
    print("\nPhase 387 passed." if not failed else "\nPhase 387 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
