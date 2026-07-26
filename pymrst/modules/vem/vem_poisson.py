"""First-order virtual element method for the pressure equation (MRST `vem`).

The virtual element method discretises `-div(K grad p) = q` on general polygonal
cells without ever forming the (unknown, non-polynomial) local basis functions.
For each cell it splits the local stiffness into a *consistency* part - exact for
linear pressures, built from the projection `Pi` onto linear polynomials - and a
*stability* part that controls the non-polynomial remainder and vanishes on
linear fields. Its defining property, the one this validates, is that it
reproduces a linear pressure field exactly on *any* grid (the patch test), which
TPFA does not.

This is the lowest-order nodal (H1-conforming) VEM: pressure unknowns live at the
grid nodes. It is the standard formulation (Beirao da Veiga et al.); MRST's
`computeVirtualIP` is the *mixed* flux-VEM that feeds the hybrid solver, a
separate and much larger assembly.
"""

from typing import Dict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _cell_nodes_ccw(g, cell: int) -> np.ndarray:
    """A convex cell's boundary node indices, ordered counter-clockwise."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
    nodes = np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
    xy = np.asarray(g.nodes.coords, dtype=float)[nodes][:, :2]
    ang = np.arctan2(xy[:, 1] - xy[:, 1].mean(), xy[:, 0] - xy[:, 0].mean())
    return nodes[np.argsort(ang)]


def _local_stiffness(verts: np.ndarray, perm: float) -> np.ndarray:
    """Lowest-order VEM local stiffness for a CCW polygon `verts` (n x 2)."""
    n = verts.shape[0]
    xE = verts.mean(axis=0)
    hE = np.max(np.linalg.norm(verts[:, None] - verts[None, :], axis=2))

    D = np.column_stack([np.ones(n),
                         (verts[:, 0] - xE[0]) / hE,
                         (verts[:, 1] - xE[1]) / hE])          # n x 3

    B = np.zeros((3, n))
    B[0, :] = 1.0 / n
    for i in range(n):
        e_prev = verts[i] - verts[(i - 1) % n]
        e_next = verts[(i + 1) % n] - verts[i]
        # outward normal * edge length for a CCW polygon is [dy, -dx]
        normal = 0.5 * (np.array([e_prev[1], -e_prev[0]])
                        + np.array([e_next[1], -e_next[0]]))
        B[1, i] = normal[0] / hE
        B[2, i] = normal[1] / hE

    G = B @ D
    Pistar = np.linalg.solve(G, B)                             # 3 x n
    Pi = D @ Pistar                                            # n x n
    Gt = G.copy()
    Gt[0, :] = 0.0
    consistency = Pistar.T @ Gt @ Pistar
    stability = (np.eye(n) - Pi).T @ (np.eye(n) - Pi)
    return perm * (consistency + stability)


def vem_poisson(g, rock, bc_pressure: Dict[int, float]) -> np.ndarray:
    """Solve `-div(K grad p) = 0` by first-order nodal VEM with Dirichlet node
    pressures `bc_pressure` (one per boundary node). Returns nodal pressures."""
    perm = np.asarray(rock.perm, dtype=float).ravel()
    if perm.size == 1:
        perm = np.full(g.cells.num, perm.item())
    coords = np.asarray(g.nodes.coords, dtype=float)

    nn = g.nodes.num
    A = sp.lil_array((nn, nn))
    for c in range(g.cells.num):
        nodes = _cell_nodes_ccw(g, c)
        Ke = _local_stiffness(coords[nodes][:, :2], float(perm[c]))
        for a in range(nodes.size):
            for b in range(nodes.size):
                A[nodes[a], nodes[b]] += Ke[a, b]

    A = A.tocsr()
    rhs = np.zeros(nn)
    fixed = np.zeros(nn, dtype=bool)
    p = np.zeros(nn)
    for node, val in bc_pressure.items():
        fixed[int(node)] = True
        p[int(node)] = val

    free = ~fixed
    rhs_free = -A[np.ix_(free, fixed)] @ p[fixed]
    p[free] = spsolve(A[np.ix_(free, free)].tocsc(), rhs_free)
    return p


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY

    def _twisted_grid(nx=6, amp=0.3):
        g = cart_grid([nx, nx], [float(nx), float(nx)])
        c = np.asarray(g.nodes.coords, dtype=float)
        x, y, L = c[:, 0].copy(), c[:, 1].copy(), float(nx)
        c[:, 0] += amp * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
        c[:, 1] += 0.6 * amp * np.sin(2 * np.pi * x / L) * np.sin(np.pi * y / L)
        g.nodes.coords = c
        return compute_geometry(g)

    def _boundary_nodes(g):
        neigh = np.asarray(g.faces.neighbors)
        node_pos = np.asarray(g.faces.nodePos)
        fnodes = np.asarray(g.faces.nodes)
        bnd_faces = np.flatnonzero(np.any(neigh < 0, axis=1))
        return np.unique(np.concatenate(
            [fnodes[node_pos[f]:node_pos[f + 1]] for f in bnd_faces]))

    def test_vem_reproduces_linear_field_on_twisted_grid():
        """The VEM patch test: a linear pressure field is recovered exactly at
        every node on a smoothly-twisted (non-orthogonal) grid."""
        g = _twisted_grid()
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        coords = np.asarray(g.nodes.coords)
        lin = lambda P: 3.0 * P[:, 0] + 2.0 * P[:, 1] + 5.0
        bnd = _boundary_nodes(g)
        bc = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in bnd}
        p = vem_poisson(g, rock, bc)
        err = np.max(np.abs(p - lin(coords)))
        assert err < 1e-9, err
        print(f"ok test_vem_reproduces_linear_field_on_twisted_grid "
              f"(max err {err:.2e})")

    def test_local_stiffness_is_spd_and_kills_constants():
        """Each cell matrix is symmetric positive-semidefinite and a constant
        nodal field produces no flux (row sums zero)."""
        g = _twisted_grid(4)
        coords = np.asarray(g.nodes.coords)
        for c in range(g.cells.num):
            nodes = _cell_nodes_ccw(g, c)
            Ke = _local_stiffness(coords[nodes][:, :2], 1.0)
            assert np.allclose(Ke, Ke.T, atol=1e-10), "not symmetric"
            assert np.allclose(Ke @ np.ones(nodes.size), 0.0, atol=1e-9), \
                "constant field must give zero"
            assert np.min(np.linalg.eigvalsh(Ke)) > -1e-10, "not PSD"
        print("ok test_local_stiffness_is_spd_and_kills_constants")

    def test_uniform_pressure_stays_uniform():
        g = _twisted_grid(5)
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        bc = {int(nd): 42.0 for nd in _boundary_nodes(g)}
        p = vem_poisson(g, rock, bc)
        assert np.allclose(p, 42.0, atol=1e-8), np.abs(p - 42.0).max()
        print("ok test_uniform_pressure_stays_uniform")

    test_vem_reproduces_linear_field_on_twisted_grid()
    test_local_stiffness_is_spd_and_kills_constants()
    test_uniform_pressure_stays_uniform()
    print("\nAll tests passed!")
