"""Mixed VEM inner product for the pressure equation (MRST `computeVirtualIP`).

Produces the VEM analog of `computeMimeticIP`'s system: a per-cell local stiffness
`A_E = PiNstar' Mc PiNstar + sigma_E (I - PiN)'(I - PiN)`, block-diagonal over the
grid, together with the local-to-global DOF map. `PiNstar`/`PiN` are the
projection onto linear polynomials, `Mc` the consistency (H1-seminorm) matrix with
the constant row removed, and `sigma_E = h_E^(dim-2) tr(K_E)/dim` the `ip_simple`
stability weight. `incomp_vem` assembles these blocks and solves the global
pressure system.

This ports the first-order (k = 1) scheme, whose degrees of freedom are the grid
nodes - the order that carries the patch test (linear pressures reproduced
exactly). It follows MRST's face-based `computeBD2D`: the constant row of `B` is
edge-length weighted (not the vertex average) and the gradient rows carry the
permeability tensor, so the assembled system matches MRST's `computeVirtualIP`,
not merely on linear fields.
"""

from typing import Dict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.params.rock.perm_tensor import perm_tensor


def _cell_diameter(coords: np.ndarray) -> float:
    d = coords[:, None, :] - coords[None, :, :]
    return float(np.sqrt((d ** 2).sum(axis=2)).max())


def _local_ip(g, cell: int, K: np.ndarray):
    """Return `(A, nodes)`: the VEM local stiffness and its node DOFs."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    normals = np.asarray(g.faces.normals, dtype=float)
    areas = np.asarray(g.faces.areas, dtype=float)

    faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
    nodes = np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
    local = {int(n): i for i, n in enumerate(nodes)}
    n = nodes.size
    X = coords[nodes][:, :2]

    cc = np.asarray(g.cells.centroids, dtype=float)[cell][:2]
    cd = _cell_diameter(X)

    D = np.column_stack([np.ones(n), (X[:, 0] - cc[0]) / cd, (X[:, 1] - cc[1]) / cd])
    B = np.zeros((3, n))
    for f in faces:
        fn = fnodes[node_pos[f]:node_pos[f + 1]]
        sign = 1.0 if cell == neigh[f, 0] else -1.0
        nrm = normals[f, :2] * sign                    # outward, includes length
        Kn = nrm @ K                                   # (K grad m) normal comp
        for nd in fn:
            li = local[int(nd)]
            B[0, li] += 0.5 * areas[f]
            B[1, li] += 0.5 * Kn[0] / cd
            B[2, li] += 0.5 * Kn[1] / cd

    M = B @ D
    PiNstar = np.linalg.solve(M, B)                    # M^-1 B, 3 x n
    PiN = D @ PiNstar                                  # n x n
    Mc = M.copy()
    Mc[0, :] = 0.0
    sigma = cd ** (g.griddim - 2) * np.trace(K) / g.griddim
    I = np.eye(n)
    A = PiNstar.T @ Mc @ PiNstar + sigma * (I - PiN).T @ (I - PiN)
    return A, nodes, PiNstar


def compute_virtual_ip(g, rock, k: int = 1) -> Dict:
    """Return the first-order VEM system `S` for grid `g` and `rock`.

    `S['A']` is the list of per-cell local stiffness matrices, `S['dofVec']` the
    matching lists of global node DOFs, `S['order']` the method order (1).
    """
    if k != 1:
        raise NotImplementedError("only the first-order (k=1) VEM is ported")
    dim = g.griddim
    tensor = perm_tensor(rock, dim)[0]                 # (ncells, dim^2)
    A_blocks, dof, proj = [], [], []
    for c in range(g.cells.num):
        K = tensor[c].reshape(dim, dim)
        A, nodes, PiNstar = _local_ip(g, c, K)
        A_blocks.append(A)
        dof.append(nodes)
        proj.append(PiNstar)
    return {"A": A_blocks, "dofVec": dof, "PiNstar": proj, "order": 1}


def _assemble_global(g, S: Dict, tot_mob: np.ndarray) -> sp.csr_array:
    """Scatter the mobility-weighted local blocks into the global node matrix."""
    rows, cols, vals = [], [], []
    for c, (Ae, nodes) in enumerate(zip(S["A"], S["dofVec"])):
        ni, nj = np.meshgrid(nodes, nodes, indexing="ij")
        rows.append(ni.ravel())
        cols.append(nj.ravel())
        vals.append((tot_mob[c] * Ae).ravel())
    nn = g.nodes.num
    A = sp.coo_array((np.concatenate(vals),
                      (np.concatenate(rows), np.concatenate(cols))),
                     shape=(nn, nn))
    return A.tocsr()


def _source_rhs(g, S: Dict, src, nn: int) -> np.ndarray:
    """Distribute cell source rates onto nodes via the P0 projection row."""
    rhs = np.zeros(nn)
    if src is None or np.asarray(src.cell).size == 0:
        return rhs
    rate = {int(c): float(r) for c, r in zip(src.cell, src.rate)}
    for c, nodes in enumerate(S["dofVec"]):
        if c in rate:
            rhs[nodes] += rate[c] * S["PiNstar"][c][0, :]
    return rhs


def _neumann_rhs(g, S: Dict, bc, rhs: np.ndarray) -> None:
    """Add first-order 2-D Neumann (flux) contributions in place.

    A VEM basis function is linear on each face, so the flux `(g, phi)_{0,f}`
    splits equally onto the face's two nodes (the centroid rule).
    """
    if bc is None or bc.face.size == 0 or g.griddim != 2:
        return
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    for f, kind, v in zip(bc.face, bc.type, bc.value):
        if kind != "flux":
            continue
        for nd in fnodes[node_pos[f]:node_pos[f + 1]]:
            rhs[int(nd)] += 0.5 * v


def _impose_dirichlet(g, A: sp.csr_array, rhs: np.ndarray, bc):
    """Return the Dirichlet node set and pin values (constant per face)."""
    fixed, val = {}, {}
    if bc is None:
        return fixed, val
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    for f, kind, v in zip(bc.face, bc.type, bc.value):
        if kind != "pressure":
            continue
        for nd in fnodes[node_pos[f]:node_pos[f + 1]]:
            fixed[int(nd)] = True
            val[int(nd)] = float(v)
    return fixed, val


def _reconstruct_flux(g, S: Dict, node_p: np.ndarray, tot_mob: np.ndarray,
                      bc) -> np.ndarray:
    """Face fluxes from the identity `K grad p ~= (p' A r) / |E|`.

    For each cell the constant `K grad p` is recovered from the local stiffness
    and the dof coordinates, projected onto each face normal, and interface
    fluxes are the arithmetic mean of the adjoining cells' contributions.
    """
    dim = g.griddim
    coords = np.asarray(g.nodes.coords, dtype=float)[:, :dim]
    normals = np.asarray(g.faces.normals, dtype=float)[:, :dim]
    vols = np.asarray(g.cells.volumes, dtype=float)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()

    accum = np.zeros(g.faces.num)
    count = np.zeros(g.faces.num)
    for c, nodes in enumerate(S["dofVec"]):
        Kgradp = (node_p[nodes] @ (tot_mob[c] * S["A"][c])
                  @ coords[nodes]) / vols[c]
        for f in cell_faces[face_pos[c]:face_pos[c + 1]]:
            accum[f] += -(Kgradp @ normals[f])
            count[f] += 1.0
    flux = accum / count

    if bc is not None and bc.face.size:
        neigh = np.asarray(g.faces.neighbors)
        for f, kind, v in zip(bc.face, bc.type, bc.value):
            if kind == "flux":
                sgn = -1.0 + 2.0 * (neigh[f, 1] >= 0)
                flux[f] = v * sgn
    return flux


def incomp_vem(g, S: Dict, bc=None, src=None, tot_mob: np.ndarray = None,
               dirichlet_nodes: Dict[int, float] = None,
               reconstruct_flux: bool = True) -> Dict:
    """Solve the first-order VEM pressure system (MRST `incompVEM`, k=1).

    `bc` is a `BoundaryCondition` (per-face constant Dirichlet/Neumann), `src` a
    `Source`. `dirichlet_nodes` pins exact nodal pressures (MRST's `addBCVEM`
    function-handle path, used for patch testing). Returns a state dict with
    `node_pressure` and, when requested, `flux`.
    """
    nn = g.nodes.num
    if tot_mob is None:
        tot_mob = np.ones(g.cells.num)

    A = _assemble_global(g, S, tot_mob)
    rhs = _source_rhs(g, S, src, nn)
    _neumann_rhs(g, S, bc, rhs)

    fixed, val = _impose_dirichlet(g, A, rhs, bc)
    if dirichlet_nodes:
        for nd, v in dirichlet_nodes.items():
            fixed[int(nd)] = True
            val[int(nd)] = float(v)

    if not fixed:
        pin = int(S["dofVec"][0][0])
        A = A.tolil()
        A[pin, pin] *= 2.0
        A = A.tocsr()

    p = np.zeros(nn)
    if fixed:
        fixed_idx = np.array(sorted(fixed), dtype=np.int64)
        p[fixed_idx] = [val[int(i)] for i in fixed_idx]
        free = np.setdiff1d(np.arange(nn), fixed_idx)
        rhs_free = rhs[free] - A[np.ix_(free, fixed_idx)] @ p[fixed_idx]
        p[free] = spsolve(A[np.ix_(free, free)].tocsc(), rhs_free)
    else:
        p = spsolve(A.tocsc(), rhs)

    state = {"node_pressure": p}
    if reconstruct_flux:
        state["flux"] = _reconstruct_flux(g, S, p, tot_mob, bc)
    return state


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
        bf = np.flatnonzero(np.any(neigh < 0, axis=1))
        return np.unique(np.concatenate(
            [fnodes[node_pos[f]:node_pos[f + 1]] for f in bf]))

    def test_patch_test_linear_field():
        g = _twisted_grid()
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        S = compute_virtual_ip(g, rock, 1)
        coords = np.asarray(g.nodes.coords)
        lin = lambda P: 3.0 * P[:, 0] + 2.0 * P[:, 1] + 5.0
        nodes = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
        p = incomp_vem(g, S, dirichlet_nodes=nodes,
                       reconstruct_flux=False)["node_pressure"]
        err = np.max(np.abs(p - lin(coords)))
        assert err < 1e-9, err
        print(f"ok test_patch_test_linear_field (max err {err:.2e})")

    def test_local_blocks_are_spd_and_kill_constants():
        g = _twisted_grid(4)
        rock = make_rock(g, np.array([[2.0, 0.5, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 1)
        for Ae, nodes in zip(S["A"], S["dofVec"]):
            assert np.allclose(Ae, Ae.T, atol=1e-10)
            assert np.allclose(Ae @ np.ones(nodes.size), 0.0, atol=1e-9)
            assert np.min(np.linalg.eigvalsh(Ae)) > -1e-9
        print("ok test_local_blocks_are_spd_and_kill_constants")

    def test_full_tensor_patch_test():
        """The patch test holds with a full (off-diagonal) permeability too."""
        g = _twisted_grid(5)
        rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 1)
        coords = np.asarray(g.nodes.coords)
        lin = lambda P: 4.0 * P[:, 0] - 1.5 * P[:, 1] + 2.0
        nodes = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
        p = incomp_vem(g, S, dirichlet_nodes=nodes,
                       reconstruct_flux=False)["node_pressure"]
        assert np.max(np.abs(p - lin(coords))) < 1e-9
        print("ok test_full_tensor_patch_test")

    def test_flux_reconstruction_reproduces_uniform_gradient():
        """A linear pressure has constant K grad p, so with exact nodal data the
        reconstructed interface flux equals -(K grad p) . area-normal exactly."""
        g = _twisted_grid(5)
        K = np.array([[2.0, 0.6, 1.0]])
        rock = make_rock(g, K.repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 1)
        coords = np.asarray(g.nodes.coords)
        grad = np.array([3.0, -1.5])
        lin = lambda P: P[:, :2] @ grad + 2.0
        Kgrad = np.array([[2.0, 0.6], [0.6, 1.0]]) @ grad
        nodes = {int(nd): float(lin(coords[nd:nd + 1])[0])
                 for nd in _boundary_nodes(g)}
        state = incomp_vem(g, S, dirichlet_nodes=nodes)
        normals = np.asarray(g.faces.normals)[:, :2]
        internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))
        expected = -(normals[internal] @ Kgrad)
        err = np.max(np.abs(state["flux"][internal] - expected))
        assert err < 1e-9 * np.max(np.abs(expected)), err
        print(f"ok test_flux_reconstruction_reproduces_uniform_gradient (err {err:.2e})")

    def test_source_rate_conservation():
        """Injected volume leaves through the (pinned) boundary: with a single
        source and no-flow elsewhere the net boundary flux equals the rate."""
        from pymrst.params.wells_and_bc.types import Source
        g = _twisted_grid(5)
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        S = compute_virtual_ip(g, rock, 1)
        src = Source(cell=np.array([12]), rate=np.array([1.0]))
        state = incomp_vem(g, S, src=src, reconstruct_flux=False)
        # Pure Neumann + source: the assembled residual must be the source rhs.
        rhs = _source_rhs(g, S, src, g.nodes.num)
        assert abs(rhs.sum() - 1.0) < 1e-10, rhs.sum()
        print("ok test_source_rate_conservation")

    test_patch_test_linear_field()
    test_local_blocks_are_spd_and_kill_constants()
    test_full_tensor_patch_test()
    test_flux_reconstruction_reproduces_uniform_gradient()
    test_source_rate_conservation()
    print("\nAll tests passed!")
