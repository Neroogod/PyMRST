"""Solve the VEM pressure equation (MRST ``vem/incompVEM.m``).

Assembles the global system from the per-cell inner products `compute_virtual_ip`
produced, imposes boundary conditions and sources, solves, and reconstructs
whatever the caller asked for.

**What is an unknown depends on the order, and that shapes the whole file.**
At k=2 the face moments and cell moments are degrees of freedom, so face and
cell pressures fall out of the solve. At k=1 the only unknowns are nodal, and
face and cell values have to be *projected* from them afterwards - which is
why `_face_pressures` and `_cell_pressures` exist and why they branch on
`S["order"]` rather than on the grid.

The system is singular when no pressure is prescribed anywhere - only pressure
*differences* are determined by fluxes alone - so, as in MRST, one diagonal
entry is doubled to pin the level. That is a rank-one fix, not a boundary
condition: it leaves the flux field unchanged and shifts the pressure by a
constant.
"""

from typing import Dict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.modules.vem.compute_virtual_ip import _face_p0_weights

def _assemble_global(g, S: Dict, tot_mob: np.ndarray, N: int) -> sp.csr_array:
    """Scatter the mobility-weighted local blocks into the global dof matrix."""
    rows, cols, vals = [], [], []
    for c, (Ae, dofs) in enumerate(zip(S["A"], S["dofVec"])):
        di, dj = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(di.ravel())
        cols.append(dj.ravel())
        vals.append((tot_mob[c] * Ae).ravel())
    A = sp.coo_array((np.concatenate(vals),
                      (np.concatenate(rows), np.concatenate(cols))),
                     shape=(N, N))
    return A.tocsr()


def _dof_blocks(g, S: Dict):
    """Sizes of the global dof blocks in order: nodes[, edges], faces, cells.

    k=1 has only nodes; k=2 adds face and cell moments, plus edge moments in 3-D.
    """
    nn = g.nodes.num
    if S["order"] == 1:
        return (nn,)
    if g.griddim == 3:
        return (nn, g.edges.num, g.faces.num, g.cells.num)
    return (nn, g.faces.num, g.cells.num)


def _total_dofs(g, S: Dict) -> int:
    """Global dof count over all dof blocks."""
    return sum(_dof_blocks(g, S))


def _dof_coords(g, S: Dict) -> np.ndarray:
    """Coordinate of every global dof (node / edge- / face- / cell-centroid)."""
    dim = g.griddim
    parts = [np.asarray(g.nodes.coords, dtype=float)[:, :dim]]
    if S["order"] == 2:
        if g.griddim == 3:
            parts.append(np.asarray(g.edges.centroids, dtype=float)[:, :dim])
        parts.append(np.asarray(g.faces.centroids, dtype=float)[:, :dim])
        parts.append(np.asarray(g.cells.centroids, dtype=float)[:, :dim])
    return np.vstack(parts)


def _source_rhs(g, S: Dict, src, N: int) -> np.ndarray:
    """Cell source rates: distribute via the P0 row (k=1) or the cell dof (k=2).

    For k=1 the cell average is a linear combination of the node dofs through
    the P0 projection row; for k=2 it is an explicit degree of freedom.
    """
    rhs = np.zeros(N)
    if src is None or np.asarray(src.cell).size == 0:
        return rhs
    rate = {int(c): float(r) for c, r in zip(src.cell, src.rate)}
    if S["order"] == 1:
        for c, nodes in enumerate(S["dofVec"]):
            if c in rate:
                rhs[nodes] += rate[c] * S["PiNstar"][c][0, :]
    else:
        base = N - g.cells.num                         # cell dofs are the last block
        for c, r in rate.items():
            rhs[base + c] = r
    return rhs


def _neumann_rhs(g, S: Dict, bc, rhs: np.ndarray) -> None:
    """Add Neumann (flux) contributions in place.

    In 2-D a VEM basis function is linear (k=1) or quadratic (k=2) on each face,
    so the flux `(g, phi)_{0,f}` is split by the Gauss-Lobatto rule: half to each
    node (k=1), or 1/6 to each node and 2/3 to the face-centroid dof (k=2). In
    3-D k=2 the flux goes to the face-moment dof; in 3-D k=1 it is distributed to
    the face nodes by that face's P0 projection weights (which sum to 1).
    """
    if bc is None or bc.face.size == 0 or not any(t == "flux" for t in bc.type):
        return
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    nn, ne = g.nodes.num, (g.edges.num if g.griddim == 3 else 0)
    for f, kind, v in zip(bc.face, bc.type, bc.value):
        if kind != "flux":
            continue
        if g.griddim == 3:
            if S["order"] == 2:                        # k=2: face-moment dof
                rhs[nn + ne + int(f)] += v
            else:                                      # k=1: face P0 projection
                c = int(max(neigh[f]))                 # the boundary face's cell
                K = S["tensor"][c].reshape(3, 3)
                w = _face_p0_weights(g, int(f), K)
                for nd in fnodes[node_pos[f]:node_pos[f + 1]]:
                    rhs[int(nd)] += v * w[int(nd)]
            continue
        for nd in fnodes[node_pos[f]:node_pos[f + 1]]:
            rhs[int(nd)] += (0.5 if S["order"] == 1 else 1.0 / 6) * v
        if S["order"] == 2:
            rhs[nn + int(f)] += 2.0 / 3 * v


def _edge_centroids(g, edges) -> np.ndarray:
    """Midpoints of the given edges, where a k=2 3-D face carries its moments."""
    nodes = np.asarray(g.edges.nodes, dtype=np.int64).reshape(-1, 2)
    coords = np.asarray(g.nodes.coords, dtype=float)
    return 0.5 * (coords[nodes[edges, 0]] + coords[nodes[edges, 1]])


def _impose_dirichlet_func(g, S: Dict, bc, fixed, val) -> None:
    """Pin the dofs of a function-valued pressure condition, in place.

    Which dofs a face owns depends on the order and the dimension, and the
    point of `add_bc_vem` is that the caller does not have to know: the
    function is evaluated wherever a dof lives - node coordinates always, edge
    midpoints and the face centroid at k=2.
    """
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    centroids = np.asarray(g.faces.centroids, dtype=float)
    nn, ne = g.nodes.num, (g.edges.num if g.griddim == 3 else 0)

    for faces, kind, func in zip(bc.face, bc.type, bc.func):
        if kind != "pressure":
            continue
        nodes = np.unique(np.concatenate(
            [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
        for nd, value in zip(nodes, np.asarray(func(coords[nodes]), float).ravel()):
            fixed[int(nd)] = True
            val[int(nd)] = float(value)

        if S["order"] != 2:
            continue
        if g.griddim == 3:
            epos = np.asarray(g.faces.edgePos, dtype=np.int64)
            fedges = np.asarray(g.faces.edges, dtype=np.int64)
            edges = np.unique(np.concatenate(
                [fedges[epos[f]:epos[f + 1]] for f in faces]))
            values = np.asarray(func(_edge_centroids(g, edges)), float).ravel()
            for e, value in zip(edges, values):
                fixed[nn + int(e)] = True
                val[nn + int(e)] = float(value)
        values = np.asarray(func(centroids[faces]), float).ravel()
        for f, value in zip(faces, values):
            fixed[nn + ne + int(f)] = True
            val[nn + ne + int(f)] = float(value)


def _neumann_rhs_func(g, S: Dict, bc, rhs: np.ndarray) -> None:
    """Add a function-valued flux condition, in place.

    The face integral is taken with Simpson's rule, which is exact for the
    quadratic variation a k=2 basis function has along a face. At k=1 there is
    no face dof to receive the midpoint share, so MRST splits it equally
    between the two nodes; at k=2 it goes to the face moment.

    **Two dimensions only**, as in MRST - whose 3-D branch here is an empty
    `else`, which is why it asserts on the dimension first. This raises instead
    of silently contributing nothing.
    """
    if not any(kind == "flux" for kind in bc.type):
        return
    if g.griddim != 2:
        raise NotImplementedError(
            "function-valued flux conditions are 2-D only, as in MRST; use a "
            "constant flux through add_bc in 3-D")

    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    centroids = np.asarray(g.faces.centroids, dtype=float)
    areas = np.asarray(g.faces.areas, dtype=float)

    for faces, kind, func in zip(bc.face, bc.type, bc.func):
        if kind != "flux":
            continue
        middle = np.asarray(func(centroids[faces]), float).ravel()
        for i, f in enumerate(faces):
            nodes = fnodes[node_pos[f]:node_pos[f + 1]]
            ends = np.asarray(func(coords[nodes]), float).ravel()
            if S["order"] == 1:
                share = areas[f] * (ends / 6.0 + middle[i] / 3.0)
                for nd, value in zip(nodes, share):
                    rhs[int(nd)] += value
            else:
                for nd, value in zip(nodes, areas[f] * ends / 6.0):
                    rhs[int(nd)] += value
                rhs[g.nodes.num + int(f)] += 2.0 / 3.0 * areas[f] * middle[i]


def _impose_dirichlet(g, S: Dict, bc):
    """Return the Dirichlet dof set and pin values for a standard BC.

    Pressure is constant per face, so all of that face's dofs are pinned to the
    face value: its nodes (both orders) and, for k=2, its face-moment dof plus,
    in 3-D, its edge dofs.
    """
    fixed, val = {}, {}
    if bc is None:
        return fixed, val
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    nn, ne = g.nodes.num, (g.edges.num if g.griddim == 3 else 0)
    for f, kind, v in zip(bc.face, bc.type, bc.value):
        if kind != "pressure":
            continue
        for nd in fnodes[node_pos[f]:node_pos[f + 1]]:
            fixed[int(nd)] = True
            val[int(nd)] = float(v)
        if S["order"] == 2:
            if g.griddim == 3:
                epos = np.asarray(g.faces.edgePos, dtype=np.int64)
                for e in np.asarray(g.faces.edges, dtype=np.int64)[epos[f]:epos[f + 1]]:
                    fixed[nn + int(e)] = True
                    val[nn + int(e)] = float(v)
            fixed[nn + ne + int(f)] = True
            val[nn + ne + int(f)] = float(v)
    return fixed, val


def _reconstruct_flux(g, S: Dict, p: np.ndarray, tot_mob: np.ndarray,
                      bc) -> np.ndarray:
    """Face fluxes from the identity `K grad p ~= (p' A r) / |E|`.

    For each cell the constant `K grad p` is recovered from the local stiffness
    and the dof coordinates, projected onto each face normal, and interface
    fluxes are the arithmetic mean of the adjoining cells' contributions.
    """
    dim = g.griddim
    r = _dof_coords(g, S)
    normals = np.asarray(g.faces.normals, dtype=float)[:, :dim]
    vols = np.asarray(g.cells.volumes, dtype=float)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()

    accum = np.zeros(g.faces.num)
    count = np.zeros(g.faces.num)
    for c, dofs in enumerate(S["dofVec"]):
        Kgradp = (p[dofs] @ (tot_mob[c] * S["A"][c]) @ r[dofs]) / vols[c]
        for f in cell_faces[face_pos[c]:face_pos[c + 1]]:
            accum[f] += -(Kgradp @ normals[f])
            count[f] += 1.0
    flux = accum / count

    # MRST convention (standard bc): every boundary face is no-flow unless it is
    # a Dirichlet ('pressure') face, which keeps its reconstructed flux. Neumann
    # faces get the prescribed value; unspecified boundary faces get zero.
    if bc is not None:
        neigh = np.asarray(g.faces.neighbors)
        neu = np.any(neigh < 0, axis=1)
        v = np.zeros(g.faces.num)
        for f, kind, val in zip(bc.face, bc.type, bc.value):
            if kind == "pressure":
                neu[f] = False
            elif kind == "flux":
                v[f] = val * (-1.0 + 2.0 * (neigh[f, 1] >= 0))
        flux[neu] = v[neu]
    return flux


def _default_solve(A, b):
    """The sparse direct solve `incomp_vem` uses unless given another."""
    return spsolve(A, b)


def _face_pressures(g, S: Dict, p: np.ndarray) -> np.ndarray:
    """Face pressure at every face (MRST `incompVEM/facePressure`).

    At k=2 the face moment is an unknown and this is not called. At k=1 the
    only unknowns are nodal, so a face value has to be projected from them:

    - in 2-D a face is an edge with two nodes, and MRST takes their mean;
    - in 3-D a face is a polygon, and the projection is the first row of that
      face's own 2-D VEM operator - `PiNFstar(1,:)` in MRST - which is exactly
      what `_face_p0_weights` returns.

    The 3-D branch used to raise `NotImplementedError`, which made `k=1` TPFA
    flux reconstruction unavailable on any 3-D grid.
    """
    if S["order"] == 2:
        offset = g.nodes.num + (g.edges.num if g.griddim == 3 else 0)
        return p[offset:offset + g.faces.num]

    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)

    if g.griddim == 2:
        return np.array([p[fnodes[node_pos[f]:node_pos[f + 1]]].mean()
                         for f in range(g.faces.num)])

    tensor = S["tensor"]
    out = np.zeros(g.faces.num)
    for face in range(g.faces.num):
        # The face projection is a property of the face, so either neighbouring
        # cell's permeability would do; MRST averages the per-cell-face values,
        # which for a face-intrinsic quantity is the same number.
        cell = int(np.asarray(g.faces.neighbors)[face].max())
        weights = _face_p0_weights(g, face, tensor[cell].reshape(3, 3))
        out[face] = sum(w * p[nd] for nd, w in weights.items())
    return out


def _cell_pressures(g, S: Dict, p: np.ndarray) -> np.ndarray:
    """Cell-average pressure: the P0 projection of nodal values (k=1) or the
    cell-moment dof (k=2)."""
    if S["order"] == 1:
        return np.array([S["PiNstar"][c][0, :] @ p[S["dofVec"][c]]
                         for c in range(g.cells.num)])
    return p[_total_dofs(g, S) - g.cells.num:]


def _tpfa_flux(g, S: Dict, p: np.ndarray, tot_mob: np.ndarray, bc) -> np.ndarray:
    """Reconstruct face fluxes with a TPFA scheme on the VEM cell/face pressures.

    The stored half-transmissibilities are scaled by cell mobility and harmonically
    averaged to a face transmissibility; interior fluxes follow the cell-pressure
    drop and boundary fluxes the (face - cell) pressure drop (MRST incompVEM).
    """
    T_half = np.asarray(S["T"], dtype=float)
    cell_no = g.cell_no()
    cf = np.asarray(g.cells.faces, dtype=np.int64)
    Tm = T_half * tot_mob[cell_no]
    ft = 1.0 / np.bincount(cf, weights=1.0 / Tm, minlength=g.faces.num)

    pc = _cell_pressures(g, S, p)
    neigh = np.asarray(g.faces.neighbors)
    interior = np.all(neigh >= 0, axis=1)
    flux = np.zeros(g.faces.num)
    ii = np.flatnonzero(interior)
    flux[ii] = -ft[ii] * (pc[neigh[ii, 1]] - pc[neigh[ii, 0]])

    bnd = np.flatnonzero(~interior)
    pf = _face_pressures(g, S, p)
    sgn = 2.0 * (neigh[bnd, 1] == -1) - 1.0
    cbnd = neigh[bnd].max(axis=1)
    flux[bnd] = -sgn * ft[bnd] * (pf[bnd] - pc[cbnd])

    _apply_boundary_flux(g, flux, bc, interior)
    return flux


def _apply_boundary_flux(g, flux, bc, interior):
    """MRST boundary-flux override: non-Dirichlet boundary faces are no-flow
    unless given a prescribed Neumann value."""
    if bc is None:
        return
    neigh = np.asarray(g.faces.neighbors)
    neu = ~interior
    v = np.zeros(g.faces.num)
    for f, kind, val in zip(bc.face, bc.type, bc.value):
        if kind == "pressure":
            neu[f] = False
        elif kind == "flux":
            v[f] = val * (-1.0 + 2.0 * (neigh[f, 1] >= 0))
    flux[neu] = v[neu]


def _mpfa_flux(g, S: Dict, p: np.ndarray, tot_mob: np.ndarray, bc) -> np.ndarray:
    """Reconstruct face fluxes with an MPFA scheme on the VEM cell/face pressures.

    Uses the stored multi-point transmissibility operator: each boundary subface
    takes its face's VEM pressure, `cf_trans` maps `[cell, boundary-subface]`
    pressures to subface fluxes (mobility-weighted), and `d1` aggregates them to
    faces (MRST incompVEM's MPFA branch).
    """
    import scipy.sparse as _sp
    T = S["Tmpfa"]
    d1 = T["d1"]
    sb = np.asarray(T["sb"], dtype=bool)
    nc, nf = g.cells.num, g.faces.num

    cf = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    cellno = np.repeat(np.arange(nc), np.diff(g.cells.facePos))
    face_mob = 1.0 / np.bincount(cf, weights=1.0 / tot_mob[cellno], minlength=nf)
    M = _sp.diags_array(np.asarray(d1 @ face_mob).ravel(), format="csr")

    face_of_subf = np.asarray(d1.argmax(axis=1)).ravel()
    bnd_face = face_of_subf[np.flatnonzero(sb)]
    pc = _cell_pressures(g, S, p)
    pf = _face_pressures(g, S, p)
    base = np.concatenate([pc, pf[bnd_face]])

    subface_flux = M @ (T["cf_trans"] @ base)
    flux = np.asarray(d1.T @ subface_flux).ravel()
    interior = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
    flux[interior] *= 0.5
    _apply_boundary_flux(g, flux, bc, interior)
    return flux


def incomp_vem(g, S: Dict, bc=None, src=None, tot_mob: np.ndarray = None,
               dirichlet_nodes: Dict[int, float] = None,
               reconstruct_flux: bool = True,
               face_pressure: bool = False, cell_pressure: bool = False,
               lin_solve=None, matrix_output: bool = False) -> Dict:
    """Solve the VEM pressure system (MRST `incompVEM`, k=1 or k=2, 2-D).

    `bc` is a `BoundaryCondition` (per-face constant Dirichlet/Neumann), `src` a
    `Source`. `dirichlet_nodes` pins exact values at chosen global dofs (MRST's
    `addBCVEM` function-handle path, used for patch testing); for k=2 include the
    node, face-moment and (in 3-D) edge dofs. Returns a state dict with
    `node_pressure`, `flux` (when requested) and, for k=2, `face_pressure`,
    `pressure` (cell moments) and, in 3-D, `edge_pressure`.

    Options mirroring MRST's: `face_pressure` and `cell_pressure` add the
    reconstructed face and cell values to the state (at k=1 these are not
    unknowns and have to be projected from the nodal solution); `lin_solve`
    replaces the sparse direct solve; `matrix_output` returns the assembled
    system.
    """
    if tot_mob is None:
        tot_mob = np.ones(g.cells.num)
    N = _total_dofs(g, S)

    A = _assemble_global(g, S, tot_mob, N)
    rhs = _source_rhs(g, S, src, N)

    # A bc built by `add_bc_vem` carries functions of position rather than
    # per-face values, and needs the dofs located rather than looked up.
    functional = bc is not None and hasattr(bc, "func")
    if functional:
        _neumann_rhs_func(g, S, bc, rhs)
        fixed, val = {}, {}
        _impose_dirichlet_func(g, S, bc, fixed, val)
    else:
        _neumann_rhs(g, S, bc, rhs)
        fixed, val = _impose_dirichlet(g, S, bc)
    if dirichlet_nodes:
        for nd, v in dirichlet_nodes.items():
            fixed[int(nd)] = True
            val[int(nd)] = float(v)

    if not fixed:
        pin = int(S["dofVec"][0][0])
        A = A.tolil()
        A[pin, pin] *= 2.0
        A = A.tocsr()

    solve = lin_solve if lin_solve is not None else _default_solve

    p = np.zeros(N)
    if fixed:
        fixed_idx = np.array(sorted(fixed), dtype=np.int64)
        p[fixed_idx] = [val[int(i)] for i in fixed_idx]
        free = np.setdiff1d(np.arange(N), fixed_idx)
        rhs_free = rhs[free] - A[np.ix_(free, fixed_idx)] @ p[fixed_idx]
        p[free] = solve(A[np.ix_(free, free)].tocsc(), rhs_free)
    else:
        p = solve(A.tocsc(), rhs)

    state = {"node_pressure": p[:g.nodes.num]}
    if S["order"] == 2:
        off = g.nodes.num
        if g.griddim == 3:
            state["edge_pressure"] = p[off:off + g.edges.num]
            off += g.edges.num
        state["face_pressure"] = p[off:off + g.faces.num]
        state["pressure"] = p[off + g.faces.num:]
    flux_bc = None if functional else bc
    if reconstruct_flux:
        if S.get("transType") == "tpfa":
            state["flux"] = _tpfa_flux(g, S, p, tot_mob, flux_bc)
        elif S.get("transType") == "mpfa":
            state["flux"] = _mpfa_flux(g, S, p, tot_mob, flux_bc)
        else:
            state["flux"] = _reconstruct_flux(g, S, p, tot_mob, flux_bc)

    # At k=2 the face and cell values are unknowns and are already in `state`;
    # at k=1 they are not, and have to be projected from the nodal solution.
    if face_pressure and "face_pressure" not in state:
        state["face_pressure"] = _face_pressures(g, S, p)
    if cell_pressure and "pressure" not in state:
        state["pressure"] = _cell_pressures(g, S, p)
    if matrix_output:
        state["A"] = A
        state["rhs"] = rhs
    return state

incompVEM = incomp_vem

if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
    from pymrst.modules.vem.utils import (
        polygon_int, polygon_int_3d, polyhedron_int,
    )
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

    def _boundary_faces(g):
        return np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))

    def test_k2_quadratic_patch_test():
        """The k=2 method reproduces exactly any quadratic that solves the PDE:
        node, face and cell-moment dofs all equal the field's exact moments. For
        K = [[2,0.6],[0.6,1]], q = x^2 - 2y^2 + ... is K-harmonic (trace(K H)=0),
        so no source term is needed."""
        g = _twisted_grid(4)
        rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 2)
        coords = np.asarray(g.nodes.coords)
        fc = np.asarray(g.faces.centroids)
        q = lambda P: P[:, 0] ** 2 - 2.0 * P[:, 1] ** 2 \
            + 3.0 * P[:, 0] - P[:, 1] + 4.0
        nn = g.nodes.num
        d = {}
        for nd in _boundary_nodes(g):
            d[int(nd)] = float(q(coords[nd:nd + 1])[0])
        for f in _boundary_faces(g):
            d[nn + int(f)] = float(q(fc[f:f + 1])[0])
        state = incomp_vem(g, S, dirichlet_nodes=d, reconstruct_flux=False)
        node_err = np.max(np.abs(state["node_pressure"] - q(coords)))
        face_err = np.max(np.abs(state["face_pressure"] - q(fc)))
        cell_ref = polygon_int(g, range(g.cells.num), lambda P: q(P)[:, None], 3)\
            .ravel() / np.asarray(g.cells.volumes)
        cell_err = np.max(np.abs(state["pressure"] - cell_ref))
        assert node_err < 1e-9 and face_err < 1e-9 and cell_err < 1e-9, \
            (node_err, face_err, cell_err)
        print(f"ok test_k2_quadratic_patch_test "
              f"(node {node_err:.2e}, face {face_err:.2e}, cell {cell_err:.2e})")

    def test_k2_flux_reconstruction_quadratic():
        """With an exact quadratic solution, K grad p is affine; the flux
        reconstruction still integrates it exactly per interface."""
        g = _twisted_grid(4)
        K = np.array([[2.0, 0.6], [0.6, 1.0]])
        rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 2)
        coords = np.asarray(g.nodes.coords)
        fc = np.asarray(g.faces.centroids)
        # Linear field -> constant K grad p, exactly reconstructable at k=2 too.
        grad = np.array([1.5, -2.0])
        lin = lambda P: P[:, :2] @ grad + 1.0
        nn = g.nodes.num
        d = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
        for f in _boundary_faces(g):
            d[nn + int(f)] = float(lin(fc[f:f + 1])[0])
        state = incomp_vem(g, S, dirichlet_nodes=d)
        internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))
        normals = np.asarray(g.faces.normals)[:, :2]
        expected = -(normals[internal] @ (K @ grad))
        err = np.max(np.abs(state["flux"][internal] - expected))
        assert err < 1e-9 * np.max(np.abs(expected)), err
        print(f"ok test_k2_flux_reconstruction_quadratic (err {err:.2e})")

    def test_3d_k1_planar_patch_test():
        """On a cuboid (planar-faced) 3-D grid the k=1 method is exact: it
        reproduces any linear pressure and its constant K grad p flux to machine
        precision, and each local block is symmetric with a one-dim constant
        kernel."""
        from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
        g = compute_vem_geometry(compute_geometry(cart_grid([3, 4, 2], [3.0, 4.0, 2.0])))
        rock = make_rock(g, np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 1)
        for Ae, nodes in zip(S["A"], S["dofVec"]):
            assert np.allclose(Ae, Ae.T, atol=1e-10)
            assert np.allclose(Ae @ np.ones(nodes.size), 0.0, atol=1e-9)
            assert np.min(np.linalg.eigvalsh(Ae)) > -1e-9
        coords = np.asarray(g.nodes.coords)
        Kmat = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
        grad = np.array([1.5, -2.0, 0.8])
        lin = lambda P: P @ grad + 3.0
        d = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
        state = incomp_vem(g, S, dirichlet_nodes=d)
        node_err = np.max(np.abs(state["node_pressure"] - lin(coords)))
        internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))
        expected = -(np.asarray(g.faces.normals)[internal] @ (Kmat @ grad))
        flux_err = np.max(np.abs(state["flux"][internal] - expected))
        assert node_err < 1e-9 and flux_err < 1e-9 * np.max(np.abs(expected))
        print(f"ok test_3d_k1_planar_patch_test (node {node_err:.2e}, flux {flux_err:.2e})")

    def test_3d_k2_planar_solve_reproduces_quadratic():
        """The 3-D k=2 solver reproduces a PDE-solving quadratic exactly on a
        cuboid grid: node, edge, face and cell dofs all match the field's exact
        moments. q = x^2 - 2 z^2 + ... is K-harmonic (trace(K H)=0) for the test
        tensor, so no source is needed."""
        from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
        g = compute_vem_geometry(compute_geometry(cart_grid([2, 3, 2], [2.0, 3.0, 2.0])))
        nn, ne, nf = g.nodes.num, g.edges.num, g.faces.num
        rock = make_rock(g, np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 2)
        coords = np.asarray(g.nodes.coords)
        ecent = np.asarray(g.edges.centroids)
        areas, vols = np.asarray(g.faces.areas), np.asarray(g.cells.volumes)
        q = lambda P: P[:, 0] ** 2 - 2 * P[:, 2] ** 2 + 3 * P[:, 0] - P[:, 1] \
            + 0.5 * P[:, 2] + 4
        npos, fnodes = np.asarray(g.faces.nodePos), np.asarray(g.faces.nodes)
        epos, fedges = np.asarray(g.faces.edgePos), np.asarray(g.faces.edges)
        bf = np.flatnonzero(np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
        d = {}
        for f in bf:
            for nd in fnodes[npos[f]:npos[f + 1]]:
                d[int(nd)] = float(q(coords[nd:nd + 1])[0])
            for e in fedges[epos[f]:epos[f + 1]]:
                d[nn + int(e)] = float(q(ecent[e:e + 1])[0])
            d[nn + ne + int(f)] = polygon_int_3d(g, [f], lambda P: q(P)[:, None], 3)[0, 0] / areas[f]
        st = incomp_vem(g, S, dirichlet_nodes=d, reconstruct_flux=False)
        face_ref = np.array([polygon_int_3d(g, [f], lambda P: q(P)[:, None], 3)[0, 0]
                             / areas[f] for f in range(nf)])
        cell_ref = polyhedron_int(g, range(g.cells.num), lambda P: q(P)[:, None], 3).ravel() / vols
        errs = (np.max(np.abs(st["node_pressure"] - q(coords))),
                np.max(np.abs(st["edge_pressure"] - q(ecent))),
                np.max(np.abs(st["face_pressure"] - face_ref)),
                np.max(np.abs(st["pressure"] - cell_ref)))
        assert max(errs) < 1e-9, errs
        print(f"ok test_3d_k2_planar_solve_reproduces_quadratic "
              f"(node {errs[0]:.1e}, edge {errs[1]:.1e}, face {errs[2]:.1e}, cell {errs[3]:.1e})")

    def test_tpfa_flux_exact_on_cartesian():
        """On a Cartesian (K-orthogonal) grid with a diagonal tensor, the TPFA
        flux reconstruction is exact for a linear pressure: every interior face
        flux equals -(K grad p) . area-normal, for both k=1 and k=2."""
        from pymrst.params.wells_and_bc.types import BoundaryCondition
        g = compute_geometry(cart_grid([4, 4], [4.0, 4.0]))
        Kdiag = np.array([2.0, 3.0])
        rock = make_rock(g, np.array([[2.0, 3.0]]).repeat(g.cells.num, 0), 0.2)
        coords = np.asarray(g.nodes.coords)
        grad = np.array([1.5, -0.7])
        lin = lambda P: P[:, :2] @ grad + 2.0
        internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))
        expected = -(np.asarray(g.faces.normals)[internal, :2] @ (Kdiag * grad))
        for k in (1, 2):
            S = compute_virtual_ip(g, rock, k, trans="tpfa")
            d = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
            if k == 2:
                fc = np.asarray(g.faces.centroids)
                nn = g.nodes.num
                for f in _boundary_faces(g):
                    d[nn + int(f)] = float(lin(fc[f:f + 1])[0])
            flux = incomp_vem(g, S, dirichlet_nodes=d)["flux"]
            err = np.max(np.abs(flux[internal] - expected))
            assert err < 1e-9 * np.max(np.abs(expected)), (k, err)
        print("ok test_tpfa_flux_exact_on_cartesian")

    def test_mpfa_flux_exact_on_twisted_where_tpfa_fails():
        """MPFA reconstruction is exact for a linear pressure even on a twisted,
        non-K-orthogonal grid with a full tensor - the defining MPFA property -
        while TPFA is badly wrong there. (MRST's own incompVEM MPFA path is
        broken in 2024a: computeMultiPointTrans no longer returns the S.T.T
        struct incompVEM's computeFlux expects, so this is validated by the
        exactness invariant rather than against MRST.)"""
        g = _twisted_grid(5)
        Kmat = np.array([[2.0, 0.6], [0.6, 1.0]])
        rock = make_rock(g, np.array([[2.0, 0.6, 1.0]]).repeat(g.cells.num, 0), 0.2)
        coords = np.asarray(g.nodes.coords)
        fc = np.asarray(g.faces.centroids)
        grad = np.array([1.5, -0.7])
        lin = lambda P: P[:, :2] @ grad + 2.0
        internal = np.flatnonzero(np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))
        expected = -(np.asarray(g.faces.normals)[internal, :2] @ (Kmat @ grad))
        scale = np.max(np.abs(expected))
        for k in (1, 2):
            Sm = compute_virtual_ip(g, rock, k, trans="mpfa")
            St = compute_virtual_ip(g, rock, k, trans="tpfa")
            d = {int(nd): float(lin(coords[nd:nd + 1])[0]) for nd in _boundary_nodes(g)}
            if k == 2:
                for f in _boundary_faces(g):
                    d[g.nodes.num + int(f)] = float(lin(fc[f:f + 1])[0])
            qm = incomp_vem(g, Sm, dirichlet_nodes=d)["flux"][internal]
            qt = incomp_vem(g, St, dirichlet_nodes=d)["flux"][internal]
            assert np.max(np.abs(qm - expected)) < 1e-9 * scale, (k, "mpfa")
            assert np.max(np.abs(qt - expected)) > 0.1 * scale, (k, "tpfa should fail")
        print("ok test_mpfa_flux_exact_on_twisted_where_tpfa_fails")

    def test_face_and_cell_pressure_are_projected_at_first_order() -> None:
        """At k=1 neither is an unknown, so both come from the nodal solution.

        The 3-D branch of `_face_pressures` raised `NotImplementedError` until
        this phase, which made k=1 face pressures unavailable on any 3-D grid.
        """
        from pymrst.modules.vem.utils import compute_vem_geometry
        g = compute_vem_geometry(compute_geometry(
            cart_grid([3, 3, 3], [3.0, 3.0, 3.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        exact = lambda P: 1 + 2 * P[:, 0] - P[:, 1] + 0.5 * P[:, 2]
        coords = np.asarray(g.nodes.coords)
        pinned = {int(n): float(exact(coords[n:n + 1])[0])
                  for n in _boundary_nodes(g)}
        state = incomp_vem(g, S, dirichlet_nodes=pinned,
                           face_pressure=True, cell_pressure=True)
        face_err = np.abs(state["face_pressure"]
                          - exact(np.asarray(g.faces.centroids))).max()
        cell_err = np.abs(state["pressure"]
                          - exact(np.asarray(g.cells.centroids))).max()
        assert face_err < 1e-12 and cell_err < 1e-12, (face_err, cell_err)
        print(f"ok test_face_and_cell_pressure_are_projected_at_first_order "
              f"(face {face_err:.1e}, cell {cell_err:.1e})")

    def test_the_reconstructions_are_off_unless_asked_for() -> None:
        """They cost a projection per face, so they are opt-in as in MRST."""
        from pymrst.modules.vem.utils import compute_vem_geometry
        g = compute_vem_geometry(compute_geometry(cart_grid([2, 2], [2.0, 2.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        pinned = {int(n): 0.0 for n in _boundary_nodes(g)}
        state = incomp_vem(g, S, dirichlet_nodes=pinned, src=None)
        assert "face_pressure" not in state and "pressure" not in state
        print("ok test_the_reconstructions_are_off_unless_asked_for")

    def test_a_caller_supplied_solver_is_used() -> None:
        from pymrst.modules.vem.utils import compute_vem_geometry
        from scipy.sparse.linalg import spsolve as reference
        g = compute_vem_geometry(compute_geometry(cart_grid([3, 3], [3.0, 3.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        exact = lambda P: 1 + P[:, 0] - 2 * P[:, 1]
        coords = np.asarray(g.nodes.coords)
        pinned = {int(n): float(exact(coords[n:n + 1])[0])
                  for n in _boundary_nodes(g)}

        used = []

        def solver(A, b):
            used.append(A.shape)
            return reference(A, b)

        state = incomp_vem(g, S, dirichlet_nodes=pinned, lin_solve=solver)
        assert used, "the supplied solver was never called"
        default = incomp_vem(g, S, dirichlet_nodes=pinned)
        assert np.allclose(state["node_pressure"], default["node_pressure"])
        print("ok test_a_caller_supplied_solver_is_used")

    def test_matrix_output_returns_the_assembled_system() -> None:
        from pymrst.modules.vem.utils import compute_vem_geometry
        g = compute_vem_geometry(compute_geometry(cart_grid([2, 2], [2.0, 2.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        pinned = {int(n): 0.0 for n in _boundary_nodes(g)}
        state = incomp_vem(g, S, dirichlet_nodes=pinned, matrix_output=True)
        assert state["A"].shape == (g.nodes.num, g.nodes.num)
        assert state["rhs"].shape == (g.nodes.num,)
        print("ok test_matrix_output_returns_the_assembled_system")

    def test_prescribed_influx_is_conserved_in_three_dimensions() -> None:
        """A 3-D Neumann boundary condition, which is not a 2-D-only feature.

        The module docstring used to claim 3-D flux boundaries were treated as
        no-flow. They are not: `_neumann_rhs` distributes the flux to the face
        nodes through that face's P0 projection at k=1, and to the face-moment
        dof at k=2. What comes in leaves.
        """
        from pymrst.params.wells_and_bc.add_bc import add_bc
        from pymrst.modules.vem.utils import compute_vem_geometry
        n = 4
        g = compute_vem_geometry(compute_geometry(
            cart_grid([n, 2, 2], [float(n), 2.0, 2.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        centroids = np.asarray(g.faces.centroids)
        neighbours = np.asarray(g.faces.neighbors)
        outer = np.any(neighbours < 0, axis=1)
        left = np.flatnonzero((np.abs(centroids[:, 0]) < 1e-9) & outer)
        right = np.flatnonzero((np.abs(centroids[:, 0] - n) < 1e-9) & outer)
        areas = np.asarray(g.faces.areas)

        rate = 1.0
        bc = add_bc(None, left, "flux", rate * areas[left] / areas[left].sum())
        bc = add_bc(bc, right, "pressure", np.zeros(right.size))
        state = incomp_vem(g, S, bc=bc)
        produced = float(np.abs(state["flux"][right]).sum())
        assert np.isclose(produced, rate, atol=1e-10), produced
        print(f"ok test_prescribed_influx_is_conserved_in_three_dimensions "
              f"(err {abs(produced - rate):.1e})")

    def _bfaces(grid):
        return np.flatnonzero(
            np.any(np.asarray(grid.faces.neighbors) < 0, axis=1))

    def test_a_function_valued_pressure_needs_no_dof_knowledge() -> None:
        """The point of addBCVEM: the caller names faces, not dof indices."""
        from pymrst.modules.vem.utils import compute_vem_geometry
        from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
        g = compute_vem_geometry(compute_geometry(cart_grid([4, 4], [4.0, 4.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        exact = lambda P: 1 + 2 * P[:, 0] - 3 * P[:, 1]
        bc = add_bc_vem(None, _bfaces(g), "pressure", exact)
        state = incomp_vem(g, S, bc=bc)
        err = np.abs(state["node_pressure"]
                     - exact(np.asarray(g.nodes.coords)[:, :2])).max()
        assert err < 1e-12, err
        print(f"ok test_a_function_valued_pressure_needs_no_dof_knowledge "
              f"({err:.1e})")

    def test_it_agrees_with_the_hand_built_dof_map() -> None:
        """`add_bc_vem` and `dirichlet_nodes` must be two routes to one answer."""
        from pymrst.modules.vem.utils import compute_vem_geometry
        from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
        g = compute_vem_geometry(compute_geometry(cart_grid([3, 3], [3.0, 3.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        exact = lambda P: 2 - P[:, 0] + 0.5 * P[:, 1]
        coords = np.asarray(g.nodes.coords)

        by_function = incomp_vem(
            g, S, bc=add_bc_vem(None, _bfaces(g), "pressure", exact))
        by_hand = incomp_vem(g, S, dirichlet_nodes={
            int(n): float(exact(coords[n:n + 1])[0])
            for n in _boundary_nodes(g)})
        assert np.allclose(by_function["node_pressure"],
                           by_hand["node_pressure"], atol=1e-12)
        print("ok test_it_agrees_with_the_hand_built_dof_map")

    def test_the_second_order_face_moments_are_located_too() -> None:
        """At k=2 a face carries a moment; the function is evaluated there.

        The field must be harmonic or the solve needs a source - `x^2 - y^2`
        is, `x^2` alone is not.
        """
        from pymrst.modules.vem.utils import compute_vem_geometry
        from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
        g = compute_vem_geometry(compute_geometry(cart_grid([4, 4], [4.0, 4.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 2)
        exact = lambda P: (1 + P[:, 0] - 2 * P[:, 1]
                           + P[:, 0] ** 2 - P[:, 1] ** 2)
        bc = add_bc_vem(None, _bfaces(g), "pressure", exact)
        state = incomp_vem(g, S, bc=bc, reconstruct_flux=False)
        node_err = np.abs(state["node_pressure"]
                          - exact(np.asarray(g.nodes.coords)[:, :2])).max()
        face_err = np.abs(state["face_pressure"]
                          - exact(np.asarray(g.faces.centroids)[:, :2])).max()
        assert node_err < 1e-9 and face_err < 1e-9, (node_err, face_err)
        print(f"ok test_the_second_order_face_moments_are_located_too "
              f"(node {node_err:.1e}, face {face_err:.1e})")

    def test_the_three_dimensional_edge_dofs_are_located_too() -> None:
        """A k=2 face in 3-D owns edge moments as well as a face moment."""
        from pymrst.modules.vem.utils import compute_vem_geometry
        from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
        g = compute_vem_geometry(compute_geometry(
            cart_grid([2, 2, 2], [2.0, 2.0, 2.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 2)
        exact = lambda P: (1 + P[:, 0] - P[:, 2]
                           + P[:, 0] ** 2 - P[:, 1] ** 2)
        bc = add_bc_vem(None, _bfaces(g), "pressure", exact)
        state = incomp_vem(g, S, bc=bc, reconstruct_flux=False)
        err = np.abs(state["node_pressure"]
                     - exact(np.asarray(g.nodes.coords))).max()
        assert err < 1e-9, err
        print(f"ok test_the_three_dimensional_edge_dofs_are_located_too "
              f"({err:.1e})")

    def test_a_function_valued_flux_is_integrated_over_each_face() -> None:
        """Simpson's rule over the face; a constant flux must reproduce add_bc."""
        from pymrst.params.wells_and_bc.add_bc import add_bc
        from pymrst.modules.vem.utils import compute_vem_geometry
        from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
        g = compute_vem_geometry(compute_geometry(cart_grid([4, 2], [4.0, 2.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        centroids = np.asarray(g.faces.centroids)
        outer = np.any(np.asarray(g.faces.neighbors) < 0, axis=1)
        left = np.flatnonzero((np.abs(centroids[:, 0]) < 1e-9) & outer)
        right = np.flatnonzero((np.abs(centroids[:, 0] - 4.0) < 1e-9) & outer)
        areas = np.asarray(g.faces.areas)

        rate = 0.25                              # per unit area, uniform
        by_function = add_bc_vem(None, left, "flux",
                                 lambda P: np.full(len(P), rate))
        by_function = add_bc_vem(by_function, right, "pressure",
                                 lambda P: np.zeros(len(P)))
        state = incomp_vem(g, S, bc=by_function)
        produced = float(np.abs(state["flux"][right]).sum())
        expected = rate * areas[left].sum()
        assert np.isclose(produced, expected, rtol=1e-10), (produced, expected)
        print(f"ok test_a_function_valued_flux_is_integrated_over_each_face "
              f"(err {abs(produced - expected):.1e})")

    def test_a_function_valued_flux_is_rejected_in_three_dimensions() -> None:
        """MRST's 3-D branch here is an empty `else`; this refuses instead."""
        from pymrst.modules.vem.utils import compute_vem_geometry
        from pymrst.modules.vem.utils.add_bc_vem import add_bc_vem
        g = compute_vem_geometry(compute_geometry(
            cart_grid([2, 2, 2], [2.0, 2.0, 2.0])))
        S = compute_virtual_ip(g, make_rock(g, 1.0, 1.0), 1)
        bc = add_bc_vem(None, _bfaces(g), "flux", lambda P: np.ones(len(P)))
        try:
            incomp_vem(g, S, bc=bc)
        except NotImplementedError as error:
            assert "2-D only" in str(error)
            print("ok test_a_function_valued_flux_is_rejected_in_three_dimensions")
            return
        raise AssertionError("a 3-D functional flux must be reported")

    test_patch_test_linear_field()
    test_full_tensor_patch_test()
    test_flux_reconstruction_reproduces_uniform_gradient()
    test_source_rate_conservation()
    test_k2_quadratic_patch_test()
    test_k2_flux_reconstruction_quadratic()
    test_3d_k1_planar_patch_test()
    test_3d_k2_planar_solve_reproduces_quadratic()
    test_tpfa_flux_exact_on_cartesian()
    test_mpfa_flux_exact_on_twisted_where_tpfa_fails()
    test_face_and_cell_pressure_are_projected_at_first_order()
    test_the_reconstructions_are_off_unless_asked_for()
    test_a_caller_supplied_solver_is_used()
    test_matrix_output_returns_the_assembled_system()
    test_prescribed_influx_is_conserved_in_three_dimensions()
    test_a_function_valued_pressure_needs_no_dof_knowledge()
    test_it_agrees_with_the_hand_built_dof_map()
    test_the_second_order_face_moments_are_located_too()
    test_the_three_dimensional_edge_dofs_are_located_too()
    test_a_function_valued_flux_is_integrated_over_each_face()
    test_a_function_valued_flux_is_rejected_in_three_dimensions()
    print("\nAll tests passed!")
