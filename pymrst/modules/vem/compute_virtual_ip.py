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

from pymrst.modules.vem.utils import (
    polygon_int,
    polygon_int_3d,
    polyhedron_int,
)
from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
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


# Second-order (k=2) monomial basis in scaled coordinates xi=(x-cx)/cd,
# eta=(y-cy)/cd: [1, xi, eta, xi^2, xi*eta, eta^2].
_K2_ALPHA = np.array([0, 1, 0, 2, 1, 0])
_K2_BETA = np.array([0, 0, 1, 0, 1, 2])


def _monomials_2d(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Value of the six k=2 monomials at scaled points, shape (n, 6)."""
    return xi[:, None] ** _K2_ALPHA * eta[:, None] ** _K2_BETA


def _grad_monomials_2d(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Scaled gradient of the five non-constant monomials, shape (n, 5, 2)."""
    z, o = np.zeros_like(xi), np.ones_like(xi)
    dxi = np.stack([o, z, 2 * xi, eta, z], axis=1)          # d/dxi of m1..m5
    deta = np.stack([z, o, z, xi, 2 * eta], axis=1)         # d/deta of m1..m5
    return np.stack([dxi, deta], axis=2)                    # (n, 5, 2)


def _local_ip_k2_2d(g, cell: int, K: np.ndarray):
    """Second-order (k=2) 2-D VEM local stiffness (MRST computeBD2D, k=2).

    Returns `(A, node_dofs, face_dofs)`: the local matrix over the ordered dof
    set [cell nodes, cell faces (edge-centroid dofs), cell moment].
    """
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    normals = np.asarray(g.faces.normals, dtype=float)
    fcent = np.asarray(g.faces.centroids, dtype=float)

    faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
    nodes = np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
    local = {int(nd): i for i, nd in enumerate(nodes)}
    nn, nf = nodes.size, faces.size
    NP = nn + nf + 1                                   # dofs: nodes, faces, cell
    nk = 6

    cc = np.asarray(g.cells.centroids, dtype=float)[cell][:2]
    cv = float(np.asarray(g.cells.volumes)[cell])
    cd = _cell_diameter(coords[nodes][:, :2])

    def scaled(P):
        return (P[:, 0] - cc[0]) / cd, (P[:, 1] - cc[1]) / cd

    # --- D matrix (NP x 6): monomials at nodes, at face centroids, cell moments.
    D = np.zeros((NP, nk))
    xi, eta = scaled(coords[nodes][:, :2])
    D[:nn, :] = _monomials_2d(xi, eta)
    xi, eta = scaled(fcent[faces][:, :2])
    D[nn:nn + nf, :] = _monomials_2d(xi, eta)
    mom = polygon_int(g, [cell],
                      lambda P: _monomials_2d(*scaled(P)), 2)[0] / cv
    D[nn + nf, :] = mom

    # --- B matrix (6 x NP): boundary integrals of grad phi . K grad m.
    B = np.zeros((nk, NP))
    for f in faces:
        fn2 = fnodes[node_pos[f]:node_pos[f + 1]]
        a, b = int(fn2[0]), int(fn2[1])
        sign = 1.0 if cell == neigh[f, 0] else -1.0
        nrm = normals[f, :2] * sign                    # outward, length-weighted
        fc = fcent[f, :2]
        for pt, w, dof in ((coords[a][:2], 1.0 / 6, local[a]),
                           (coords[b][:2], 1.0 / 6, local[b]),
                           (fc, 2.0 / 3, nn + list(faces).index(f))):
            gx, ge = scaled(pt[None, :])
            gm = _grad_monomials_2d(gx, ge)[0]         # (5, 2) scaled gradient
            B[1:, dof] += w * (gm @ K) @ nrm / cd
    B[0, nn + nf] = 1.0                                # P0 = cell moment dof
    K2 = np.array([K[0, 0], K[0, 1], K[1, 1]])
    B[[3, 4, 5], nn + nf] += -2.0 * K2 * cv / cd ** 2

    M = B @ D
    PiNstar = np.linalg.solve(M, B)
    PiN = D @ PiNstar
    Mc = M.copy()
    Mc[0, :] = 0.0
    sigma = cd ** (g.griddim - 2) * np.trace(K) / g.griddim
    A = PiNstar.T @ Mc @ PiNstar + sigma * (np.eye(NP) - PiN).T @ (np.eye(NP) - PiN)
    return A, nodes, faces


def _face_p0_weights(g, face: int, K3: np.ndarray) -> Dict[int, float]:
    """P0 projection row of a planar face's 2-D VEM (MRST `PiNFstar(1,:)`).

    Runs the first-order 2-D VEM on the face (treated as a polygon in its own
    in-plane coordinate system, with the permeability projected onto the face)
    and returns `{global node: weight}`, where the weights map the face's nodal
    values to the face average - exactly the quantity the 3-D cell B matrix
    needs to integrate `K grad m . n` over each face.
    """
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    edge_pos = np.asarray(g.faces.edgePos, dtype=np.int64)
    fedges = np.asarray(g.faces.edges, dtype=np.int64)
    e_nodes = np.asarray(g.edges.nodes, dtype=np.int64).reshape(-1, 2)
    e_len = np.asarray(g.edges.lengths, dtype=float)
    e_nrm = np.asarray(g.faces.edgeNormals, dtype=float)
    coords = np.asarray(g.nodes.coords, dtype=float)
    fc = np.asarray(g.faces.centroids, dtype=float)[face]
    fdiam = float(np.asarray(g.faces.diameters)[face])

    nds = fnodes[node_pos[face]:node_pos[face + 1]]
    local = {int(nd): i for i, nd in enumerate(nds)}
    n = nds.size

    # In-plane orthonormal basis (v1 along the first edge, v2 = normal x v1).
    v1 = coords[nds[1]] - coords[nds[0]]
    v1 /= np.linalg.norm(v1)
    fn = np.asarray(g.faces.normals, dtype=float)[face]
    v2 = np.cross(fn, v1)
    v2 /= np.linalg.norm(v2)
    F = np.column_stack([v1, v2])                      # 3x2 global->face map
    X2 = (coords[nds] - fc) @ F                         # face node local coords
    K2 = F.T @ K3 @ F

    D = np.column_stack([np.ones(n), X2[:, 0] / fdiam, X2[:, 1] / fdiam])
    B = np.zeros((3, n))
    for ei, e in enumerate(fedges[edge_pos[face]:edge_pos[face + 1]]):
        a, b = int(e_nodes[e, 0]), int(e_nodes[e, 1])
        nrm = (e_nrm[edge_pos[face] + ei] * e_len[e]) @ F   # length-weighted, local
        mid = 0.5 * (X2[local[a]] + X2[local[b]])
        if nrm @ mid < 0:                              # enforce outward (from fc)
            nrm = -nrm
        Kn = nrm @ K2
        for nd in (a, b):
            li = local[nd]
            B[0, li] += 0.5 * e_len[e]
            B[1, li] += 0.5 * Kn[0] / fdiam
            B[2, li] += 0.5 * Kn[1] / fdiam

    pi_row0 = np.linalg.solve(B @ D, B)[0, :]          # PiNFstar first row
    return {int(nd): pi_row0[local[int(nd)]] for nd in nds}


def _local_ip_3d_k1(g, cell: int, K: np.ndarray):
    """First-order (k=1) 3-D VEM local stiffness (MRST computeVirtualIP, 3-D).

    The cell B matrix integrates `K grad m . n` over each face using that face's
    2-D VEM P0 projection; the constant row is the uniform vertex average.
    """
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
    local = {int(nd): i for i, nd in enumerate(nodes)}
    n = nodes.size
    X = coords[nodes]

    cc = np.asarray(g.cells.centroids, dtype=float)[cell]
    cd = float(np.asarray(g.cells.diameters)[cell])

    D = np.column_stack([np.ones(n), (X - cc) / cd])   # [1, xi, eta, zeta]
    B = np.zeros((4, n))
    B[0, :] = 1.0 / n
    for f in faces:
        sign = 1.0 if cell == neigh[f, 0] else -1.0
        nu = normals[f] / areas[f] * sign              # unit outward normal
        Kn = K @ nu                                    # (K grad m)_normal, 3-vec
        w = _face_p0_weights(g, int(f), K)
        for nd in fnodes[node_pos[f]:node_pos[f + 1]]:
            B[1:, local[int(nd)]] += Kn * w[int(nd)] * areas[f] / cd

    M = B @ D
    PiNstar = np.linalg.solve(M, B)
    PiN = D @ PiNstar
    Mc = M.copy()
    Mc[0, :] = 0.0
    sigma = cd ** (g.griddim - 2) * np.trace(K) / g.griddim
    A = PiNstar.T @ Mc @ PiNstar + sigma * (np.eye(n) - PiN).T @ (np.eye(n) - PiN)
    return A, nodes, PiNstar


# 3-D k=2 monomials in scaled coords xi=(x-cx)/cd: exponents and constant
# Hessians (in xi) for the six quadratic monomials.
_K2_3D_EXP = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
                       (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)])


def _monomials_3d(P: np.ndarray, cc: np.ndarray, cd: float) -> np.ndarray:
    """The ten 3-D k=2 monomials at points `P`, shape (n, 10)."""
    xi = (P - cc) / cd
    return np.prod(xi[:, None, :] ** _K2_3D_EXP[None, :, :], axis=2)


def _mono_grad_hess_3d(alpha: int):
    """Scaled-gradient function and constant Hessian of monomial `alpha`.

    Every monomial has degree <= 2, so the Hessian (in xi) is a constant matrix.
    """
    e = _K2_3D_EXP[alpha]
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                if e[i] == 2:
                    H[i, j] = 2.0                       # d^2/dxi_i^2 of xi_i^2
            elif e[i] == 1 and e[j] == 1:
                H[i, j] = 1.0                           # d^2/dxi_i dxi_j of mixed

    def grad(xi):                                       # xi: (3,) -> (3,)
        g = np.zeros(3)
        for d in range(3):
            if e[d] == 0:
                continue
            ee = e.copy()
            ee[d] -= 1
            g[d] = e[d] * np.prod(xi ** ee)
        return g
    return grad, H


def _face_quad(X2: np.ndarray, k: int = 3):
    """Quadrature points/weights over the 2-D polygon `X2` (its own local frame).

    Triangulates the polygon and maps a precision-`k` reference-triangle rule
    onto every triangle; returns `(points (m, 2), weights (m,))`.
    """
    from pymrst.modules.vem.utils import triangle_quad_rule
    from scipy.spatial import Delaunay
    Xq, w, _, vol = triangle_quad_rule(k)
    pts, wts = [], []
    for t in Delaunay(X2).simplices:
        v0 = X2[t[0]]
        A2 = np.array([X2[t[1]] - v0, X2[t[2]] - v0])
        det = abs(A2[0, 0] * A2[1, 1] - A2[0, 1] * A2[1, 0])
        pts.append(v0 + Xq @ A2)
        wts.append(vol * det * w)
    return np.vstack(pts), np.concatenate(wts)


def _face_k2_projection(g, face: int, K3: np.ndarray):
    """First return the face's k=2 2-D VEM H1 projection `PiNFstar` (6 x NF), the
    global dof ids of its dofs [face nodes, face edges, face moment], and the
    face's in-plane frame `(T, fc, fd)` with its node polygon `X2`. Mirrors
    `computeBD2D` (k=2) on the face flattened through its centroid `fc`."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    edge_pos = np.asarray(g.faces.edgePos, dtype=np.int64)
    fedges = np.asarray(g.faces.edges, dtype=np.int64)
    e_nodes = np.asarray(g.edges.nodes, dtype=np.int64).reshape(-1, 2)
    e_len = np.asarray(g.edges.lengths, dtype=float)
    e_cent = np.asarray(g.edges.centroids, dtype=float)
    e_nrm = np.asarray(g.faces.edgeNormals, dtype=float)
    coords = np.asarray(g.nodes.coords, dtype=float)
    fc = np.asarray(g.faces.centroids, dtype=float)[face]
    fd = float(np.asarray(g.faces.diameters)[face])
    area = float(np.asarray(g.faces.areas)[face])

    nds = fnodes[node_pos[face]:node_pos[face + 1]]
    local = {int(nd): i for i, nd in enumerate(nds)}
    nn = nds.size
    eids = fedges[edge_pos[face]:edge_pos[face + 1]]
    ne = eids.size
    NF = nn + ne + 1

    v1 = coords[nds[1]] - coords[nds[0]]
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(np.asarray(g.faces.normals, dtype=float)[face], v1)
    v2 /= np.linalg.norm(v2)
    T = np.column_stack([v1, v2])                      # 3x2
    K2 = T.T @ K3 @ T
    to2 = lambda P: (P - fc) @ T                       # global -> local 2-D

    X2 = to2(coords[nds])

    def mon2(P2):                                      # local 2-D monomials
        s = P2 / fd
        return s[:, 0][:, None] ** _K2_ALPHA * s[:, 1][:, None] ** _K2_BETA

    D = np.zeros((NF, 6))
    D[:nn] = mon2(X2)
    D[nn:nn + ne] = mon2(to2(e_cent[eids]))
    qp, qw = _face_quad(X2)                             # fc-flattened face rule
    D[nn + ne] = (qw @ mon2(qp)) / area

    B = np.zeros((6, NF))
    for j, e in enumerate(eids):
        a, b = int(e_nodes[e, 0]), int(e_nodes[e, 1])
        la, lb = local[a], local[b]
        # Use MRST's in-plane edge normals as-is (validated in phase 410); their
        # consistent orientation is what computeBD2D expects.
        nrm = (e_nrm[edge_pos[face] + j] * e_len[e]) @ T
        ec2 = to2(e_cent[e:e + 1])[0]
        for pt, wt, dof in ((X2[la], 1.0 / 6, la), (X2[lb], 1.0 / 6, lb),
                            (ec2, 2.0 / 3, nn + j)):
            s = pt / fd
            gm = _grad_monomials_2d(np.array([s[0]]), np.array([s[1]]))[0]
            B[1:, dof] += wt * (gm @ K2) @ nrm / fd
    B[0, nn + ne] = 1.0
    K2v = np.array([K2[0, 0], K2[0, 1], K2[1, 1]])
    B[[3, 4, 5], nn + ne] += -2.0 * K2v * area / fd ** 2

    PiNFstar = np.linalg.solve(B @ D, B)               # 6 x NF
    nnodes, nedges = g.nodes.num, g.edges.num
    dof_glob = ([int(nd) for nd in nds]
                + [nnodes + int(e) for e in eids]
                + [nnodes + nedges + int(face)])
    return PiNFstar, dof_glob, (T, fc, fd, X2)


def _local_ip_3d_k2(g, cell: int, K: np.ndarray):
    """Second-order (k=2) 3-D VEM local stiffness (MRST computeVirtualIP, 3-D).

    Dofs are [cell nodes, cell edges, cell faces, cell moment]. The cell B matrix
    integrates `K grad m_alpha . n` over each face by expanding the (linear) flux
    in the face's linear monomials and pairing it, through the face's k=2 VEM
    projection `PiNFstar` and the exact face moments `int_f M_beta L_gamma`, with
    the face dofs - the same quantity MRST assembles via polynomial products.
    """
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    epos = np.asarray(g.cells.edgePos, dtype=np.int64)
    cedges = np.asarray(g.cells.edges, dtype=np.int64)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    normals = np.asarray(g.faces.normals, dtype=float)
    areas = np.asarray(g.faces.areas, dtype=float)
    e_cent = np.asarray(g.edges.centroids, dtype=float)
    nnodes, nedges, nfaces = g.nodes.num, g.edges.num, g.faces.num

    faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
    nodes = np.asarray(g.cells.nodes, dtype=np.int64)[
        np.asarray(g.cells.nodePos)[cell]:np.asarray(g.cells.nodePos)[cell + 1]]
    edges = cedges[epos[cell]:epos[cell + 1]]
    ncn, nce, ncf = nodes.size, edges.size, faces.size
    NP = ncn + nce + ncf + 1

    # Local dof position of every global dof of this cell.
    pos = {int(nd): i for i, nd in enumerate(nodes)}
    pos.update({nnodes + int(e): ncn + i for i, e in enumerate(edges)})
    pos.update({nnodes + nedges + int(f): ncn + nce + i
                for i, f in enumerate(faces)})
    cell_dof = NP - 1

    cc = np.asarray(g.cells.centroids, dtype=float)[cell]
    cd = float(np.asarray(g.cells.diameters)[cell])
    vol = float(np.asarray(g.cells.volumes)[cell])

    # Face projections and fc-flattened quadrature, computed once per face.
    fdata = {}
    for f in faces:
        PiNF, dof_glob, (T, fc, fd, X2) = _face_k2_projection(g, int(f), K)
        qp, qw = _face_quad(X2)                         # local 2-D points/weights
        gpts = fc + qp @ T.T                            # back to global 3-D
        fdata[int(f)] = (PiNF, dof_glob, T, fc, fd, qp, qw, gpts)

    # --- D matrix (NP x 10). Node/edge dofs are monomial values. Each face-moment
    # dof follows MRST's recipe: constant -> 1, linear monomials -> value at the
    # face centroid (centroid rule), quadratics -> the fc-flattened face integral.
    # The cell-moment dof uses the divergence theorem over the same flattened
    # faces, keeping the moments consistent on non-planar-faced cells.
    D = np.zeros((NP, 10))
    D[:ncn] = _monomials_3d(coords[nodes], cc, cd)
    D[ncn:ncn + nce] = _monomials_3d(e_cent[edges], cc, cd)
    ax = _K2_3D_EXP[:, 0] + 1                           # x-antiderivative powers
    cell_mom = np.zeros(10)
    for i, f in enumerate(faces):
        PiNF, dof_glob, T, fc, fd, qp, qw, gpts = fdata[int(f)]
        row = np.zeros(10)
        row[0] = 1.0
        row[1:4] = (fc - cc) / cd                       # linear at face centroid
        row[4:] = (qw @ _monomials_3d(gpts, cc, cd)[:, 4:]) / areas[f]
        D[ncn + nce + i] = row
        sign = 1.0 if cell == neigh[f, 0] else -1.0
        nux = normals[f, 0] / areas[f] * sign          # unit outward normal, x
        xig = (gpts - cc) / cd
        antideriv = (cd / ax * xig[:, 0:1] ** ax
                     * xig[:, 1:2] ** _K2_3D_EXP[:, 1]
                     * xig[:, 2:3] ** _K2_3D_EXP[:, 2])
        cell_mom += nux * (qw @ antideriv)
    D[cell_dof] = cell_mom / vol

    # --- B matrix (10 x NP): boundary flux integrals + volume divergence term.
    grads_hess = [_mono_grad_hess_3d(a) for a in range(10)]
    B = np.zeros((10, NP))
    for f in faces:
        PiNF, dof_glob, T, fc, fd, qp, qw, gpts = fdata[int(f)]
        sign = 1.0 if cell == neigh[f, 0] else -1.0
        nu = normals[f] / areas[f] * sign              # unit outward normal
        # Face moments Q[beta, gamma] = int_f M_beta * L_gamma, L=[1, u, w] in the
        # face's local frame, on the fc-flattened polygon.
        M = (qp[:, 0] / fd)[:, None] ** _K2_ALPHA * (qp[:, 1] / fd)[:, None] ** _K2_BETA
        L = np.column_stack([np.ones(qp.shape[0]), qp[:, 0], qp[:, 1]])
        Q = np.einsum("q,qb,qg->bg", qw, M, L)         # 6 x 3
        xi_fc = (fc - cc) / cd
        for alpha in range(1, 10):
            grad, H = grads_hess[alpha]
            g0 = nu @ K @ (grad(xi_fc) / cd)           # g_alpha(fc)
            slope = nu @ K @ (H / cd ** 2)             # grad_x g_alpha (3-vec)
            e = np.array([g0, slope @ T[:, 0], slope @ T[:, 1]])
            contrib = PiNF.T @ (Q @ e)                 # NF-vector over face dofs
            for i, gd in enumerate(dof_glob):
                B[alpha, pos[gd]] += contrib[i]
    B[0, cell_dof] = 1.0
    for alpha in range(4, 10):
        _, H = grads_hess[alpha]
        B[alpha, cell_dof] += -np.trace(K @ H) / cd ** 2 * vol

    M = B @ D
    PiNstar = np.linalg.solve(M, B)
    PiN = D @ PiNstar
    Mc = M.copy()
    Mc[0, :] = 0.0
    sigma = cd ** (g.griddim - 2) * np.trace(K) / g.griddim
    A = PiNstar.T @ Mc @ PiNstar + sigma * (np.eye(NP) - PiN).T @ (np.eye(NP) - PiN)
    dofvec = np.concatenate([nodes, nnodes + edges, nnodes + nedges + faces,
                             [nnodes + nedges + nfaces + cell]])
    return A, dofvec


def compute_virtual_ip(g, rock, k: int = 1, trans: str = None) -> Dict:
    """Return the VEM system `S` for grid `g` and `rock`.

    `S['A']` is the list of per-cell local stiffness matrices, `S['dofVec']` the
    matching lists of GLOBAL dof indices, `S['order']` the method order. For
    k=1 the dofs are grid nodes; for k=2 they are [nodes, faces, cell] with the
    global numbering nodes | faces+nNodes | cell+nNodes+nFaces.

    `trans` selects how `incomp_vem` reconstructs fluxes from the VEM pressure:
    `None` uses the direct VEM gradient; `'tpfa'` stores the two-point
    half-transmissibilities (`computeTrans`); `'mpfa'` stores the multi-point
    transmissibility operator (`computeMultiPointTrans`). With a transmissibility
    scheme the flux comes from a TPFA/MPFA reconstruction on the VEM cell/face
    pressures rather than the direct gradient.
    """
    if k not in (1, 2):
        raise NotImplementedError("only k=1 and k=2 VEM are ported")
    if trans not in (None, "tpfa", "mpfa"):
        raise NotImplementedError("trans must be None, 'tpfa' or 'mpfa'")
    dim = g.griddim
    if dim == 3 and g.edges is None:
        compute_vem_geometry(g)                        # ensure edge topology
    tensor = perm_tensor(rock, dim)[0]                 # (ncells, dim^2)
    A_blocks, dof, proj = [], [], []
    nn, nfc = g.nodes.num, g.faces.num
    for c in range(g.cells.num):
        K = tensor[c].reshape(dim, dim)
        if k == 1:
            if dim == 3:
                A, nodes, PiNstar = _local_ip_3d_k1(g, c, K)
            else:
                A, nodes, PiNstar = _local_ip(g, c, K)
            A_blocks.append(A)
            dof.append(nodes)
            proj.append(PiNstar)
        elif dim == 2:
            A, nodes, faces = _local_ip_k2_2d(g, c, K)
            A_blocks.append(A)
            dof.append(np.concatenate([nodes, faces + nn, [nn + nfc + c]]))
        else:
            A, dofvec = _local_ip_3d_k2(g, c, K)
            A_blocks.append(A)
            dof.append(dofvec)
    S = {"A": A_blocks, "dofVec": dof, "order": k, "transType": trans}
    if k == 1:
        S["PiNstar"] = proj
    if k == 1 and dim == 3:
        S["tensor"] = tensor                           # for 3-D Neumann face P0
    if trans == "tpfa":
        from pymrst.solvers.compute_trans import compute_trans
        S["T"] = compute_trans(g, rock)                # half-face trans
    elif trans == "mpfa":
        from pymrst.modules.mpfa.compute_multipoint_trans import (
            compute_multipoint_trans,
        )
        S["Tmpfa"] = compute_multipoint_trans(g, rock)
    return S

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

    def test_local_blocks_are_spd_and_kill_constants():
        g = _twisted_grid(4)
        rock = make_rock(g, np.array([[2.0, 0.5, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 1)
        for Ae, nodes in zip(S["A"], S["dofVec"]):
            assert np.allclose(Ae, Ae.T, atol=1e-10)
            assert np.allclose(Ae @ np.ones(nodes.size), 0.0, atol=1e-9)
            assert np.min(np.linalg.eigvalsh(Ae)) > -1e-9
        print("ok test_local_blocks_are_spd_and_kill_constants")

    def test_3d_k1_twisted_kills_constants_and_psd():
        """On a twisted (non-planar-faced) grid, each 3-D k=1 block still kills
        constants exactly (partition of unity via the divergence theorem) and is
        positive semidefinite. Linear-field consistency and exact symmetry hold
        only for planar faces - a property of MRST's 3-D VEM that PyMRST matches
        exactly (validate_phase411/412), so those are validated against MRST
        rather than asserted analytically here."""
        from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
        nx, amp = 3, 0.2
        g = cart_grid([nx, nx, nx], [float(nx)] * 3)
        c = np.asarray(g.nodes.coords, dtype=float)
        x, y, z, L = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy(), float(nx)
        c[:, 0] += amp * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
        c[:, 1] += 0.75 * amp * np.sin(np.pi * x / L) * np.sin(np.pi * z / L)
        c[:, 2] += 0.5 * amp * np.sin(np.pi * y / L) * np.sin(np.pi * z / L)
        g.nodes.coords = c
        g = compute_vem_geometry(compute_geometry(g))
        rock = make_rock(g, np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 1)
        worst = 0.0
        for Ae, nodes in zip(S["A"], S["dofVec"]):
            worst = max(worst, np.max(np.abs(Ae @ np.ones(nodes.size))))
            assert np.min(np.linalg.eigvalsh((Ae + Ae.T) / 2)) > -1e-9
        assert worst < 1e-9, worst
        print(f"ok test_3d_k1_twisted_kills_constants_and_psd (max |A1| {worst:.2e})")

    def test_3d_k2_planar_quadratic_energy():
        """On a cuboid (planar-faced) 3-D grid the k=2 method is exact: each of
        the 27-dof blocks is symmetric with a one-dim constant kernel and
        reproduces the exact quadratic-field energy int K grad q . grad q for a
        general quadratic and a full 3-D tensor."""
        from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)
        g = compute_vem_geometry(compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0])))
        Kmat = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
        rock = make_rock(g, np.array([[2.0, 0.5, 0.3, 1.5, 0.2, 1.0]]).repeat(g.cells.num, 0), 0.2)
        S = compute_virtual_ip(g, rock, 2)
        coords = np.asarray(g.nodes.coords)
        ecent = np.asarray(g.edges.centroids)
        nn, ne, nf = g.nodes.num, g.edges.num, g.faces.num
        qc = np.array([1.0, -0.5, 0.7, 0.3, -0.4, 0.6])
        q = lambda P: (qc[0] * P[:, 0] ** 2 + qc[1] * P[:, 0] * P[:, 1]
                       + qc[2] * P[:, 0] * P[:, 2] + qc[3] * P[:, 1] ** 2
                       + qc[4] * P[:, 1] * P[:, 2] + qc[5] * P[:, 2] ** 2
                       + 2 * P[:, 0] - P[:, 1] + 0.5 * P[:, 2] + 4)
        gq = lambda P: np.column_stack([
            2 * qc[0] * P[:, 0] + qc[1] * P[:, 1] + qc[2] * P[:, 2] + 2,
            qc[1] * P[:, 0] + 2 * qc[3] * P[:, 1] + qc[4] * P[:, 2] - 1,
            qc[2] * P[:, 0] + qc[4] * P[:, 1] + 2 * qc[5] * P[:, 2] + 0.5])
        areas, vols = np.asarray(g.faces.areas), np.asarray(g.cells.volumes)

        def dofval(gd):
            if gd < nn:
                return q(coords[gd:gd + 1])[0]
            if gd < nn + ne:
                return q(ecent[gd - nn:gd - nn + 1])[0]
            if gd < nn + ne + nf:
                f = gd - nn - ne
                return polygon_int_3d(g, [f], lambda P: q(P)[:, None], 3)[0, 0] / areas[f]
            c = gd - nn - ne - nf
            return polyhedron_int(g, [c], lambda P: q(P)[:, None], 3)[0, 0] / vols[c]

        worst = 0.0
        for c, (Ae, dofs) in enumerate(zip(S["A"], S["dofVec"])):
            assert np.allclose(Ae, Ae.T, atol=1e-9)
            assert np.allclose(Ae @ np.ones(dofs.size), 0.0, atol=1e-8)
            assert np.min(np.linalg.eigvalsh(Ae)) > -1e-9
            dq = np.array([dofval(int(gd)) for gd in dofs])
            integ = polyhedron_int(g, [c],
                                   lambda P: ((gq(P) @ Kmat) * gq(P)).sum(1)[:, None], 3)[0, 0]
            worst = max(worst, abs(dq @ Ae @ dq - integ) / abs(integ))
        assert worst < 1e-8, worst
        print(f"ok test_3d_k2_planar_quadratic_energy (max energy err {worst:.2e})")

    test_local_blocks_are_spd_and_kill_constants()
    test_3d_k1_twisted_kills_constants_and_psd()
    test_3d_k2_planar_quadratic_energy()
    print("\nAll tests passed!")
