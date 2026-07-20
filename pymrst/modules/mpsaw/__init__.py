"""Cell-centered multipoint stress approximation utilities.

This module provides the local algebra PyMRST needs for an MPSA-W style
finite-volume stress path: reconstruct displacement gradients from neighboring
cell centers, symmetrize them into strain, compute isotropic stress, and form
face tractions/cell residual forces.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class MPSAWResult:
    """Recovered MPSA-W style fields."""

    gradient: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    face_tractions: np.ndarray
    residual: np.ndarray


@dataclass
class MPSAInteractionRegion:
    """Node-centered MPSA-W interaction region."""

    node: int
    cells: np.ndarray
    faces: np.ndarray
    cell_faces: np.ndarray
    continuity_points: np.ndarray


@dataclass
class MPSAAssembly:
    """Cell-centered MPSA-W assembly surface."""

    B: sp.csr_array
    rhs: np.ndarray
    regions: list[MPSAInteractionRegion]
    div: sp.csr_array
    gradient_op: sp.csr_array
    strain_op: sp.csr_array
    stress_op: sp.csr_array


def mpsaw_cell_gradient(g, cell_displacement: np.ndarray) -> np.ndarray:
    """Least-squares cell displacement gradient.

    ``cell_displacement`` is cell-centered with shape ``(nc, griddim)``.
    Returns ``grad[c, i, j] = d u_i / d x_j``. Neighboring cells are used first;
    if a boundary cell is underdetermined, nearest cells fill the stencil.
    """
    u = np.asarray(cell_displacement, dtype=float)
    dim = g.griddim
    if u.shape != (g.cells.num, dim):
        raise ValueError("cell_displacement must have shape (cells, griddim)")
    centers = g.cells.centroids[:, :dim]
    grad = np.zeros((g.cells.num, dim, dim))
    neigh = g.faces.neighbors
    cell_neighbors = [[] for _ in range(g.cells.num)]
    for a, b in neigh:
        if a >= 0 and b >= 0:
            cell_neighbors[int(a)].append(int(b))
            cell_neighbors[int(b)].append(int(a))

    for c in range(g.cells.num):
        stencil = list(dict.fromkeys(cell_neighbors[c]))
        if len(stencil) < dim:
            dist = np.linalg.norm(centers - centers[c], axis=1)
            for j in np.argsort(dist):
                if j != c and int(j) not in stencil:
                    stencil.append(int(j))
                if len(stencil) >= dim:
                    break
        dx = centers[stencil] - centers[c]
        du = u[stencil] - u[c]
        sol, *_ = np.linalg.lstsq(dx, du, rcond=None)
        grad[c] = sol.T
    return grad


def mpsaw_interaction_regions(g, eta: float = 0.0) -> list[MPSAInteractionRegion]:
    """Build node-centered interaction regions.

    ``eta`` is MRST's continuity-point interpolation parameter. ``eta=0`` uses
    face centroids, while ``eta=1`` moves the point to the node.
    """
    dim = g.griddim
    regions: list[MPSAInteractionRegion] = []
    face_nodes = [
        g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
        for f in range(g.faces.num)
    ]
    cell_faces = [
        g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
        for c in range(g.cells.num)
    ]
    if getattr(g.cells, "nodes", None) is not None:
        cell_nodes = [
            g.cells.nodes[g.cells.nodePos[c]:g.cells.nodePos[c + 1]]
            for c in range(g.cells.num)
        ]
    else:
        cell_nodes = [
            np.unique(np.concatenate([face_nodes[f] for f in faces_c]))
            for faces_c in cell_faces
        ]
    for node in range(g.nodes.num):
        faces = np.array([
            f for f, nodes in enumerate(face_nodes) if node in set(nodes)
        ], dtype=np.int64)
        cells = []
        cf_pairs = []
        cont = []
        for c, faces_c in enumerate(cell_faces):
            local = [f for f in faces_c if f in set(faces)]
            if not local:
                continue
            if node not in set(cell_nodes[c]):
                continue
            cells.append(c)
            for f in local:
                cf_pairs.append((c, int(f)))
                fc = g.faces.centroids[f, :dim]
                nc = g.nodes.coords[node, :dim]
                cont.append((1.0 - eta) * fc + eta * nc)
        regions.append(MPSAInteractionRegion(
            node=node,
            cells=np.asarray(cells, dtype=np.int64),
            faces=faces,
            cell_faces=np.asarray(cf_pairs, dtype=np.int64).reshape(-1, 2),
            continuity_points=np.asarray(cont, dtype=float).reshape(-1, dim),
        ))
    return regions


def mpsaw_strain(gradient: np.ndarray) -> np.ndarray:
    """Symmetric strain in Voigt form from cell displacement gradients."""
    grad = np.asarray(gradient, dtype=float)
    dim = grad.shape[1]
    eps = 0.5 * (grad + np.swapaxes(grad, 1, 2))
    if dim == 2:
        return np.column_stack([eps[:, 0, 0], eps[:, 1, 1], 2.0 * eps[:, 0, 1]])
    if dim == 3:
        return np.column_stack([
            eps[:, 0, 0], eps[:, 1, 1], eps[:, 2, 2],
            2.0 * eps[:, 1, 2], 2.0 * eps[:, 0, 2], 2.0 * eps[:, 0, 1],
        ])
    raise ValueError("only 2D and 3D grids are supported")


def mpsaw_isotropic_stress(
    strain: np.ndarray,
    young: float | np.ndarray,
    poisson: float | np.ndarray,
) -> np.ndarray:
    """Isotropic stress for MPSA-W strain vectors."""
    eps = np.asarray(strain, dtype=float)
    n = eps.shape[0]
    E = np.broadcast_to(np.asarray(young, dtype=float), n)
    nu = np.broadcast_to(np.asarray(poisson, dtype=float), n)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    if eps.shape[1] == 3:
        tr = eps[:, 0] + eps[:, 1]
        stress = np.zeros_like(eps)
        stress[:, 0] = lam * tr + 2.0 * mu * eps[:, 0]
        stress[:, 1] = lam * tr + 2.0 * mu * eps[:, 1]
        stress[:, 2] = mu * eps[:, 2]
        return stress
    if eps.shape[1] == 6:
        tr = eps[:, 0] + eps[:, 1] + eps[:, 2]
        stress = np.zeros_like(eps)
        stress[:, :3] = lam[:, None] * tr[:, None] + 2.0 * mu[:, None] * eps[:, :3]
        stress[:, 3:] = mu[:, None] * eps[:, 3:]
        return stress
    raise ValueError("strain must have 3 or 6 Voigt components")


def _stress_tensor(stress: np.ndarray) -> np.ndarray:
    s = np.asarray(stress, dtype=float)
    if s.shape[1] == 3:
        return np.stack([
            s[:, 0], s[:, 2],
            s[:, 2], s[:, 1],
        ], axis=1).reshape(-1, 2, 2)
    return np.stack([
        s[:, 0], s[:, 5], s[:, 4],
        s[:, 5], s[:, 1], s[:, 3],
        s[:, 4], s[:, 3], s[:, 2],
    ], axis=1).reshape(-1, 3, 3)


def mpsaw_face_tractions(g, stress: np.ndarray) -> np.ndarray:
    """Return one traction vector per face using adjacent-cell stress averages."""
    dim = g.griddim
    tensors = _stress_tensor(stress)
    tractions = np.zeros((g.faces.num, dim))
    for f, (a, b) in enumerate(g.faces.neighbors):
        if a >= 0 and b >= 0:
            sigma = 0.5 * (tensors[a] + tensors[b])
            normal = g.faces.normals[f, :dim]
        elif a >= 0:
            sigma = tensors[a]
            normal = g.faces.normals[f, :dim]
        else:
            sigma = tensors[b]
            normal = -g.faces.normals[f, :dim]
        area = np.linalg.norm(normal)
        n = normal / area if area > 0.0 else normal
        tractions[f] = sigma @ n
    return tractions


def mpsaw_residual(g, face_tractions: np.ndarray) -> np.ndarray:
    """Assemble cell force imbalance from face tractions."""
    dim = g.griddim
    tr = np.asarray(face_tractions, dtype=float)
    residual = np.zeros((g.cells.num, dim))
    for f, (a, b) in enumerate(g.faces.neighbors):
        area = g.faces.areas[f]
        force = tr[f, :dim] * area
        if a >= 0:
            residual[a] += force
        if b >= 0:
            residual[b] -= force
    return residual


def _flatten_cell(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, dtype=float).reshape(-1)


def _basis_displacement(g, dof: int) -> np.ndarray:
    dim = g.griddim
    u = np.zeros((g.cells.num, dim))
    cell, comp = divmod(dof, dim)
    u[cell, comp] = 1.0
    return u


def mpsaw_divergence_operator(g) -> sp.csr_array:
    """Cell-centered volumetric strain operator."""
    dim = g.griddim
    ndof = g.cells.num * dim
    rows, cols, vals = [], [], []
    for j in range(ndof):
        grad = mpsaw_cell_gradient(g, _basis_displacement(g, j))
        div = np.trace(grad, axis1=1, axis2=2)
        nz = np.flatnonzero(np.abs(div) > 0.0)
        rows.extend(nz.tolist())
        cols.extend([j] * nz.size)
        vals.extend(div[nz].tolist())
    return sp.csr_array((vals, (rows, cols)), shape=(g.cells.num, ndof))


def mpsaw_assemble(
    g,
    young: float | np.ndarray,
    poisson: float | np.ndarray,
    *,
    eta: float = 0.0,
    body_force: np.ndarray | None = None,
) -> MPSAAssembly:
    """Assemble a cell-centered MPSA-W-style linear elasticity operator."""
    dim = g.griddim
    ndof = g.cells.num * dim
    regions = mpsaw_interaction_regions(g, eta=eta)
    rows, cols, vals = [], [], []
    strain_rows, strain_cols, strain_vals = [], [], []
    stress_rows, stress_cols, stress_vals = [], [], []
    nlin = 6 if dim == 3 else 3
    for j in range(ndof):
        u = _basis_displacement(g, j)
        out = mpsaw_recover(g, u, young, poisson)
        residual = out.residual.reshape(-1)
        nz = np.flatnonzero(np.abs(residual) > 0.0)
        rows.extend(nz.tolist())
        cols.extend([j] * nz.size)
        vals.extend(residual[nz].tolist())

        strain = out.strain.reshape(-1)
        snz = np.flatnonzero(np.abs(strain) > 0.0)
        strain_rows.extend(snz.tolist())
        strain_cols.extend([j] * snz.size)
        strain_vals.extend(strain[snz].tolist())

        stress = out.stress.reshape(-1)
        tnz = np.flatnonzero(np.abs(stress) > 0.0)
        stress_rows.extend(tnz.tolist())
        stress_cols.extend([j] * tnz.size)
        stress_vals.extend(stress[tnz].tolist())

    B = sp.csr_array((vals, (rows, cols)), shape=(ndof, ndof))
    strain_op = sp.csr_array(
        (strain_vals, (strain_rows, strain_cols)),
        shape=(g.cells.num * nlin, ndof),
    )
    stress_op = sp.csr_array(
        (stress_vals, (stress_rows, stress_cols)),
        shape=(g.cells.num * nlin, ndof),
    )
    rhs = np.zeros(ndof)
    if body_force is not None:
        bf = np.asarray(body_force, dtype=float)
        if bf.ndim == 1:
            bf = np.broadcast_to(bf, (g.cells.num, dim))
        rhs = (bf[:, :dim] * g.cells.volumes[:, None]).reshape(-1)
    return MPSAAssembly(
        B=B,
        rhs=rhs,
        regions=regions,
        div=mpsaw_divergence_operator(g),
        gradient_op=sp.csr_array((0, ndof)),
        strain_op=strain_op,
        stress_op=stress_op,
    )


def mpsaw_recover(
    g,
    cell_displacement: np.ndarray,
    young: float | np.ndarray,
    poisson: float | np.ndarray,
) -> MPSAWResult:
    """Recover MPSA-W style gradient, strain, stress, tractions and residual."""
    gradient = mpsaw_cell_gradient(g, cell_displacement)
    strain = mpsaw_strain(gradient)
    stress = mpsaw_isotropic_stress(strain, young, poisson)
    tractions = mpsaw_face_tractions(g, stress)
    residual = mpsaw_residual(g, tractions)
    return MPSAWResult(gradient, strain, stress, tractions, residual)
