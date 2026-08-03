"""Virtual element method for 3D linear elasticity (MRST vemmech).

Ports createAugmentedGrid, Enu2C, C2D, calculateQC, VEM3D_div,
VEM_assemble, and VEM_linElast for 3D grids. The assembly follows
Gain, Talischi & Paulino (doi:10.1016/j.cma.2014.05.005):

    K_cell = |E| W_C D W_C' + (I - P_P)' S_E (I - P_P)

with W_C built from the q_i weights of calculateQC and the standard
trace-based stabilization. Displacement DOFs are node-ordered with the
spatial dimension running fastest.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.utils.eigs import multi_symm_eig
from pymrst.utils.mcolon import mcolon
from pymrst.utils.rldecode import rldecode

NLIN3 = 6
NLIN2 = 3


def _nlin(g: Grid) -> int:
    return NLIN3 if g.griddim == 3 else NLIN2


def create_augmented_grid(g: Grid) -> Grid:
    """Add cells.nodes/nodePos (unique, ascending per cell) for VEM."""
    assert g.griddim in (2, 3), "vemmech port supports 2D and 3D grids"
    if g.griddim == 2:
        nodes, node_pos = [], [0]
        for c in range(g.cells.num):
            lo, hi = g.cells.facePos[c], g.cells.facePos[c + 1]
            faces = g.cells.faces[lo:hi]
            cn = np.unique(np.concatenate([
                g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
                for f in faces
            ]))
            xy = g.nodes.coords[cn, :2]
            center = g.cells.centroids[c, :2]
            order = np.argsort(np.arctan2(xy[:, 1] - center[1],
                                          xy[:, 0] - center[0]))
            nodes.extend(cn[order])
            node_pos.append(len(nodes))
        g.cells.nodes = np.asarray(nodes, dtype=np.int64)
        g.cells.nodePos = np.asarray(node_pos, dtype=np.int64)
        if "createAugmentedGrid" not in g.type:
            g.type.append("createAugmentedGrid")
        return g

    cell_no = g.cell_no()
    faces = g.cells.faces
    n_per_face = np.diff(g.faces.nodePos)[faces]
    pair_cells = np.repeat(cell_no, n_per_face)
    pair_nodes = g.faces.nodes[
        mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
    ]
    cn = np.unique(np.column_stack([pair_cells, pair_nodes]), axis=0)
    g.cells.nodes = cn[:, 1]
    g.cells.nodePos = np.concatenate(
        [[0], np.cumsum(np.bincount(cn[:, 0], minlength=g.cells.num))]
    )
    return g


def enu2c(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Isotropic elasticity tensor rows in Voigt form (MRST ``Enu2C``).

    Returns ``(nc, 9)`` for 2D and ``(nc, 36)`` for 3D.
    """
    E = np.asarray(E, dtype=np.float64).ravel()
    nu = np.asarray(nu, dtype=np.float64).ravel()
    nc = E.size
    # Match MRST's 2D branch: plane-strain form in Voigt order
    # [xx, yy, xy].
    if nc and getattr(enu2c, "_force_2d", False):
        raise RuntimeError
    c = np.zeros((nc, 6, 6))
    for i in range(3):
        for j in range(3):
            c[:, i, j] = nu if i != j else 1.0 - nu
        c[:, 3 + i, 3 + i] = (1.0 - 2.0 * nu) / 2.0
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (c * fac[:, None, None]).reshape(nc, 36)


def enu2c_2d(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """2D isotropic elasticity tensor rows, Voigt order ``[xx, yy, xy]``."""
    E = np.asarray(E, dtype=np.float64).ravel()
    nu = np.asarray(nu, dtype=np.float64).ravel()
    c = np.zeros((E.size, 3, 3))
    c[:, 0, 0] = 1.0 - nu
    c[:, 0, 1] = nu
    c[:, 1, 0] = nu
    c[:, 1, 1] = 1.0 - nu
    c[:, 2, 2] = (1.0 - 2.0 * nu) / 2.0
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (c * fac[:, None, None]).reshape(E.size, 9)


def c2d(C: np.ndarray) -> np.ndarray:
    """Scale Voigt tensor entries to strain-energy form (C2D, 2D/3D)."""
    C = np.asarray(C, dtype=np.float64)
    if C.shape[1] == 9:
        factor = np.ones((3, 3))
        factor[:2, 2:] = 2.0
        factor[2:, :2] = 2.0
        factor[2:, 2:] = 4.0
        return C * factor.reshape(1, 9, order="F")
    factor = np.ones((6, 6))
    factor[:3, 3:] = 2.0
    factor[3:, :3] = 2.0
    factor[3:, 3:] = 4.0
    return C * factor.reshape(1, 36, order="F")


def inv_ci(C: np.ndarray) -> np.ndarray:
    r"""$C^{-1} : I$ per cell, in Voigt form (nc, 6) - MRST's `invCi`.

    The strain produced by unit isotropic stress. Its trace is the drained
    compressibility $1/K$, which is what the fixed-stress splitting adds to
    the flow equation, so this is the tensor that makes that scheme stable.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.shape[1] == 9:
        C = C.reshape(-1, 3, 3)
        identity = np.tile([1.0, 1.0, 0.0], (C.shape[0], 1))
        return np.linalg.solve(C, identity[..., None])[..., 0]
    C = C.reshape(-1, 6, 6)
    identity = np.tile([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], (C.shape[0], 1))
    return np.linalg.solve(C, identity[..., None])[..., 0]


def calculate_qf(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """2D node weights (MRST ``calculateQF``), ordered by cell nodes."""
    assert g.griddim == 2
    assert g.cells.nodes is not None, "run create_augmented_grid first"
    qf = np.zeros((g.cells.nodes.size, 2))
    qf_vol = np.zeros(g.cells.nodes.size)
    for c in range(g.cells.num):
        sl = slice(g.cells.nodePos[c], g.cells.nodePos[c + 1])
        pts = g.nodes.coords[g.cells.nodes[sl], :2]
        n = pts.shape[0]
        center = g.cells.centroids[c, :2]
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            edge = p1 - p0
            # For CCW polygons, this is the outward unnormalized normal.
            normal = np.array([edge[1], -edge[0]])
            rel = 0.5 * (p0 + p1) - center
            tetvol = float(np.dot(normal, rel))
            qf[sl.start + i] += 0.5 * normal
            qf[sl.start + (i + 1) % n] += 0.5 * normal
            qf_vol[sl.start + i] += 0.25 * tetvol
            qf_vol[sl.start + (i + 1) % n] += 0.25 * tetvol
    return qf, qf_vol


def vem2d_div(g: Grid) -> sp.csr_array:
    """Discrete VEM divergence for 2D grids."""
    qf, _ = calculate_qf(g)
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.nodePos))
    nodes = g.cells.nodes
    dofs = 2 * nodes[:, None] + np.arange(2)
    return sp.csr_array(
        (qf.ravel(), (np.repeat(cell_no, 2), dofs.ravel())),
        shape=(g.cells.num, 2 * g.nodes.num),
    )


def calculate_qc(g: Grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node weights (calculateQC): qc (cell-node, 3), qf (face-node,),
    qcvol (cell-node,)."""
    node_pos = g.faces.nodePos
    nodes = g.faces.nodes
    nlc = np.diff(node_pos)
    face_no = np.repeat(np.arange(g.faces.num), nlc)

    xn = g.nodes.coords[nodes]
    xbb = np.column_stack([
        np.bincount(face_no, weights=xn[:, d], minlength=g.faces.num)
        for d in range(3)
    ]) / nlc[:, None]
    xb = g.faces.centroids
    n_unit = g.faces.normals / g.faces.areas[:, None]

    # Edge vectors between consecutive (cyclically ordered) face nodes.
    e = np.zeros_like(xn)
    first = node_pos[:-1]
    last = node_pos[1:] - 1
    ind1 = mcolon(first + 1, last + 1)     # positions 2..n per face
    ind2 = mcolon(first, last)             # positions 1..n-1
    ind2 = ind2[~np.isin(ind2, last)] if False else None
    # recompute exactly as MRST: E(1..n-1) = x(2..n) - x(1..n-1)
    i1 = mcolon(first + 1, last + 1)
    i2 = mcolon(first, last)
    inner1 = np.concatenate([np.arange(a + 1, b + 1)
                             for a, b in zip(first, last)]) \
        if g.faces.num else np.zeros(0, np.int64)
    inner2 = np.concatenate([np.arange(a, b)
                             for a, b in zip(first, last)]) \
        if g.faces.num else np.zeros(0, np.int64)
    e[inner2] = xn[inner1] - xn[inner2]
    e[last] = xn[first] - xn[last]

    ne = np.cross(e, n_unit[face_no])
    qf = np.zeros(nodes.size)
    dxbb = (xb - xbb)[face_no]
    qf[first] = np.sum((ne[last] + ne[first]) * dxbb[first], axis=1)
    qf[inner1] = np.sum((ne[inner2] + ne[inner1]) * dxbb[inner1], axis=1)
    qf = (g.faces.areas / nlc)[face_no] + 0.5 * qf

    # Distribute face weights to (cell, node) pairs.
    assert g.cells.nodes is not None, "run create_augmented_grid first"
    cell_no_n = np.repeat(np.arange(g.cells.num), np.diff(g.cells.nodePos))
    lookup = sp.csr_array(
        (np.arange(1, g.cells.nodes.size + 1),
         (cell_no_n, g.cells.nodes)),
        shape=(g.cells.num, g.nodes.num),
    )
    num_cn = g.cells.nodes.size
    qc = np.zeros((num_cn, 3))
    qcvol = np.zeros(num_cn)
    for side in range(2):
        cells = g.faces.neighbors[:, side][face_no]
        qcl = n_unit[face_no] * qf[:, None]
        if side == 1:
            qcl = -qcl
        keep = cells >= 0
        cellk = cells[keep]
        qclk = qcl[keep]
        rcord = g.faces.centroids[face_no][keep] - g.cells.centroids[cellk]
        cn = np.asarray(
            lookup[cellk, nodes[keep]]
        ).ravel().astype(np.int64) - 1
        assert np.all(cn >= 0), "face node missing from cell node list"
        np.add.at(qcvol, cn, np.sum(rcord * qclk, axis=1) / 3.0)
        np.add.at(qc, cn, qclk)
    return qc, qf, qcvol


def vem3d_div(g: Grid) -> sp.csr_array:
    """Discrete VEM divergence: (nc, 3*nn) mapping node displacements to
    volume-integrated cell divergence (VEM3D_div)."""
    faces = g.cells.faces
    cell_no = g.cell_no()
    nfl = np.diff(g.faces.nodePos)[faces]
    inodes = mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
    nodes = g.faces.nodes[inodes]
    sgn = np.where(g.faces.neighbors[faces, 0] == cell_no, 1.0, -1.0)
    nvec = g.faces.normals[faces] * (sgn / g.faces.areas[faces])[:, None]
    _, qf, _ = calculate_qc(g)
    matel = np.repeat(nvec, nfl, axis=0) * qf[inodes][:, None]
    jind = (3 * np.repeat(nodes, 3)
            + np.tile(np.arange(3), nodes.size))
    iind = np.repeat(np.repeat(cell_no, nfl), 3)
    return sp.csr_array(
        (matel.ravel(), (iind, jind)), shape=(g.cells.num, 3 * g.nodes.num)
    )


def vem_div(g: Grid) -> sp.csr_array:
    return vem3d_div(g) if g.griddim == 3 else vem2d_div(g)


def _matrix_grid(g):
    """Return the mechanical Matrix grid used by MRST TEDFM wrappers."""
    return getattr(g, "Matrix", g)


def c2d_tedfm(C: np.ndarray, g=None) -> np.ndarray:
    """TEDFM wrapper for MRST ``C2DTEDFM``.

    MRST's TEDFM mechanics routines operate on ``G.Matrix`` for mechanical
    cells/nodes, but the strain-energy scaling itself is identical to
    ``C2D``. The optional grid argument is accepted for API parity.
    """
    return c2d(C)


def calculate_qf_tedfm(g) -> tuple[np.ndarray, np.ndarray]:
    """2D TEDFM node weights (MRST ``calculateQFTEDFM``).

    The TEDFM functions store the mechanical grid in ``G.Matrix``. Plain
    grids are accepted too, which keeps the helper useful for validation and
    small standalone examples.
    """
    return calculate_qf(_matrix_grid(g))


def calculate_qc_tedfm(g) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D TEDFM node weights (MRST ``calculateQCTEDFM``)."""
    return calculate_qc(_matrix_grid(g))


def vem3d_div_tedfm(g) -> sp.csr_array:
    """3D TEDFM VEM divergence (MRST ``VEM3D_divTEDFM``)."""
    return vem3d_div(_matrix_grid(g))


def vem_div_tedfm(g) -> sp.csr_array:
    """TEDFM VEM divergence (MRST ``VEM_divTEDFM``)."""
    mg = _matrix_grid(g)
    return vem3d_div_tedfm(g) if mg.griddim == 3 else vem2d_div(mg)


def vem_assemble_tedfm(
    g,
    C: np.ndarray,
    alpha_scaling: float = 1.0,
) -> sp.csr_array:
    """TEDFM VEM stiffness matrix (MRST ``VEM_assembleTEDFM``).

    This mirrors MRST's TEDFM assembly convention: all mechanics topology and
    geometry is read from ``G.Matrix``. The local VEM algebra is otherwise the
    same as ordinary ``VEM_assemble``.
    """
    return vem_assemble(_matrix_grid(g), C, alpha_scaling=alpha_scaling)


def vem_lin_elast_tedfm(
    g,
    C: np.ndarray,
    el_bc,
    load: Callable[[np.ndarray], np.ndarray],
    pressure: Optional[np.ndarray] = None,
    alpha_scaling: float = 1.0,
):
    """Solve TEDFM linear elasticity (MRST ``VEM_linElastTEDFM``).

    PyMRST uses the same :class:`ElasticBC` data structure as ``vem_lin_elast``.
    The solve is performed on ``G.Matrix`` and the returned ``extra`` includes
    MRST-style ``disc`` operators for coupled mechanics/flow use.
    """
    mg = _matrix_grid(g)
    uu, extra = vem_lin_elast(
        mg,
        C,
        el_bc,
        load=load,
        pressure=pressure,
        alpha_scaling=alpha_scaling,
    )
    vdiv = vem_div_tedfm(g)
    isdir = extra["isdirdofs"]
    v_dir = extra["V_dir"]
    extra["disc"] = {
        "A": extra["A"],
        "isdirdofs": isdir,
        "rhs": extra["rhs"],
        "V_dir": v_dir,
        "ovol_div": vdiv,
        "gradP": vdiv[:, ~isdir].T,
        "div": vdiv[:, ~isdir],
        "divrhs": vdiv @ v_dir,
    }
    return uu, extra


def _cell_projections(g: Grid, cell: int, qc_all: np.ndarray):
    """The VEM projection matrices of one cell: (NC, NR, WC, WR, dofs).

    WC is the strain projector - cell strain (Voigt) = WC.T @ u_local - and
    depends only on geometry, not on the elasticity tensor, which is why the
    strain and stress operators can be built without C.
    """
    sl = slice(g.cells.nodePos[cell], g.cells.nodePos[cell + 1])
    nodes = g.cells.nodes[sl]
    nn = nodes.size
    dim = g.griddim
    nlin = _nlin(g)
    xb = g.nodes.coords[nodes, :dim] - g.nodes.coords[nodes, :dim].mean(axis=0)

    qc = qc_all[sl, :dim] / g.cells.volumes[cell]     # 2*q_i / |E|

    NC = np.zeros((dim * nn, nlin))
    NR = np.zeros((dim * nn, nlin))
    WC = np.zeros((dim * nn, nlin))
    WR = np.zeros((dim * nn, nlin))
    for d in range(dim):
        NC[d::dim, d] = xb[:, d]
        NR[d::dim, d] = 1.0
        WC[d::dim, d] = qc[:, d]
        WR[d::dim, d] = 1.0 / nn
    q = qc / 2.0
    if dim == 3:
        step = 3
        perms = [(1, 0, 2), (0, 2, 1), (2, 1, 0)]
        nrvs = [(1, -1, 0), (0, 1, -1), (-1, 0, 1)]
        nrcs = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
    else:
        step = 2
        perms = [(1, 0)]
        nrvs = [(1, -1)]
        nrcs = [(1, 1)]
    for m, (pm, nrv, nrc) in enumerate(zip(perms, nrvs, nrcs)):
        for d in range(dim):
            NR[d::step, dim + m] = nrv[d] * xb[:, pm[d]]
            NC[d::step, dim + m] = nrc[d] * xb[:, pm[d]]
            WR[d::step, dim + m] = nrv[d] * q[:, pm[d]]
            WC[d::step, dim + m] = nrc[d] * q[:, pm[d]]

    dofs = (dim * nodes[:, None] + np.arange(dim)).ravel()
    return NC, NR, WC, WR, dofs


def vem_assemble(g: Grid, C: np.ndarray,
                 alpha_scaling: float = 1.0,
                 stabilization=None) -> sp.csr_array:
    """Global VEM stiffness matrix (VEM_assemble, 2D/3D, single block)."""
    qc_all = calculate_qc(g)[0] if g.griddim == 3 else calculate_qf(g)[0]
    D_all = c2d(C)
    nc = g.cells.num
    dim = g.griddim
    nlin = _nlin(g)

    rows, cols, vals = [], [], []
    for cell in range(nc):
        NC, NR, WC, WR, dofs = _cell_projections(g, cell, qc_all)

        PP = NR @ WR.T + NC @ WC.T
        D = D_all[cell].reshape(nlin, nlin)

        # Trace-based stabilization (Gain et al.).
        tr_d = np.trace(D)
        dnc = np.sum(NC * NC, axis=0)
        I = np.eye(dofs.size)
        if stabilization is None:
            alpha = abs(tr_d * g.cells.volumes[cell] / dnc.sum()) * alpha_scaling
            SE = alpha * I
        else:
            supplied = stabilization(cell, dofs) if callable(stabilization) else stabilization
            if isinstance(supplied, (list, tuple)):
                supplied = supplied[cell]
            supplied = supplied.toarray() if sp.issparse(supplied) else np.asarray(supplied)
            if supplied.ndim == 0:
                SE = float(supplied) * I
            elif supplied.shape == (dofs.size, dofs.size):
                SE = supplied
            elif supplied.shape == (dim * g.nodes.num, dim * g.nodes.num):
                SE = supplied[np.ix_(dofs, dofs)]
            else:
                raise ValueError(
                    "custom stabilization must be scalar, one local matrix per "
                    "cell, a global displacement matrix, or callable(cell, dofs)"
                )
        kh = g.cells.volumes[cell] * (WC @ D @ WC.T) \
            + (I - PP).T @ SE @ (I - PP)

        rows.append(np.repeat(dofs, dofs.size))
        cols.append(np.tile(dofs, dofs.size))
        vals.append(kh.ravel())

    ndof = dim * g.nodes.num
    return sp.csr_array(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    )


def vem_strain_operator(g: Grid) -> sp.csr_array:
    """Node displacements -> cell strain in Voigt form.

    MRST's `operators.global_strain` (= WC' * assemb'). Cell c's strain is
    rows ``nlin*c..nlin*(c+1)``. Stress follows as `c2d(C)` applied blockwise, which is
    what `vem_stress_operator` does.
    """
    qc_all = calculate_qc(g)[0] if g.griddim == 3 else calculate_qf(g)[0]
    nlin = _nlin(g)
    dim = g.griddim
    rows, cols, vals = [], [], []
    for cell in range(g.cells.num):
        _, _, WC, _, dofs = _cell_projections(g, cell, qc_all)
        r = nlin * cell + np.arange(nlin)
        rows.append(np.repeat(r, dofs.size))
        cols.append(np.tile(dofs, nlin))
        vals.append(WC.T.ravel())
    return sp.csr_array(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(nlin * g.cells.num, dim * g.nodes.num),
    )


def vem_stress_operator(g: Grid, C: np.ndarray) -> sp.csr_array:
    """Node displacements -> cell effective stress in Voigt form.

    MRST's `operators.global_stress` = D * global_strain.
    """
    nlin = 3 if np.asarray(C).shape[1] == 9 else 6
    D_all = c2d(C).reshape(-1, nlin, nlin)
    blocks = sp.block_diag([D_all[c] for c in range(g.cells.num)],
                           format="csr")
    return (blocks @ vem_strain_operator(g)).tocsr()


def calculate_stress_vem(
    g: Grid,
    uu: np.ndarray,
    C: Optional[np.ndarray] = None,
    strain_op: Optional[sp.spmatrix] = None,
    stress_op: Optional[sp.spmatrix] = None,
    do_patch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell stress and strain from VEM displacement.

    Python counterpart of MRST ``calculateStressVEM``. ``uu`` may be either
    ``(nnode, griddim)`` or a flattened node-major displacement vector. If
    ``stress_op`` is not supplied, ``C`` is required and the operator is built
    with :func:`vem_stress_operator`.
    """
    if do_patch:
        raise NotImplementedError("MRST patchRecovery is not ported")
    dim = g.griddim
    nlin = _nlin(g)
    u = np.asarray(uu, dtype=float)
    if u.ndim == 2:
        u = u[:, :dim].reshape(-1)
    else:
        u = u.ravel()
    if strain_op is None:
        strain_op = vem_strain_operator(g)
    if stress_op is None:
        if C is None:
            raise ValueError("C is required when stress_op is not supplied")
        stress_op = vem_stress_operator(g, C)
    strain = np.asarray(strain_op @ u, dtype=float).reshape(g.cells.num, nlin)
    stress = np.asarray(stress_op @ u, dtype=float).reshape(g.cells.num, nlin)
    if g.griddim == 3:
        stress = stress[:, [0, 1, 2, 3, 5, 4]]
        stress[:, 3:] /= 2.0
        strain = strain[:, [0, 1, 2, 3, 5, 4]]
    else:
        stress[:, 2] /= 2.0
    return stress, strain


def _stress_tensor_from_voigt(g: Grid, stress: np.ndarray) -> np.ndarray:
    stress = np.asarray(stress, dtype=float)
    if g.griddim == 2:
        return np.stack([
            stress[:, 0], stress[:, 2],
            stress[:, 2], stress[:, 1],
        ], axis=1).reshape(-1, 2, 2)
    return np.stack([
        stress[:, 0], stress[:, 5], stress[:, 4],
        stress[:, 5], stress[:, 1], stress[:, 3],
        stress[:, 4], stress[:, 3], stress[:, 2],
    ], axis=1).reshape(-1, 3, 3)


def cal_stress_eigs_vem(
    g: Grid,
    stress: np.ndarray,
    use_c: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues/eigenvectors of VEM stress tensors.

    Mirrors MRST ``calStressEigsVEM``. With ``use_c=True`` the calculation goes
    through the batched symmetric eigensolver; otherwise it loops cell by cell,
    matching MRST's default branch.
    """
    tensors = _stress_tensor_from_voigt(g, stress)
    dim = g.griddim
    if use_c:
        flat = np.concatenate([t.reshape(-1, order="F") for t in tensors])
        vals, vecs = multi_symm_eig(flat, np.full(g.cells.num, dim),
                                    vectors=True)
        return vals.reshape(g.cells.num, dim), \
            vecs.reshape(g.cells.num, dim * dim)
    sigm = np.empty((g.cells.num, dim))
    evec = np.empty((g.cells.num, dim * dim))
    for c, tensor in enumerate(tensors):
        vals, vecs = np.linalg.eig(tensor)
        sigm[c] = vals.real
        evec[c] = vecs.real.reshape(-1, order="F")
    return sigm, evec


@dataclass
class ElasticBC:
    """Boundary conditions for vem_lin_elast.

    disp_nodes/disp_mask/disp_vals: Dirichlet nodes, per-node component
    masks, and prescribed displacements. force_faces/force: boundary
    faces with surface force density (Pa) per face.
    """

    disp_nodes: np.ndarray
    disp_mask: np.ndarray
    disp_vals: np.ndarray
    force_faces: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None


def vem_lin_elast(
    g: Grid,
    C: np.ndarray,
    el_bc: ElasticBC,
    load: Callable[[np.ndarray], np.ndarray],
    pressure: Optional[np.ndarray] = None,
    alpha_scaling: float = 1.0,
):
    """Solve linear elasticity with VEM (VEM_linElast, 2D/3D).

    Returns (uu, extra) where uu is (n_nodes, griddim) displacement and extra
    holds the reduced system, divergence operator and Dirichlet masks
    for poroelastic coupling.
    """
    dim = g.griddim
    S = vem_assemble(g, C, alpha_scaling=alpha_scaling)
    if dim == 3:
        qc_all, qf_all, _ = calculate_qc(g)
        q_cell, q_force = qc_all, qf_all
    else:
        q_cell, qf_vol = calculate_qf(g)
        q_force = np.repeat(g.faces.areas / 2.0, np.diff(g.faces.nodePos))

    bcnodes, first_ix = np.unique(el_bc.disp_nodes, return_index=True)
    u_bc = el_bc.disp_vals[first_ix]
    mask = np.zeros((bcnodes.size, dim), dtype=bool)
    inv = np.searchsorted(bcnodes, el_bc.disp_nodes)
    np.logical_or.at(mask, inv, el_bc.disp_mask.astype(bool))

    ndof = dim * g.nodes.num
    dirdofs = (dim * bcnodes[:, None] + np.arange(dim))[mask]
    u_dir = u_bc[mask]
    isdir = np.zeros(ndof, dtype=bool)
    isdir[dirdofs] = True

    v_dir = np.zeros(ndof)
    v_dir[dirdofs] = u_dir
    rhs = -S @ v_dir

    # Volume load, 'dual_grad_type' method.
    cells_nodes = g.cells.nodes
    nlc = np.diff(g.cells.nodePos)
    xc = np.repeat(g.cells.centroids[:, :dim], nlc, axis=0)
    rel_vec = -(xc - g.nodes.coords[cells_nodes, :dim])
    ll = load(xc) * (q_cell[:, :dim] * rel_vec)
    dofs = (dim * cells_nodes[:, None] + np.arange(dim)).ravel()
    rhs += np.bincount(dofs, weights=ll.ravel(), minlength=ndof)

    div = vem_div(g)
    if pressure is not None:
        rhs += div.T @ pressure

    if el_bc.force_faces is not None and el_bc.force_faces.size > 0:
        faces = el_bc.force_faces
        assert np.all(np.sum(g.faces.neighbors[faces] == -1, axis=1) == 1)
        lnn = np.diff(g.faces.nodePos)[faces]
        inodes = mcolon(g.faces.nodePos[faces], g.faces.nodePos[faces + 1])
        nodes = g.faces.nodes[inodes]
        force = np.repeat(el_bc.force[:, :dim], lnn, axis=0) \
            * q_force[inodes][:, None]
        fdofs = (dim * nodes[:, None] + np.arange(dim)).ravel()
        rhs += np.bincount(fdofs, weights=force.ravel(), minlength=ndof)

    free = ~isdir
    A = S[free][:, free]
    A = (A + A.T) / 2.0
    x = spla.spsolve(A.tocsc(), rhs[free])

    u = np.zeros(ndof)
    u[isdir] = u_dir[np.argsort(dirdofs)] if False else v_dir[isdir]
    u[free] = x
    uu = u.reshape(-1, dim)

    extra = {
        "S": S, "A": A, "isdirdofs": isdir, "V_dir": v_dir,
        "div": div, "rhs": rhs[free],
    }
    return uu, extra


# Keep the historical monolithic module API, but make the MRST-shaped files in
# ``pymrst.modules.vemmech.utils`` the active implementations for these helpers.
from pymrst.modules.vemmech.utils.enu2c import enu2c, enu2c_2d  # noqa: E402,F811
from pymrst.modules.vemmech.utils.c2d import c2d  # noqa: E402,F811
from pymrst.modules.vemmech.utils.c2d_tedfm import c2d_tedfm  # noqa: E402,F811
from pymrst.modules.vemmech.utils.calculate_qf import calculate_qf  # noqa: E402,F811
from pymrst.modules.vemmech.utils.calculate_qf_tedfm import calculate_qf_tedfm  # noqa: E402,F811
from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc  # noqa: E402,F811
from pymrst.modules.vemmech.utils.calculate_qc_tedfm import calculate_qc_tedfm  # noqa: E402,F811
from pymrst.modules.vemmech.utils.calculate_stress_vem import (  # noqa: E402,F811
    calculate_stress_vem,
    vem_strain_operator,
    vem_stress_operator,
)
from pymrst.modules.vemmech.utils.cal_stress_eigs_vem import cal_stress_eigs_vem  # noqa: E402,F811
from pymrst.modules.vemmech.utils.vem2d_div import vem2d_div  # noqa: E402,F811
from pymrst.modules.vemmech.utils.vem3d_div import vem3d_div  # noqa: E402,F811
from pymrst.modules.vemmech.utils.vem_div import vem_div  # noqa: E402,F811
from pymrst.modules.vemmech.utils.vem3d_div_tedfm import vem3d_div_tedfm  # noqa: E402,F811
from pymrst.modules.vemmech.utils.vem_div_tedfm import vem_div_tedfm  # noqa: E402,F811
from pymrst.modules.vemmech.vem_assemble import vem_assemble  # noqa: E402,F811
from pymrst.modules.vemmech.vem_assemble_tedfm import vem_assemble_tedfm  # noqa: E402,F811
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast  # noqa: E402,F811
from pymrst.modules.vemmech.vem_lin_elast_tedfm import vem_lin_elast_tedfm  # noqa: E402,F811


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry

    def _column(nz=8, twist=False):
        g = cart_grid([2, 2, nz], [2.0, 2.0, float(nz)])
        if twist:
            pts = g.nodes.coords
            interior = np.all((pts[:, :2] > 1e-9)
                              & (pts[:, :2] < 2 - 1e-9), axis=1)
            pts[interior, 0] += 0.15 * np.sin(pts[interior, 2])
            pts[interior, 1] += 0.1 * np.cos(pts[interior, 0])
        g = compute_geometry(g)
        return create_augmented_grid(g)

    def test_augmented_grid():
        """Each hexahedral cell lists its 8 nodes in ascending order."""
        g = _column()
        assert np.all(np.diff(g.cells.nodePos) == 8)
        for c in range(g.cells.num):
            nd = g.cells.nodes[g.cells.nodePos[c]:g.cells.nodePos[c + 1]]
            assert np.all(np.diff(nd) > 0)
        print("ok test_augmented_grid")

    def test_qc_partition_of_unity():
        """qc sums to zero per cell; qcvol sums to cell volumes."""
        g = _column(twist=True)
        qc, qf, qcvol = calculate_qc(g)
        cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.nodePos))
        for d in range(3):
            s = np.bincount(cell_no, weights=qc[:, d])
            assert np.max(np.abs(s)) < 1e-10   # closed surface
        vol = np.bincount(cell_no, weights=qcvol)
        assert np.allclose(vol, g.cells.volumes, rtol=1e-10)
        # qf partitions face areas
        face_no = np.repeat(np.arange(g.faces.num),
                            np.diff(g.faces.nodePos))
        area = np.bincount(face_no, weights=qf)
        assert np.allclose(area, g.faces.areas, rtol=1e-10)
        print("ok test_qc_partition_of_unity")

    def test_patch_uniaxial_compression():
        """Homogeneous column under top load reproduces Hooke's law."""
        nz = 8
        g = _column(nz)
        E, nu = 1e9, 0.25
        C = enu2c(np.full(g.cells.num, E), np.full(g.cells.num, nu))
        bd = np.any(g.faces.neighbors == -1, axis=1)
        fc = g.faces.centroids
        top = np.flatnonzero(bd & (fc[:, 2] < 1e-9))
        bottom_n = np.unique(np.abs(
            g.faces.nodes[mcolon(
                g.faces.nodePos[np.flatnonzero(bd & (fc[:, 2] > nz - 1e-9))],
                g.faces.nodePos[np.flatnonzero(bd
                                               & (fc[:, 2] > nz - 1e-9)) + 1])]
        ))
        # roller sides: fix normal components on x/y boundary nodes
        pts = g.nodes.coords
        nodes_x = np.flatnonzero((pts[:, 0] < 1e-9) | (pts[:, 0] > 2 - 1e-9))
        nodes_y = np.flatnonzero((pts[:, 1] < 1e-9) | (pts[:, 1] > 2 - 1e-9))
        dn = np.concatenate([bottom_n, nodes_x, nodes_y])
        dm = np.vstack([
            np.tile([False, False, True], (bottom_n.size, 1)),
            np.tile([True, False, False], (nodes_x.size, 1)),
            np.tile([False, True, False], (nodes_y.size, 1)),
        ])
        bc = ElasticBC(disp_nodes=dn, disp_mask=dm,
                       disp_vals=np.zeros((dn.size, 3)),
                       force_faces=top,
                       force=np.tile([0.0, 0.0, 1e6], (top.size, 1)))
        uu, extra = vem_lin_elast(
            g, C, bc, load=lambda x: np.zeros_like(x)
        )
        # Confined (oedometer) modulus: sigma = (lambda + 2 mu) eps
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        eps = 1e6 / (lam + 2 * mu)
        top_nodes = np.flatnonzero(pts[:, 2] < 1e-9)
        assert np.allclose(uu[top_nodes, 2], eps * nz, rtol=1e-8), (
            f"top uz {uu[top_nodes, 2].mean():.4e} vs {eps * nz:.4e}"
        )
        # displacement varies linearly with depth
        assert np.allclose(uu[:, 2], eps * (nz - pts[:, 2]), rtol=1e-6,
                           atol=1e-12 * eps * nz)
        print("ok test_patch_uniaxial_compression")

    def test_div_consistency():
        """VEM divergence is exact for linear fields on planar faces and
        translation-invariant on twisted (non-planar) grids."""
        g = _column(6)
        div = vem3d_div(g)
        u = np.column_stack([
            0.3 * g.nodes.coords[:, 0],
            -0.1 * g.nodes.coords[:, 1],
            0.2 * g.nodes.coords[:, 2],
        ]).ravel()
        assert np.allclose(div @ u, 0.4 * g.cells.volumes, rtol=1e-12)

        gt = _column(6, twist=True)
        divt = vem3d_div(gt)
        ut = np.column_stack([
            0.3 * gt.nodes.coords[:, 0],
            -0.1 * gt.nodes.coords[:, 1],
            0.2 * gt.nodes.coords[:, 2],
        ]).ravel()
        rel = np.abs(divt @ ut - 0.4 * gt.cells.volumes) / gt.cells.volumes
        assert rel.max() < 1e-4       # non-planar face quadrature error
        rigid = np.tile([1.0, -2.0, 3.0], gt.nodes.num)
        assert np.max(np.abs(divt @ rigid)) < 1e-10
        print("ok test_div_consistency")

    def test_2d_qf_and_div_consistency():
        """2D calculateQF partitions volumes and exact linear divergence."""
        g = create_augmented_grid(
            compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        )
        qf, qf_vol = calculate_qf(g)
        cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.nodePos))
        for d in range(2):
            assert np.allclose(np.bincount(cell_no, weights=qf[:, d]), 0.0)
        assert np.allclose(
            np.bincount(cell_no, weights=qf_vol), g.cells.volumes
        )

        div = vem2d_div(g)
        u = np.column_stack([
            0.3 * g.nodes.coords[:, 0],
            -0.1 * g.nodes.coords[:, 1],
        ]).ravel()
        assert np.allclose(div @ u, 0.2 * g.cells.volumes, rtol=1e-12)
        print("ok test_2d_qf_and_div_consistency")

    def test_2d_patch_compression_runs():
        """Plane-strain square under a top load gives a stable 2D solve."""
        g = create_augmented_grid(
            compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        )
        E, nu = 1e9, 0.25
        C = enu2c_2d(np.full(g.cells.num, E), np.full(g.cells.num, nu))
        pts = g.nodes.coords[:, :2]
        bottom = np.flatnonzero(pts[:, 1] < 1e-9)
        left = np.flatnonzero(pts[:, 0] < 1e-9)
        dn = np.concatenate([bottom, left])
        dm = np.vstack([
            np.tile([False, True], (bottom.size, 1)),
            np.tile([True, False], (left.size, 1)),
        ])
        bd = np.any(g.faces.neighbors == -1, axis=1)
        top = np.flatnonzero(bd & (g.faces.centroids[:, 1] > 2.0 - 1e-9))
        bc = ElasticBC(disp_nodes=dn, disp_mask=dm,
                       disp_vals=np.zeros((dn.size, 2)),
                       force_faces=top,
                       force=np.tile([0.0, -1e6], (top.size, 1)))
        uu, extra = vem_lin_elast(
            g, C, bc, load=lambda x: np.zeros_like(x)
        )
        assert uu.shape == (g.nodes.num, 2)
        assert extra["S"].shape == (2 * g.nodes.num, 2 * g.nodes.num)
        assert np.isfinite(uu).all()
        print("ok test_2d_patch_compression_runs")

    def test_calculate_stress_vem_matches_operators():
        """Standalone stress utility is the existing global operators exposed."""
        g = create_augmented_grid(
            compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        )
        C = enu2c_2d(np.full(g.cells.num, 1e9),
                     np.full(g.cells.num, 0.25))
        u = np.column_stack([
            0.01 * g.nodes.coords[:, 0],
            -0.02 * g.nodes.coords[:, 1],
        ])
        stress, strain = calculate_stress_vem(g, u, C)
        assert np.allclose(strain.ravel(), vem_strain_operator(g) @ u.ravel())
        raw_stress = (vem_stress_operator(g, C) @ u.ravel()).reshape(
            g.cells.num, 3
        )
        raw_stress[:, 2] /= 2.0
        assert np.allclose(stress, raw_stress)
        print("ok test_calculate_stress_vem_matches_operators")

    def test_cal_stress_eigs_vem_2d():
        """Principal stresses of diagonal 2D stress tensors are exact."""
        g = create_augmented_grid(
            compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
        )
        stress = np.array([[3.0, 1.0, 0.0], [2.0, 5.0, 0.0]])
        sigm, evec = cal_stress_eigs_vem(g, stress, use_c=True)
        assert np.allclose(sigm, np.array([[1.0, 3.0], [2.0, 5.0]]))
        assert evec.shape == (g.cells.num, 4)
        print("ok test_cal_stress_eigs_vem_2d")

    def test_tedfm_wrappers_use_matrix_grid():
        """TEDFM helpers operate on G.Matrix like MRST's fracture wrappers."""
        g = create_augmented_grid(
            compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        )
        gw = SimpleNamespace(Matrix=g, griddim=g.griddim)
        C = enu2c(np.full(g.cells.num, 1e9), np.full(g.cells.num, 0.25))
        assert np.allclose(calculate_qc_tedfm(gw)[0], calculate_qc(g)[0])
        assert np.allclose(vem3d_div_tedfm(gw).toarray(),
                           vem3d_div(g).toarray())
        assert np.allclose(vem_div_tedfm(gw).toarray(), vem_div(g).toarray())
        assert np.allclose(vem_assemble_tedfm(gw, C).toarray(),
                           vem_assemble(g, C).toarray())
        print("ok test_tedfm_wrappers_use_matrix_grid")

    test_augmented_grid()
    test_qc_partition_of_unity()
    test_patch_uniaxial_compression()
    test_div_consistency()
    test_2d_qf_and_div_consistency()
    test_2d_patch_compression_runs()
    test_calculate_stress_vem_matches_operators()
    test_cal_stress_eigs_vem_2d()
    test_tedfm_wrappers_use_matrix_grid()
    print("\nAll tests passed!")
