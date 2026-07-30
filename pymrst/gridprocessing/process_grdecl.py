"""Corner-point grid processing, ported from MRST processGRDECL.m.

Computes topology and geometry of general corner-point grids given in
ECLIPSE SPECGRID/COORD/ZCORN format, including fault-face subdivision,
inactive cells (ACTNUM), and pinched layers:

a) corner coordinates of every cell (buildCornerPtNodes),
b) unique points,
c) auxiliary top/bottom layers so faulted outer boundaries process
   cleanly,
d) i/j-direction connectivity: fully matching pillar-pair stacks become
   regular quads (findFaces); mismatching stacks are subdivided by the
   interval-overlap walk of findConnections and the envelope/intersection
   construction of computeFaceGeometry (findFaults),
e) k-direction faces with connectivity restored across pinched layers
   (findVerticalFaces),
f) cell assembly and removal of auxiliary/inactive/pinched cells.

Internally 1-based with 0 = void, exactly like the MATLAB code; the
result is converted to the 0-based PyMRST Grid at the end.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes
from pymrst.utils.gridtools import mcolon, rlencode


def _flat(a: np.ndarray) -> np.ndarray:
    return np.ravel(a, order="F")


def _F(a: np.ndarray, shape) -> np.ndarray:
    return np.reshape(a, shape, order="F")


def _index(i, j, k, sz) -> np.ndarray:
    """MATLAB index(): F-order linear indices of an ndgrid of ranges."""
    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
    # NumPy promotes empty ``meshgrid`` results to float for some mixtures of
    # ranges. Thin corner-point fixtures legitimately produce an empty axis.
    indices = tuple(np.asarray(v, dtype=np.int64) for v in (ii, jj, kk))
    return _flat(np.ravel_multi_index(indices, sz, order="F"))


def build_corner_pt_nodes(
    grdecl: Dict, coincidence_tol: float = 100 * np.finfo(float).eps
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner coordinates X, Y, Z of shape (2nx, 2ny, 2nz), F-ordered."""
    nx, ny, nz = grdecl["cartDims"]
    shape = (2 * nx, 2 * ny, 2 * nz)

    pillar_ix = _F(np.arange((nx + 1) * (ny + 1)), (nx + 1, ny + 1))
    line_ix = np.zeros(shape, dtype=np.int64)
    line_ix[0::2, 0::2, :] = pillar_ix[:nx, :ny, None]
    line_ix[1::2, 0::2, :] = pillar_ix[1:, :ny, None]
    line_ix[0::2, 1::2, :] = pillar_ix[:nx, 1:, None]
    line_ix[1::2, 1::2, :] = pillar_ix[1:, 1:, None]

    lines = np.reshape(grdecl["COORD"], (-1, 6))[_flat(line_ix)]
    zcorn = np.asarray(grdecl["ZCORN"], dtype=np.float64).ravel()

    denom = lines[:, 5] - lines[:, 2]
    collapsed = np.abs(denom) < abs(coincidence_tol)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (zcorn - lines[:, 2]) / denom
    t[collapsed] = 0.0

    x = lines[:, 0] + t * (lines[:, 3] - lines[:, 0])
    y = lines[:, 1] + t * (lines[:, 4] - lines[:, 1])
    return _F(x, shape), _F(y, shape), _F(zcorn, shape)


class _PreGrid:
    """Accumulates faces during processing (1-based, 0 = void)."""

    def __init__(self, coords: np.ndarray):
        self.coords = coords                       # (npts, 3), grows
        self.face_nodes: List[np.ndarray] = []     # 1-based point numbers
        self.num_nodes: List[np.ndarray] = []
        self.neighbors: List[np.ndarray] = []      # (n, 2), 0 = void
        self.tag: List[np.ndarray] = []            # 1 = fault face
        self.cell_tags: List[np.ndarray] = []      # (n, 2) direction tags

    def append(self, nodes, nnodes, neigh, tag, cell_tags):
        if len(nnodes) == 0:
            return
        self.face_nodes.append(np.asarray(nodes, dtype=np.int64))
        self.num_nodes.append(np.asarray(nnodes, dtype=np.int64))
        self.neighbors.append(np.asarray(neigh, dtype=np.int64))
        self.tag.append(np.full(len(nnodes), tag, dtype=np.int64))
        self.cell_tags.append(
            np.tile(np.asarray(cell_tags, dtype=np.int64), (len(nnodes), 1))
        )


def _find_faces(pre: _PreGrid, P, B, actnum_f, tags):
    """Regular i-direction faces on fully matching pillar-pair stacks."""
    sz = P.shape
    szB = B.shape
    Pf, Bf = _flat(P), _flat(B)
    di, dj, dk = 1, sz[0], sz[0] * sz[1]

    k = _index(range(1, sz[0] - 1, 2), range(0, sz[1], 2),
               range(0, sz[2] - 1, 2), sz)
    f = np.column_stack([Pf[k], Pf[k + dj], Pf[k + dj + dk], Pf[k + dk]])
    k = k + di
    g = np.column_stack([Pf[k], Pf[k + dj], Pf[k + dj + dk], Pf[k + dk]])

    c1 = Bf[_index(range(szB[0] - 1), range(szB[1]), range(szB[2]), szB)]
    c2 = Bf[_index(range(1, szB[0]), range(szB[1]), range(szB[2]), szB)]

    match = _F(np.all(f == g, axis=1),
               (sz[0] // 2 - 1, sz[1] // 2, sz[2] // 2))
    h = _flat(np.repeat(np.all(match, axis=2)[:, :, None],
                        sz[2] // 2, axis=2))
    c1, c2, f = c1[h], c2[h], f[h]

    # Regular boundary faces on the two outermost pillar planes.
    k = _index([0, sz[0] - 1], range(0, sz[1], 2), range(0, sz[2] - 1, 2), sz)
    fb = np.column_stack([Pf[k], Pf[k + dj], Pf[k + dj + dk], Pf[k + dk]])
    cb = Bf[_index([0, szB[0] - 1], range(szB[1]), range(szB[2]), szB)]
    tag_b = np.tile(np.asarray(tags), cb.size // 2)

    f = np.vstack([f, fb])
    c1 = np.concatenate([c1, cb * (tag_b == tags[1])])
    c2 = np.concatenate([c2, cb * (tag_b == tags[0])])

    # Inactive neighbors become void; drop faces void on both sides,
    # zero-area faces, and repeated pinch nodes.
    m = c1 != 0
    c1[m] = c1[m] * actnum_f[c1[m] - 1]
    m = c2 != 0
    c2[m] = c2[m] * actnum_f[c2[m] - 1]

    degenerate = (f[:, 0] == f[:, 3]) & (f[:, 1] == f[:, 2])
    keep = ((c1 != 0) | (c2 != 0)) & ~degenerate
    f, c1, c2 = f[keep], c1[keep], c2[keep]

    fnan = f.astype(float)
    fnan[f[:, 0] == f[:, 3], 0] = np.nan
    fnan[f[:, 1] == f[:, 2], 1] = np.nan
    valid = ~np.isnan(fnan)
    nn = valid.sum(axis=1)
    nodes = fnan[valid].astype(np.int64)

    pre.append(nodes, nn, np.column_stack([c1, c2]), 0, tags)


def _find_vertical_faces(pre: _PreGrid, P, actnum, tags):
    """k-direction faces; pinched layers merge via run-length encoding."""
    sz = P.shape
    Pf = _flat(P)
    nb = (sz[0] // 2, sz[1] // 2, sz[2] // 2)
    B = _F(np.arange(1, np.prod(nb) + 1), nb)

    ii, jj, kk = np.meshgrid(range(0, sz[0], 2), range(0, sz[1], 2),
                             range(sz[2]), indexing="ij")
    lin = np.ravel_multi_index((ii, jj, kk), sz, order="F")
    k = _flat(np.transpose(lin, (2, 0, 1)))
    f = np.column_stack([Pf[k], Pf[k + 1], Pf[k + 1 + sz[0]], Pf[k + sz[0]]])

    faces, n = rlencode(f)

    c = np.zeros(2 * actnum.size + 1, dtype=np.int64)
    c[1::2] = _flat(np.transpose(B * actnum, (2, 0, 1)))
    p = c[np.concatenate([[0], np.cumsum(n)])]
    new = np.column_stack([p[:-1], p[1:]])
    keep = np.any(new != 0, axis=1)

    pre.append(faces[keep].ravel(), np.full(keep.sum(), 4),
               new[keep], 0, tags)


def _find_connections(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    """Interval-overlap walk along concatenated pillar-pair stacks."""
    out: List[Tuple[int, int]] = []
    j1 = j2 = 0
    nb = zb.shape[0]
    for i in range(za.shape[0] - 1):
        j = min(j1, j2)
        while j < nb and (zb[j, 0] < za[i + 1, 0] or zb[j, 1] < za[i + 1, 1]):
            if _do_intersect(za[i], za[i + 1], zb[j], zb[j + 1]):
                out.append((i, j))
            if zb[j, 0] < za[i + 1, 0]:
                j1 = j
            if zb[j, 1] < za[i + 1, 1]:
                j2 = j
            j += 1
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _do_intersect(za1, za2, zb1, zb2) -> bool:
    if (za1[0] == za2[0] and za1[1] == za2[1]) or \
       (zb1[0] == zb2[0] and zb1[1] == zb2[1]):
        return False
    return bool(
        max(za1[0], zb1[0]) < min(za2[0], zb2[0])
        or max(za1[1], zb1[1]) < min(za2[1], zb2[1])
        or (za1[0] - zb1[0]) * (za1[1] - zb1[1]) < 0
        or (za2[0] - zb2[0]) * (za2[1] - zb2[1]) < 0
    )


def _intersection(la, lb, pts):
    """Coordinates where line pairs (rows of point numbers) cross."""
    z = pts[:, 2]
    za, zb = z[la - 1], z[lb - 1]
    t = (zb[:, 0] - za[:, 0]) / ((za[:, 1] - za[:, 0]) - (zb[:, 1] - zb[:, 0]))
    out = np.zeros((la.shape[0], 3))
    for d in range(3):
        v = pts[:, d]
        va = v[la - 1]
        out[:, d] = (va[:, 1] - va[:, 0]) * t + va[:, 0]
    return out


def _compute_face_geometry(a, b, C, points):
    """Fault sub-face polygons (computeFaceGeometry port)."""
    pa = np.hstack([a[C[:, 0]], a[C[:, 0] + 1]])
    pb = np.hstack([b[C[:, 1]], b[C[:, 1] + 1]])
    n = pa.shape[0]
    if n < 1:
        return points, np.zeros(0, dtype=np.int64), np.zeros(0, np.int64)

    if np.all(pa == pb):
        return points, np.full(n, 4, dtype=np.int64), pa.ravel()

    z = points[:, 2]
    az = z[pa - 1]
    bz = z[pb - 1]

    # Envelope points on the pillars: max of bottoms, min of tops.
    pick_b = np.hstack([az[:, :2] < bz[:, :2], az[:, 2:] > bz[:, 2:]])
    I = pa.copy()
    I[pick_b] = pb[pick_b]

    # Candidate line-pair intersections (4 blocks, F-order like MATLAB).
    PP = np.vstack([pa[:, :2], pa[:, 2:], pa[:, :2], pa[:, 2:]])
    QQ = np.vstack([pb[:, :2], pb[:, 2:], pb[:, 2:], pb[:, :2]])
    cross = (z[PP[:, 0] - 1] - z[QQ[:, 0] - 1]) \
        * (z[PP[:, 1] - 1] - z[QQ[:, 1] - 1]) < 0
    q = _intersection(PP[cross], QQ[cross], points)
    q, inv = np.unique(q, axis=0, return_inverse=True)

    f = np.full(4 * n, np.nan)
    f[cross] = inv + points.shape[0] + 1        # 1-based new point numbers
    f = np.column_stack([f[:n], f[n:2 * n], f[2 * n:3 * n], f[3 * n:]])
    points = np.vstack([points, q])

    J = np.full((n, 8), np.nan)
    If = I.astype(float)
    If[I[:, 0] == I[:, 2], 0] = np.nan
    If[I[:, 1] == I[:, 3], 1] = np.nan
    J[:, [0, 2, 4, 6]] = If[:, [0, 1, 3, 2]]
    J[:, 1] = f[:, 0]                            # A12 x B12
    J[:, 5] = f[:, 1]                            # A34 x B34

    bot_a_top_b = ~np.isnan(f[:, 2])             # A12 x B34
    top_a_bot_b = ~np.isnan(f[:, 3])             # A34 x B12

    left = az[:, 0] > bz[:, 2]
    ind = bot_a_top_b & left
    J[ind, 0] = J[ind, 6] = np.nan
    J[ind, 7] = f[ind, 2]
    ind = bot_a_top_b & ~left
    J[ind, 2] = J[ind, 4] = np.nan
    J[ind, 3] = f[ind, 2]

    left = bz[:, 0] > az[:, 2]
    ind = top_a_bot_b & left
    J[ind, 0] = J[ind, 6] = np.nan
    J[ind, 7] = f[ind, 3]
    ind = top_a_bot_b & ~left
    J[ind, 2] = J[ind, 4] = np.nan
    J[ind, 3] = f[ind, 3]

    flat = np.hstack([J, np.full((n, 1), np.inf)]).ravel()
    flat = flat[~np.isnan(flat)]
    vals, _ = rlencode(flat)                     # drop repeated pinch nodes
    num_nodes = np.diff(np.concatenate([[0],
                        np.flatnonzero(np.isinf(vals)) + 1])) - 1
    corners = vals[np.isfinite(vals)].astype(np.int64)
    return points, num_nodes, corners


def _find_faults(pre: _PreGrid, P, B, actnum_f, tags):
    """Faulted i-direction stacks: subdivide by interval overlaps."""
    P3 = np.transpose(P, (2, 0, 1))
    B3 = np.transpose(B, (2, 0, 1))
    szP = P3.shape
    szB = B3.shape
    Pf, Bf = _flat(P3), _flat(B3)
    di, dj = szP[0], szP[0] * szP[1]

    k = _index(range(szP[0]), range(1, szP[1] - 1, 2),
               range(0, szP[2], 2), szP)
    a = np.column_stack([Pf[k], Pf[k + dj]])
    b = np.column_stack([Pf[k + di], Pf[k + di + dj]])

    cA = Bf[_index(range(szB[0]), range(szB[1] - 1), range(szB[2]), szB)]
    cB = Bf[_index(range(szB[0]), range(1, szB[1]), range(szB[2]), szB)]

    sz2 = (szP[0], szP[1] // 2 - 1, szP[2] // 2)
    match = _F(np.all(a == b, axis=1), sz2)
    h = _flat(np.repeat(np.all(match, axis=0)[None, :, :], szP[0], axis=0))

    a, b = a[~h], b[~h]
    hc = h[::2]
    cA, cB = cA[~hc], cB[~hc]
    if a.shape[0] == 0:
        return

    # Per-stack z offsets keep stacks separated in one flat walk.
    dz = pre.coords[:, 2].max() - pre.coords[:, 2].min() + 1.0
    auxz = _F(np.arange(1, np.prod(sz2[1:]) + 1), sz2[1:]) * dz
    dZ = _flat(np.repeat(auxz[None, :, :], szP[0], axis=0))
    dZ = dZ[~h]

    act_a = actnum_f[np.repeat(cA, 2) - 1] == 1
    act_b = actnum_f[np.repeat(cB, 2) - 1] == 1
    a, dZa = a[act_a], dZ[act_a]
    b, dZb = b[act_b], dZ[act_b]
    cA = cA[actnum_f[cA - 1] == 1]
    cB = cB[actnum_f[cB - 1] == 1]

    z = pre.coords[:, 2]
    za = z[a - 1] + dZa[:, None]
    zb = z[b - 1] + dZb[:, None]

    C = _find_connections(za, zb)
    if C.shape[0] == 0:
        return

    ka = np.zeros(a.shape[0] - 1, dtype=np.int64)
    ka[0::2] = cA
    kb = np.zeros(b.shape[0] - 1, dtype=np.int64)
    kb[0::2] = cB
    new = np.column_stack([ka[C[:, 0]], kb[C[:, 1]]])
    keep = np.any(new != 0, axis=1)
    new = new[keep]

    pre.coords, num_nodes, corners = _compute_face_geometry(
        a, b, C[keep], pre.coords
    )
    pre.append(corners, num_nodes, new, 1, tags)


def process_grdecl(grdecl: Dict, tolerance: float = 0.0,
                   check_grid: bool = True) -> Grid:
    """Build an unstructured Grid from SPECGRID/COORD/ZCORN(/ACTNUM)."""
    grdecl = dict(grdecl)
    nx, ny, nz = grdecl["cartDims"]
    if tolerance > 0:
        grdecl["ZCORN"] = tolerance * np.round(
            np.asarray(grdecl["ZCORN"]) / tolerance
        )

    X, Y, Z = build_corner_pt_nodes(grdecl)

    if "ACTNUM" in grdecl and grdecl["ACTNUM"] is not None:
        act = _F(np.asarray(grdecl["ACTNUM"]).ravel() != 0, (nx, ny, nz))
    else:
        act = np.ones((nx, ny, nz), dtype=bool)

    # Axis orientation checks (build_coordinates).
    reverse = [False, False, False]
    a8 = np.repeat(np.repeat(np.repeat(act, 2, 0), 2, 1), 2, 2)
    zm = np.where(a8, Z, np.nan)
    dzm = np.diff(zm, axis=2)
    with np.errstate(invalid="ignore"):
        if not np.any(dzm > 0):
            Z = -Z
            zm = np.where(a8, Z, np.nan)
            dzm = np.diff(zm, axis=2)
            reverse[2] = True
        if np.any(dzm < 0):
            import warnings
            warnings.warn("non-monotone corner-point depths detected")

    v1 = np.array([X[-1, 0, 0] - X[0, 0, 0], Y[-1, 0, 0] - Y[0, 0, 0], 0.0])
    v2 = np.array([X[0, -1, 0] - X[0, 0, 0], Y[0, -1, 0] - Y[0, 0, 0], 0.0])
    v3 = np.array([0.0, 0.0, Z[0, 0, -1] - Z[0, 0, 0]])
    if np.dot(np.cross(v1, v2), v3) < 0:
        Y = -Y
        reverse[1] = True

    X = np.where(np.isnan(X), np.inf, X)
    Y = np.where(np.isnan(Y), np.inf, Y)

    # Auxiliary top/bottom layers.
    minz, maxz = Z.min(), Z.max()
    e = np.zeros((2 * nx, 2 * ny, 1))
    Z = np.concatenate([minz - 2 + e, minz - 1 + e, Z,
                        maxz + 1 + e, maxz + 2 + e], axis=2)
    X = X[:, :, [0, 0] + list(range(2 * nz)) + [2 * nz - 1, 2 * nz - 1]]
    Y = Y[:, :, [0, 0] + list(range(2 * nz)) + [2 * nz - 1, 2 * nz - 1]]

    act_aux = np.concatenate(
        [np.ones((nx, ny, 1), bool), act, np.ones((nx, ny, 1), bool)], axis=2
    )
    dims_aux = (nx, ny, nz + 2)

    # Unique points, natural ordering by (z, y, x).
    zyx = np.column_stack([_flat(Z), _flat(Y), _flat(X)])
    coords_zyx, from_map = np.unique(zyx, axis=0, return_inverse=True)
    coords = coords_zyx[:, ::-1].copy()
    P = _F(from_map + 1, Z.shape)                      # 1-based point ids
    B = _F(np.arange(1, np.prod(dims_aux) + 1), dims_aux)

    pre = _PreGrid(coords)
    act_f = _flat(act_aux).astype(np.int64)

    # i-direction, then j (via permutation), then k.
    _find_faces(pre, P, B, act_f, (1, 2))
    _find_faults(pre, P, B, act_f, (1, 2))

    n_before_j = sum(len(v) for v in pre.num_nodes)
    Pj = np.transpose(P, (1, 0, 2))
    Bj = np.transpose(B, (1, 0, 2))
    _find_faces(pre, Pj, Bj, act_f, (3, 4))
    _find_faults(pre, Pj, Bj, act_f, (3, 4))
    n_after_j = sum(len(v) for v in pre.num_nodes)

    act_k = act_aux.copy()
    _find_vertical_faces(pre, P, act_k, (5, 6))

    # Assemble flat arrays.
    face_nodes = np.concatenate(pre.face_nodes)
    num_nodes = np.concatenate(pre.num_nodes)
    neighbors = np.vstack(pre.neighbors)
    ftag = np.concatenate(pre.tag)
    cell_tags = np.vstack(pre.cell_tags)
    node_pos = np.concatenate([[0], np.cumsum(num_nodes)])

    # Reverse node order of j-faces (permutation flipped orientation).
    for fidx in range(n_before_j, n_after_j):
        s, t = node_pos[fidx], node_pos[fidx + 1]
        face_nodes[s:t] = face_nodes[s:t][::-1]

    # Remove auxiliary/inactive/pinched cells.
    act_final = act_aux.copy()
    act_final[:, :, 0] = False
    act_final[:, :, -1] = False
    act_ff = _flat(act_final)

    ncell_aux = np.prod(dims_aux)
    counts = np.bincount(neighbors[neighbors > 0], minlength=ncell_aux + 1)
    keep_cell = act_ff & (counts[1:] > 0)

    new_id = np.zeros(ncell_aux + 1, dtype=np.int64)
    new_id[1:][keep_cell] = np.arange(1, keep_cell.sum() + 1)
    neighbors = new_id[neighbors]

    keep_face = np.any(neighbors != 0, axis=1)
    neighbors = neighbors[keep_face]
    ftag = ftag[keep_face]
    cell_tags = cell_tags[keep_face]
    sel = mcolon(node_pos[:-1][keep_face], node_pos[1:][keep_face])
    face_nodes = face_nodes[sel]
    num_nodes = num_nodes[keep_face]
    node_pos = np.concatenate([[0], np.cumsum(num_nodes)])

    # Prune unused nodes.
    used = np.zeros(pre.coords.shape[0] + 1, dtype=bool)
    used[face_nodes] = True
    node_map = np.zeros(pre.coords.shape[0] + 1, dtype=np.int64)
    node_map[used] = np.arange(used.sum())          # 0-based new ids
    coords = pre.coords[used[1:]]
    face_nodes0 = node_map[face_nodes]

    # Undo axis reversals.
    for ax in range(3):
        if reverse[ax]:
            coords[:, ax] = -coords[:, ax]
    if sum(reverse) % 2 == 1:
        for fidx in range(num_nodes.size):
            s, t = node_pos[fidx], node_pos[fidx + 1]
            face_nodes0[s:t] = face_nodes0[s:t][::-1]

    # cells.faces by (cell, face) sort, half-face tags from cellTags.
    nf = neighbors.shape[0]
    fidx = np.arange(nf)
    both_c = np.concatenate([neighbors[:, 0], neighbors[:, 1]])
    both_f = np.concatenate([fidx, fidx])
    both_t = np.concatenate([cell_tags[:, 1], cell_tags[:, 0]])  # fliplr
    hf = both_c > 0
    order = np.lexsort((both_f[hf], both_c[hf]))
    cf = both_f[hf][order]
    ct = both_t[hf][order] - 1                       # tags 0..5
    cells_of_hf = both_c[hf][order]
    face_pos = np.concatenate(
        [[0], np.cumsum(np.bincount(cells_of_hf - 1,
                                    minlength=keep_cell.sum()))]
    )

    index_map = np.flatnonzero(keep_cell) - nx * ny  # drop aux layer offset

    g = Grid(
        cells=Cells(
            num=int(keep_cell.sum()),
            facePos=face_pos.astype(np.int64),
            faces=cf.astype(np.int64),
            indexMap=index_map.astype(np.int64),
            tags=ct.astype(np.int64),
        ),
        faces=Faces(
            num=nf,
            nodePos=node_pos.astype(np.int64),
            nodes=face_nodes0.astype(np.int64),
            neighbors=(neighbors - 1).astype(np.int64),
            tag=ftag.astype(np.int64),
        ),
        nodes=Nodes(num=coords.shape[0], coords=coords),
        griddim=3,
        type=["processGRDECL"],
        cartDims=(nx, ny, nz),
    )

    if check_grid:
        assert np.all(np.diff(g.cells.facePos) > 3), "cell with < 4 faces"
        assert np.all(np.diff(g.faces.nodePos) > 2), "face with < 3 nodes"
        assert not np.any(np.isinf(g.nodes.coords)), "inf node coordinates"
        g.check()
    return g


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.gridprocessing.geometry import compute_geometry
    from pymrst.gridprocessing.tensor_grid import tensor_grid

    def _regular_grdecl(nx, ny, nz, dx=100.0, dy=80.0, dz=5.0,
                        fault_i: Optional[int] = None, throw: float = 0.0):
        """Simple corner-point description; optional fault plane at i."""
        xp = np.arange(nx + 1) * dx
        yp = np.arange(ny + 1) * dy
        coord = np.zeros(((nx + 1) * (ny + 1), 6))
        pil = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                coord[pil] = [xp[i], yp[j], 0.0, xp[i], yp[j], nz * dz]
                pil += 1
        zc = np.zeros((2 * nx, 2 * ny, 2 * nz))
        for k in range(nz):
            zc[:, :, 2 * k] = k * dz
            zc[:, :, 2 * k + 1] = (k + 1) * dz
        if fault_i is not None:
            zc[2 * fault_i:, :, :] += throw
        return {"cartDims": (nx, ny, nz), "COORD": coord.ravel(),
                "ZCORN": _flat(zc)}

    def test_regular_grid_matches_tensor():
        """Unfaulted corner-point grid equals the tensor grid."""
        gd = _regular_grdecl(4, 3, 2)
        g = compute_geometry(process_grdecl(gd))
        t = compute_geometry(tensor_grid(
            np.arange(5) * 100.0, np.arange(4) * 80.0, np.arange(3) * 5.0
        ))
        assert g.cells.num == t.cells.num == 24
        assert g.faces.num == t.faces.num
        assert np.allclose(np.sort(g.cells.volumes), np.sort(t.cells.volumes))
        assert np.allclose(g.cells.centroids[np.lexsort(g.cells.centroids.T)],
                           t.cells.centroids[np.lexsort(t.cells.centroids.T)])
        assert np.isclose(g.cells.volumes.sum(), 400 * 240 * 10)
        print("ok test_regular_grid_matches_tensor")

    def test_faulted_grid():
        """Half-cell fault throw: volume conserved, fault faces tagged,
        cells close under the divergence theorem."""
        gd = _regular_grdecl(4, 3, 4, fault_i=2, throw=2.5)
        g = compute_geometry(process_grdecl(gd))
        assert g.cells.num == 48
        assert np.isclose(g.cells.volumes.sum(), 400 * 240 * 20)
        assert np.any(g.faces.tag > 0), "no fault faces found"
        # each fault cell pair connects across half a cell height
        cell_no = g.cell_no()
        sgn = np.where(g.faces.neighbors[g.cells.faces, 0] == cell_no,
                       1.0, -1.0)
        net = np.zeros((g.cells.num, 3))
        np.add.at(net, cell_no, sgn[:, None] * g.faces.normals[g.cells.faces])
        assert np.max(np.abs(net)) < 1e-8, "cells do not close"
        internal = np.all(g.faces.neighbors >= 0, axis=1)
        n_fault = int(np.sum(g.faces.tag[internal] > 0))
        assert n_fault == 3 * (2 * 4 - 1), (
            f"expected 21 internal fault subfaces, got {n_fault}"
        )
        print("ok test_faulted_grid")

    def test_actnum_and_pinch():
        """Inactive cells vanish; pinched layers connect through."""
        gd = _regular_grdecl(3, 2, 3)
        act = np.ones((3, 2, 3), dtype=np.int64)
        act[0, 0, 0] = 0
        gd["ACTNUM"] = _flat(act)
        g = process_grdecl(gd)
        assert g.cells.num == 17
        # collapse middle layer to zero thickness everywhere: pinch.
        # Layer 2 becomes [5, 5] and layer 3 grows to [5, 15].
        gd2 = _regular_grdecl(3, 2, 3)
        zc = _F(gd2["ZCORN"], (6, 4, 6)).copy()
        zc[:, :, 3] = zc[:, :, 2]     # layer-2 bottom = its top
        zc[:, :, 4] = zc[:, :, 2]     # layer-3 top moves up to the pinch
        gd2["ZCORN"] = _flat(zc)
        g2 = compute_geometry(process_grdecl(gd2))
        assert g2.cells.num == 12, f"pinched cells remain: {g2.cells.num}"
        # layers 1 and 3 must still be connected vertically
        internal = np.all(g2.faces.neighbors >= 0, axis=1)
        z1 = g2.cells.centroids[g2.faces.neighbors[internal, 0], 2]
        z2 = g2.cells.centroids[g2.faces.neighbors[internal, 1], 2]
        assert np.any(np.abs(z1 - z2) > 1e-9), "no vertical connections"
        print("ok test_actnum_and_pinch")

    def test_spe1_grdecl():
        """The real SPE1 GRDECL file processes to the known grid."""
        path = Path(__file__).parents[3] / (
            "mrst-2024a-Geomech/examples/data/SPE1/SPE1.GRDECL"
        )
        from pymrst.modules.deckformat.grdecl import read_grdecl
        gd = read_grdecl(path)
        g = compute_geometry(process_grdecl(gd))
        assert g.cells.num == 300 and g.cartDims == (10, 10, 3)
        assert np.isclose(g.cells.volumes.sum(),
                          10000.0 * 10000.0 * 100.0)   # ft^3
        vols = g.cells.volumes[np.argsort(g.cells.indexMap)]
        assert np.allclose(vols[:100], 1000 * 1000 * 20.0)
        assert np.allclose(vols[200:], 1000 * 1000 * 50.0)
        print("ok test_spe1_grdecl")

    test_regular_grid_matches_tensor()
    test_faulted_grid()
    test_actnum_and_pinch()
    test_spe1_grdecl()
    print("\nAll tests passed!")
