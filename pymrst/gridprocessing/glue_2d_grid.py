"""Glue two 2D grids along a shared boundary (MRST `glue2DGrid`).

Stitches two 2D grids into one where their boundaries meet. The two grids are
concatenated, their boundaries are traced as oriented loops, nodes that coincide
within a tolerance are merged, and the boundary faces that now lie between a cell
of each grid are welded so those cells become neighbours. Where one grid's
boundary node falls partway along the other's boundary edge, that edge is first
split so the boundaries conform. This is how two independently-meshed regions -
two blocks of a reservoir, a refined patch against a coarse background - are
joined into a single simulation grid.

Faces here have exactly two nodes (2D). Boundary is the `-1` neighbour, as
elsewhere in this port (MRST uses `0`).
"""

from types import SimpleNamespace
from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def _face_node_pairs(g) -> np.ndarray:
    """The `(nf, 2)` node pairs of a 2D grid's faces."""
    return np.asarray(g.faces.nodes, dtype=np.int64).reshape(-1, 2)


def _boundary_faces(g) -> np.ndarray:
    return np.flatnonzero(np.any(np.asarray(g.faces.neighbors) == -1, axis=1))


def _loop_orientation(x: np.ndarray, y: np.ndarray) -> int:
    """+1 clockwise, -1 counter-clockwise (MRST `loop_orientation`)."""
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    i = np.arange(x.size - 2)
    area = np.sum(x[i + 1] * (y[i + 2] - y[i]) + y[i + 1] * (x[i] - x[i + 2])) / 2
    return int(np.sign(area))


def _boundary_loop(g) -> SimpleNamespace:
    """Boundary nodes and faces in clockwise order (MRST `boundary_loop`)."""
    bfaces = _boundary_faces(g)
    pairs = _face_node_pairs(g)
    assert np.all(np.diff(np.asarray(g.faces.nodePos)) == 2), "2D faces only"

    bnodes = pairs[bfaces].astype(float)          # working copy, NaN as we consume
    n_idx = np.zeros(bfaces.size, dtype=np.int64)
    f_idx = np.zeros(bfaces.size, dtype=np.int64)

    f_idx[0] = bfaces[0]
    n_idx[0] = int(bnodes[0, 0])
    bfaces = bfaces.astype(float)
    bfaces[0] = np.nan

    count = 0
    fix = 0
    while True:
        side = 1 if n_idx[count] == bnodes[fix, 0] else 0
        count += 1
        n_idx[count] = int(bnodes[fix, side])
        bnodes[fix, :] = np.nan

        hit = np.flatnonzero(np.any(bnodes == n_idx[count], axis=1))
        if hit.size == 0:
            break
        fix = int(hit[0])
        f_idx[count] = int(bfaces[fix])
        bfaces[fix] = np.nan
        if np.all(np.isnan(bfaces)):
            break

    if not np.all(np.isnan(bfaces)):
        raise NotImplementedError("internal boundaries are not supported")

    xy = np.asarray(g.nodes.coords, dtype=float)[n_idx]
    if _loop_orientation(xy[:, 0], xy[:, 1]) < 0:      # make clockwise
        n_idx = np.flipud(n_idx)
        f_idx = np.roll(np.flipud(f_idx), -1)
    return SimpleNamespace(nodes=n_idx, faces=f_idx)


def _coinciding_nodes(g1, n1, g2, n2, tol) -> np.ndarray:
    """Pairs `[node_in_g1, node_in_g2]` overlapping within `tol` (MRST
    `coinciding_nodes`)."""
    p1 = np.asarray(g1.nodes.coords, dtype=float)[n1]
    p2 = np.asarray(g2.nodes.coords, dtype=float)[n2]
    d = np.sqrt(((p2[:, None, :2] - p1[None, :, :2]) ** 2).sum(axis=2))
    i, j = np.where(d < tol)                            # i in n2, j in n1
    return np.column_stack([np.asarray(n1)[j], np.asarray(n2)[i]])


def _concatenate(g1, g2) -> SimpleNamespace:
    """Concatenate two grids without deduplication (MRST `concatenate_grids`)."""
    n1n = g1.nodes.num
    n1f = g1.faces.num
    n1c = g1.cells.num

    coords = np.vstack([np.asarray(g1.nodes.coords, float),
                        np.asarray(g2.nodes.coords, float)])

    neigh2 = np.asarray(g2.faces.neighbors, dtype=np.int64).copy()
    neigh2[neigh2 != -1] += n1c
    neighbors = np.vstack([np.asarray(g1.faces.neighbors, dtype=np.int64), neigh2])

    fnodes = np.concatenate([np.asarray(g1.faces.nodes, dtype=np.int64),
                             np.asarray(g2.faces.nodes, dtype=np.int64) + n1n])

    cfaces = np.concatenate([np.asarray(g1.cells.faces, dtype=np.int64),
                             np.asarray(g2.cells.faces, dtype=np.int64) + n1f])
    fp1 = np.asarray(g1.cells.facePos, dtype=np.int64)
    fp2 = np.asarray(g2.cells.facePos, dtype=np.int64)
    face_pos = np.concatenate([fp1, fp2[1:] + fp1[-1]])

    nf = n1f + g2.faces.num
    return SimpleNamespace(
        nodes=SimpleNamespace(num=coords.shape[0], coords=coords),
        faces=SimpleNamespace(num=nf, neighbors=neighbors, nodes=fnodes,
                              nodePos=np.arange(0, 2 * nf + 1, 2, dtype=np.int64)),
        cells=SimpleNamespace(num=n1c + g2.cells.num, faces=cfaces,
                              facePos=face_pos))


def _identify_mergefaces(loop1, loop2, cnodes, offset) -> np.ndarray:
    """Boundary face pairs to merge: faces whose two endpoints both coincide
    across the grids (MRST `identify_mergefaces`)."""
    fn1 = np.column_stack([loop1.nodes, np.roll(loop1.nodes, -1)])
    fn2 = np.column_stack([loop2.nodes, np.roll(loop2.nodes, -1)])

    t1 = fn1.astype(float).copy()
    t2 = fn2.astype(float).copy()
    for a, b in cnodes:
        t1[t1 == a] = 0.0
        t2[t2 == b] = 0.0
    # a loop edge is a merge candidate when *both* its endpoints coincided
    f1 = np.flatnonzero(np.nansum(t1, axis=1) == 0)

    merges = []
    for e in f1:
        nodes1 = fn1[e]
        # its counterpart in loop2: the g2-nodes matching (nodes1[1], nodes1[0])
        m2a = cnodes[cnodes[:, 0] == nodes1[1], 1]
        m2b = cnodes[cnodes[:, 0] == nodes1[0], 1]
        if m2a.size == 0 or m2b.size == 0:
            continue
        target = np.array([m2a[0], m2b[0]])
        f2 = np.flatnonzero(np.all(fn2 == target, axis=1))
        if f2.size != 1:
            continue
        merges.append([loop1.faces[e], loop2.faces[f2[0]] + offset])
    return np.array(merges, dtype=np.int64) if merges else np.zeros((0, 2), np.int64)


def _cleanup_unused_faces(g) -> None:
    """Drop faces no cell references and renumber (MRST `cleanup_unused_faces`)."""
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    used = np.unique(cell_faces)
    keep = np.zeros(g.faces.num, dtype=bool)
    keep[used] = True

    new_id = -np.ones(g.faces.num, dtype=np.int64)
    new_id[keep] = np.arange(keep.sum(), dtype=np.int64)

    pairs = _face_node_pairs_ns(g)[keep]
    g.faces.neighbors = np.asarray(g.faces.neighbors)[keep]
    g.faces.nodes = pairs.ravel()
    g.faces.num = int(keep.sum())
    g.faces.nodePos = np.arange(0, 2 * g.faces.num + 1, 2, dtype=np.int64)
    g.cells.faces = new_id[cell_faces]


def _face_node_pairs_ns(g) -> np.ndarray:
    return np.asarray(g.faces.nodes, dtype=np.int64).reshape(-1, 2)


def _merge_all_faces(g, mfaces) -> None:
    """Give the second face of each pair both cells; the first is left unused
    (MRST `merge_all_faces`)."""
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    # the (single) real cell behind each first face
    cells = neigh[mfaces[:, 0]].max(axis=1)            # the non -1 entry
    tgt = neigh[mfaces[:, 1]].copy()
    for row, c in zip(range(mfaces.shape[0]), cells):
        slot = np.flatnonzero(tgt[row] == -1)[0]
        tgt[row, slot] = c
    neigh[mfaces[:, 1]] = tgt
    g.faces.neighbors = neigh

    reindex = np.arange(np.asarray(g.cells.faces).max() + 1, dtype=np.int64)
    reindex[mfaces[:, 0]] = mfaces[:, 1]
    g.cells.faces = reindex[np.asarray(g.cells.faces, dtype=np.int64)]


def _ensure_conformal(target, other, tol):
    """Split target boundary edges where other's boundary nodes fall on them,
    so boundary contact is node-to-node only (MRST `ensure_conformal_boundaries`).

    For the common case of already-matching boundaries this is a no-op.
    """
    bf = _boundary_faces(target)
    pairs = _face_node_pairs(target)
    bnodes_t = np.unique(pairs[bf])
    bnodes_o = np.unique(_face_node_pairs(other)[_boundary_faces(other)])

    cn = _coinciding_nodes(other, bnodes_o, target, bnodes_t, tol)
    free = np.setdiff1d(bnodes_o, cn[:, 0]) if cn.size else bnodes_o
    if free.size == 0:
        return target

    oc = np.asarray(other.nodes.coords, float)[free]
    tc = np.asarray(target.nodes.coords, float)
    inserts: dict = {}
    for p in oc:
        for f in bf:
            a, b = tc[pairs[f, 0], :2], tc[pairs[f, 1], :2]
            fd = b - a
            L = np.hypot(*fd)
            if L == 0:
                continue
            u = fd / L
            delta = u[1] * (p[0] - a[0]) - (p[1] - a[1]) * u[0]     # normal dist
            if abs(delta) >= tol:
                continue
            if np.dot(p[:2] - a, u) > 0 and np.dot(p[:2] - b, -u) > 0:
                inserts.setdefault(int(f), []).append(p)
                break
    if not inserts:
        return target

    return _split_boundary_edges(target, inserts)


def _split_boundary_edges(target, inserts):
    """Insert the given points into the named boundary edges, splitting each."""
    coords = list(np.asarray(target.nodes.coords, float))
    pairs = _face_node_pairs(target).tolist()
    neigh = np.asarray(target.faces.neighbors, dtype=np.int64).tolist()
    cell_faces = np.asarray(target.cells.faces, dtype=np.int64)
    face_pos = np.asarray(target.cells.facePos, dtype=np.int64)

    # replace each split face with a chain of small faces through the new nodes
    new_faces_for = {}
    for f, pts in inserts.items():
        a = np.asarray(coords[pairs[f][0]])[:2]
        pts = sorted(pts, key=lambda q: np.hypot(*(np.asarray(q)[:2] - a)))
        chain = [pairs[f][0]]
        for q in pts:
            coords.append(np.array([q[0], q[1], 0.0])[:np.asarray(coords[0]).size])
            chain.append(len(coords) - 1)
        chain.append(pairs[f][1])
        seg_faces = []
        for k in range(len(chain) - 1):
            pairs.append([chain[k], chain[k + 1]])
            neigh.append(neigh[f])
            seg_faces.append(len(pairs) - 1)
        new_faces_for[f] = seg_faces

    # rebuild cells.faces, expanding split faces into their chains
    new_cf = []
    new_fp = [0]
    for c in range(target.cells.num):
        block = list(cell_faces[face_pos[c]:face_pos[c + 1]])
        expanded = []
        for fc in block:
            expanded += new_faces_for.get(int(fc), [int(fc)])
        new_cf += expanded
        new_fp.append(len(new_cf))

    g = SimpleNamespace(
        nodes=SimpleNamespace(num=len(coords), coords=np.array(coords)),
        faces=SimpleNamespace(num=len(pairs),
                              neighbors=np.array(neigh, dtype=np.int64),
                              nodes=np.array(pairs, dtype=np.int64).ravel(),
                              nodePos=np.arange(0, 2 * len(pairs) + 1, 2)),
        cells=SimpleNamespace(num=target.cells.num,
                              faces=np.array(new_cf, dtype=np.int64),
                              facePos=np.array(new_fp, dtype=np.int64)))
    _cleanup_unused_faces(g)
    return g


def glue_2d_grid(g1, g2, tol: float = 1e-3, *, return_mapping=False):
    """Return the single 2D grid formed by gluing `g1` and `g2` along their
    shared boundary."""
    g1c = _ensure_conformal(g1, g2, tol)
    g2c = _ensure_conformal(g2, g1, tol)

    g = _concatenate(g1c, g2c)
    loop1 = _boundary_loop(g1c)
    loop2 = _boundary_loop(g2c)
    cnodes = _coinciding_nodes(g1c, loop1.nodes, g2c, loop2.nodes, tol)

    if cnodes.size:
        keep = cnodes[:, 0]
        discard = cnodes[:, 1] + g1c.nodes.num
        subst = np.arange(g.nodes.num, dtype=np.int64)
        subst[discard] = keep
        g.faces.nodes = subst[np.asarray(g.faces.nodes, dtype=np.int64)]

        remaining = np.setdiff1d(np.arange(g.nodes.num), discard)
        renum = -np.ones(g.nodes.num, dtype=np.int64)
        renum[remaining] = np.arange(remaining.size, dtype=np.int64)
        g.faces.nodes = renum[g.faces.nodes]
        g.nodes.coords = np.asarray(g.nodes.coords)[remaining]
        g.nodes.num = remaining.size

    mfaces = _identify_mergefaces(loop1, loop2, cnodes, g1c.faces.num)
    if mfaces.shape[0]:
        _merge_all_faces(g, mfaces)
    _cleanup_unused_faces(g)

    result = Grid(
        cells=Cells(num=int(g.cells.num),
                    facePos=np.asarray(g.cells.facePos, dtype=np.int64),
                    faces=np.asarray(g.cells.faces, dtype=np.int64),
                    indexMap=np.arange(g.cells.num, dtype=np.int64)),
        faces=Faces(num=int(g.faces.num),
                    nodePos=np.asarray(g.faces.nodePos, dtype=np.int64),
                    nodes=np.asarray(g.faces.nodes, dtype=np.int64),
                    neighbors=np.asarray(g.faces.neighbors, dtype=np.int64)),
        nodes=Nodes(num=int(g.nodes.num), coords=np.asarray(g.nodes.coords)),
        griddim=2, type=["glue2DGrid"])
    if return_mapping:
        return result, _glue_mapping(result, g1, g2, tol)
    return result


def _glue_mapping(result, g1, g2, tol):
    """Map glued entities back to both inputs; ``-1`` means no parent."""
    node_parent = np.full((result.nodes.num, 2), -1, dtype=np.int64)
    for column, source in enumerate((g1, g2)):
        delta = (
            result.nodes.coords[:, None, :2]
            - np.asarray(source.nodes.coords)[None, :, :2]
        )
        distance = np.linalg.norm(delta, axis=2)
        nearest = np.argmin(distance, axis=1)
        hit = distance[np.arange(result.nodes.num), nearest] < tol
        node_parent[hit, column] = nearest[hit]

    face_parent = np.full((result.faces.num, 2), -1, dtype=np.int64)
    result_pairs = result.faces.nodes.reshape(-1, 2)
    result_keys = [
        frozenset(map(tuple, result.nodes.coords[pair, :2]))
        for pair in result_pairs
    ]
    for column, source in enumerate((g1, g2)):
        source_pairs = np.asarray(source.faces.nodes).reshape(-1, 2)
        source_coords = np.asarray(source.nodes.coords)
        lookup = {
            frozenset(map(tuple, source_coords[pair, :2])): face
            for face, pair in enumerate(source_pairs)
        }
        for face, key in enumerate(result_keys):
            face_parent[face, column] = lookup.get(key, -1)

    cell_parent = np.full((result.cells.num, 2), -1, dtype=np.int64)
    cell_parent[:g1.cells.num, 0] = np.arange(g1.cells.num)
    cell_parent[g1.cells.num:, 1] = np.arange(g2.cells.num)
    return SimpleNamespace(
        cells=cell_parent,
        faces=face_parent,
        nodes=node_parent,
    )


if __name__ == "__main__":
    from pymrst.gridprocessing.tessellation_grid import tessellation_grid

    def _square(x0):
        """A single unit-quad cell at x in [x0, x0+1], y in [0, 1]."""
        p = np.array([[x0, 0.0], [x0 + 1, 0.0], [x0 + 1, 1.0], [x0, 1.0]])
        return tessellation_grid(p, [[0, 1, 2, 3]])

    def test_two_squares_glue_into_a_2x1_strip():
        """Two unit squares sharing the x=1 edge glue into a 2-cell grid with
        one internal face joining them."""
        g1 = _square(0.0)
        g2 = _square(1.0)
        g = glue_2d_grid(g1, g2)
        assert g.cells.num == 2, g.cells.num
        neigh = np.asarray(g.faces.neighbors)
        internal = np.all(neigh >= 0, axis=1)
        assert internal.sum() == 1, internal.sum()
        assert set(neigh[internal][0]) == {0, 1}
        print(f"ok test_two_squares_glue_into_a_2x1_strip "
              f"({g.cells.num} cells, {g.faces.num} faces)")

    def test_glue_merges_duplicate_boundary_nodes():
        """The shared edge's two coincident node pairs collapse: 6 nodes, not 8."""
        g = glue_2d_grid(_square(0.0), _square(1.0))
        assert g.nodes.num == 6, g.nodes.num
        print(f"ok test_glue_merges_duplicate_boundary_nodes ({g.nodes.num} nodes)")

    def test_glued_grid_face_count():
        """Two quads glued: 7 faces (4 + 4 - 1 shared)."""
        g = glue_2d_grid(_square(0.0), _square(1.0))
        assert g.faces.num == 7, g.faces.num
        print(f"ok test_glued_grid_face_count ({g.faces.num} faces)")

    test_two_squares_glue_into_a_2x1_strip()
    test_glue_merges_duplicate_boundary_nodes()
    test_glued_grid_face_count()
    print("\nAll tests passed!")
