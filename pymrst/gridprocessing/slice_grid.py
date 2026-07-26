"""Cut a 3D grid along a plane, splitting the cells it crosses (MRST `sliceGrid`).

Where the plane passes through a cell, that cell is divided into two; where it
crosses a face, that face is split into two; and a new "cut face" - the polygon
of the cross-section - is inserted between the two halves so they remain
neighbours only across it. The result is a valid grid with the plane represented
as internal faces, which is what a fault, a well trajectory, or a reporting
cross-section needs.

This is a clean-cut implementation using a global edge-crossing (marching)
formulation: it handles a planar cut that passes through cell *interiors* (no
node lies exactly on the plane, each crossed face is crossed on exactly two
edges). MRST's `sliceGrid` additionally copes with cuts that snap to existing
nodes, partial cuts, and the resulting degenerate/duplicate faces via its
`topologicalCellSplit`; those cases are out of this port's scope and are flagged.
The geometric core - which cells are cut and where - is
`compute_grid_slice_polygons`.
"""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def _plane(g, coord, normal, cut_dir):
    """Return `(n, d, window)`: the plane `n·x = d`, and `window = (seg0, vu, L)`
    bounding the cut to a finite segment, or `None` for an infinite plane."""
    coord = np.atleast_2d(np.asarray(coord, dtype=float))
    if coord.shape[0] == 1:
        assert normal is not None, "a single point needs a plane normal"
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        return n, float(coord[0].dot(n)), None
    from scipy.linalg import null_space
    seg = coord[:2]
    v = seg[1] - seg[0]
    L = np.linalg.norm(v)
    vu = v / L
    cd = np.asarray(cut_dir, dtype=float)
    cd /= np.linalg.norm(cd)
    n = null_space(np.vstack([vu, cd]))[:, 0]
    return n, float(seg[0].dot(n)), (seg[0], vu, L)


def _rebuild_cell_faces(num_cells, neighbors):
    """cells.faces / facePos from the face-neighbour incidence."""
    cf, cell = [], []
    for f, (a, b) in enumerate(neighbors):
        for c in (a, b):
            if c >= 0:
                cf.append(f)
                cell.append(c)
    cell = np.asarray(cell, dtype=np.int64)
    order = np.argsort(cell, kind="stable")
    faces = np.asarray(cf, dtype=np.int64)[order]
    counts = np.bincount(cell[order], minlength=num_cells)
    face_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return faces, face_pos


def slice_grid(g, coord, normal: Optional[np.ndarray] = None,
               cut_dir: Optional[np.ndarray] = None, tol: float = 1e-9):
    """Return the grid cut by the plane through `coord` (point+`normal` or
    segment+`cut_dir`), with the crossed cells split in two."""
    coords = np.asarray(g.nodes.coords, dtype=float)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    nc0 = g.cells.num

    n, d, window = _plane(g, coord, normal, cut_dir)
    sd = coords.dot(n) - d
    side = np.zeros(sd.size, dtype=np.int64)
    side[sd > tol] = 1
    side[sd < -tol] = -1                                 # 0 == on the plane

    # crossing node on every edge whose endpoints strictly straddle the plane
    cross = {}
    new_coords = []

    def crossing(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cross:
            s = sd[key[0]] / (sd[key[0]] - sd[key[1]])
            cross[key] = coords.shape[0] + len(new_coords)
            new_coords.append(coords[key[0]] + s * (coords[key[1]] - coords[key[0]]))
        return cross[key]

    def boundary_t(point):
        """Position of a plane point along the cut segment (window coordinate)."""
        seg0, vu, _ = window
        return float((np.asarray(point) - seg0).dot(vu))

    # a cell is cut when it has nodes strictly on both sides of the plane
    cut_cell = np.zeros(nc0, dtype=bool)
    for f in range(g.faces.num):
        seg = fnodes[node_pos[f]:node_pos[f + 1]]
        s = side[seg]
        if (s > 0).any() and (s < 0).any():
            for c in neigh[f]:
                if c >= 0:
                    cut_cell[c] = True

    # with a finite cutting segment, keep only cells the segment fully traverses;
    # partially-reached cells are left whole (MRST's removePartialCuts)
    fully_cut = cut_cell.copy()
    if window is not None:
        _, _, L = window
        tspan = {int(c): [np.inf, -np.inf] for c in np.flatnonzero(cut_cell)}
        for f in range(g.faces.num):
            seg = fnodes[node_pos[f]:node_pos[f + 1]]
            s = side[seg]
            m = seg.size
            bt = []
            for k in range(m):
                u, w = seg[k], seg[(k + 1) % m]
                if side[u] == 0:
                    bt.append(boundary_t(coords[u]))
                if side[u] * side[w] < 0:
                    ss = sd[u] / (sd[u] - sd[w])
                    bt.append(boundary_t(coords[u] + ss * (coords[w] - coords[u])))
            if bt:
                for c in neigh[f]:
                    if c >= 0 and cut_cell[c]:
                        lo, hi = tspan[int(c)]
                        tspan[int(c)] = [min(lo, min(bt)), max(hi, max(bt))]
        wtol = 1e-9 * max(L, 1.0)
        for c in np.flatnonzero(cut_cell):
            lo, hi = tspan[int(c)]
            if lo < -wtol or hi > L + wtol:
                fully_cut[c] = False

    cut_ids = np.flatnonzero(fully_cut)
    plus_id = np.full(nc0, -1, dtype=np.int64)
    plus_id[cut_ids] = nc0 + np.arange(cut_ids.size)     # + half of each cut cell

    def half(c, side_pos):
        if c < 0:
            return -1
        return plus_id[c] if (side_pos and fully_cut[c]) else c

    out_nodes, out_np, out_neigh = [], [0], []
    # per cut cell, the boundary edges (pairs of on-plane / crossing points) that
    # bound its cut polygon
    cell_cut_edges = {int(c): [] for c in cut_ids}

    def record_edge(a, b, bnd):
        if len(bnd) == 2 and bnd[0] != bnd[1]:
            for c in (a, b):
                if c >= 0 and fully_cut[c]:
                    cell_cut_edges[int(c)].append((bnd[0], bnd[1]))

    for f in range(g.faces.num):
        seg = fnodes[node_pos[f]:node_pos[f + 1]]
        s = side[seg]
        a, b = neigh[f]
        has_plus, has_minus = (s > 0).any(), (s < 0).any()
        splitting = ((a >= 0 and fully_cut[a]) or (b >= 0 and fully_cut[b]))

        if has_plus and has_minus and splitting:         # face is truly split
            m = seg.size
            plus, minus, bnd = [], [], []
            for k in range(m):
                u, w = seg[k], seg[(k + 1) % m]
                if side[u] >= 0:
                    plus.append(u)
                if side[u] <= 0:
                    minus.append(u)
                if side[u] == 0:
                    bnd.append(u)
                if side[u] * side[w] < 0:                 # strict crossing
                    x = crossing(u, w)
                    plus.append(x)
                    minus.append(x)
                    bnd.append(x)
            record_edge(a, b, bnd)
            out_nodes.extend(minus)
            out_np.append(out_np[-1] + len(minus))
            out_neigh.append([half(a, False), half(b, False)])
            out_nodes.extend(plus)
            out_np.append(out_np[-1] + len(plus))
            out_neigh.append([half(a, True), half(b, True)])
        else:                                            # face on one side
            # all-on-plane faces (the plane coincides with an existing face) are
            # kept as-is; only a *cut* cell with such a face would be the
            # zero-volume degenerate MRST warns about
            if not has_plus and not has_minus and (fully_cut[neigh[f]] & (neigh[f] >= 0)).any():
                raise NotImplementedError(
                    "the plane coincides with a face of a cut cell (zero volume)")
            side_pos = has_plus                          # False also for all-0
            out_nodes.extend(seg.tolist())
            out_np.append(out_np[-1] + seg.size)
            out_neigh.append([half(a, side_pos), half(b, side_pos)])
            # a face touching the plane along an edge (two on-plane nodes)
            # still bounds the cut polygon of its cut cells
            record_edge(a, b, [int(x) for x in seg[s == 0]])

    all_coords = np.vstack([coords, np.array(new_coords)]) if new_coords else coords

    # one cut-polygon face per cut cell, between its - and + halves; wind it so
    # its normal points from the minus half to the plus half (along +n)
    for c in cut_ids:
        ring = _order_ring(cell_cut_edges[int(c)])
        if np.dot(_polygon_normal(all_coords[ring]), n) < 0:
            ring = ring[::-1]
        out_nodes.extend(ring)
        out_np.append(out_np[-1] + len(ring))
        out_neigh.append([int(c), int(plus_id[c])])
    neighbors = np.asarray(out_neigh, dtype=np.int64)
    faces_nodes = np.asarray(out_nodes, dtype=np.int64)
    faces_pos = np.asarray(out_np, dtype=np.int64)
    num_cells = nc0 + cut_ids.size
    cell_faces, cell_face_pos = _rebuild_cell_faces(num_cells, neighbors)

    return Grid(
        cells=Cells(num=int(num_cells), facePos=cell_face_pos, faces=cell_faces,
                    indexMap=np.arange(num_cells, dtype=np.int64)),
        faces=Faces(num=int(faces_pos.size - 1), nodePos=faces_pos,
                    nodes=faces_nodes, neighbors=neighbors),
        nodes=Nodes(num=int(all_coords.shape[0]), coords=all_coords),
        griddim=3, type=["sliceGrid"])


def _polygon_normal(pts: np.ndarray) -> np.ndarray:
    """Newell's method: an area-weighted normal of a (planar) polygon."""
    p = np.asarray(pts, dtype=float)
    q = np.roll(p, -1, axis=0)
    nx = np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2]))
    ny = np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0]))
    nz = np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))
    return np.array([nx, ny, nz])


def _order_ring(edges):
    """Order the crossing-node pairs (one per cut face) into a single ring."""
    from collections import defaultdict
    seen = set()
    uniq = []
    for u, w in edges:                                   # drop repeated edges
        key = (u, w) if u < w else (w, u)
        if key not in seen:
            seen.add(key)
            uniq.append((u, w))
    edges = uniq
    adj = defaultdict(list)
    for u, w in edges:
        adj[u].append(w)
        adj[w].append(u)
    start = edges[0][0]
    ring = [start]
    prev = None
    cur = start
    while True:
        nxt = [x for x in adj[cur] if x != prev]
        if not nxt:
            break
        step = nxt[0]
        if step == start:
            break
        ring.append(step)
        prev, cur = cur, step
        if len(ring) > 2 * len(edges):
            break
    return ring


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_single_cube_splits_in_two():
        """A cube cut by x=0.5 becomes two cells of equal volume, joined by one
        new internal face."""
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        s = compute_geometry(slice_grid(
            g, np.array([[0.5, -1.0, 0.0], [0.5, 2.0, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0])))
        assert s.cells.num == 2, s.cells.num
        assert np.allclose(s.cells.volumes, 0.5), s.cells.volumes
        print(f"ok test_single_cube_splits_in_two "
              f"({s.cells.num} cells, vols {s.cells.volumes})")

    def test_total_volume_is_conserved():
        g = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
        s = compute_geometry(slice_grid(
            g, np.array([[1.5, -1.0, 0.0], [1.5, 3.0, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0])))
        assert np.isclose(s.cells.volumes.sum(), g.cells.volumes.sum())
        # the x=1.5 plane cuts the 4 cells of the middle column (j=0..1,k=0..1)
        assert s.cells.num == g.cells.num + 4, s.cells.num
        print(f"ok test_total_volume_is_conserved "
              f"({g.cells.num} -> {s.cells.num} cells)")

    def test_oblique_plane_conserves_volume():
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        # plane x + 0.3y = 1.1, chosen to miss every grid node
        s = compute_geometry(slice_grid(
            g, np.array([1.1, 0.0, 0.5]), normal=np.array([1.0, 0.3, 0.0])))
        assert np.isclose(s.cells.volumes.sum(), g.cells.volumes.sum())
        assert np.all(s.cells.volumes > 0)
        print(f"ok test_oblique_plane_conserves_volume "
              f"({s.cells.num} cells, total vol {s.cells.volumes.sum():.3f})")

    def test_cut_snapped_to_grid_nodes():
        """A plane x=1 falls exactly on the node column between the two cells of
        a 2x1x1 grid - a snapped cut. No cell straddles it, so nothing is split
        and the grid is unchanged (each cell only touches the plane)."""
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        s = compute_geometry(slice_grid(
            g, np.array([[1.0, -1.0, 0.0], [1.0, 2.0, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0])))
        assert s.cells.num == 2, s.cells.num          # nothing straddles x=1
        assert np.isclose(s.cells.volumes.sum(), g.cells.volumes.sum())
        print("ok test_cut_snapped_to_grid_nodes")

    def test_diagonal_cut_through_cell_corners():
        """The plane x=y passes through the corner nodes of a 2x2x1 grid, cutting
        the two off-diagonal cells while snapping to shared corners."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        s = compute_geometry(slice_grid(
            g, np.array([[-1.0, -1.0, 0.0], [3.0, 3.0, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0])))
        assert np.isclose(s.cells.volumes.sum(), g.cells.volumes.sum()), \
            s.cells.volumes.sum()
        assert np.all(s.cells.volumes > 1e-12), s.cells.volumes
        print(f"ok test_diagonal_cut_through_cell_corners "
              f"({g.cells.num} -> {s.cells.num} cells, vol {s.cells.volumes.sum():.3f})")

    def test_partial_cut_leaves_unreached_cell_whole():
        """A segment that stops at x=2.5 cuts cells 0 and 1 of a 3x1x1 grid but
        not cell 2 (the segment ends inside it) - MRST's removePartialCuts."""
        g = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        s = compute_geometry(slice_grid(
            g, np.array([[0.0, 0.5, 0.0], [2.5, 0.5, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0])))
        assert s.cells.num == 5, s.cells.num          # cells 0,1 cut; cell 2 whole
        assert np.isclose(s.cells.volumes.sum(), g.cells.volumes.sum())
        assert np.any(np.isclose(s.cells.volumes, 1.0)), "cell 2 stays a unit cell"
        print(f"ok test_partial_cut_leaves_unreached_cell_whole "
              f"({g.cells.num} -> {s.cells.num} cells)")

    test_single_cube_splits_in_two()
    test_total_volume_is_conserved()
    test_oblique_plane_conserves_volume()
    test_cut_snapped_to_grid_nodes()
    test_diagonal_cut_through_cell_corners()
    test_partial_cut_leaves_unreached_cell_whole()
    print("\nAll tests passed!")
