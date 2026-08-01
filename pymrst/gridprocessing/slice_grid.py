"""Cut a 3D grid along a plane, splitting the cells it crosses (MRST `sliceGrid`).

Where the plane passes through a cell, that cell is divided into two; where it
crosses a face, that face is split into two; and a new "cut face" - the polygon
of the cross-section - is inserted between the two halves so they remain
neighbours only across it. The result is a valid grid with the plane represented
as internal faces, which is what a fault, a well trajectory, or a reporting
cross-section needs.

This implementation uses a global edge-crossing (marching) formulation. It
handles clean planar cuts, cuts snapped to existing nodes, finite partial cuts,
repeated cuts, MRST's face-only mode, and cells whose collapsed or in-plane
faces make the cut degenerate. The geometric core - which cells are cut and
where - is `compute_grid_slice_polygons`.
"""

import warnings
from types import SimpleNamespace
from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes

# the 27 buckets around a point, so a coincidence test never misses a match
# that quantisation pushed into an adjacent bucket; the point's own bucket is
# first because it is nearly always the one that matches
_BUCKET_NEIGHBOURS = np.array(
    [[0, 0, 0]] + [[i, j, k]
                   for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                   if (i, j, k) != (0, 0, 0)],
    dtype=np.int64,
)


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
    neighbors = np.asarray(neighbors, dtype=np.int64)
    # row-major so each face contributes its left neighbour then its right,
    # which is the order the equivalent Python loop produced
    cell = neighbors.ravel()
    face = np.repeat(np.arange(neighbors.shape[0], dtype=np.int64), 2)
    attached = cell >= 0
    cell, face = cell[attached], face[attached]
    order = np.argsort(cell, kind="stable")
    faces = face[order]
    counts = np.bincount(cell[order], minlength=num_cells)
    face_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return faces, face_pos


def slice_grid(g, coord, normal: Optional[np.ndarray] = None,
               cut_dir: Optional[np.ndarray] = None, tol: float = 1e-9,
               *, only_faces: bool = False, return_mapping: bool = False,
               return_slice: bool = False, gix_input=None):
    """Return the grid cut by the plane through `coord` (point+`normal` or
    segment+`cut_dir`), with the crossed cells split in two.

    A list of coordinates performs multiple cuts in sequence. With
    ``return_mapping=True``, return ``(grid, gix)`` where ``gix`` contains
    MRST-compatible ``old``, ``new`` and ``parent`` entity arrays. With
    ``return_slice=True``, also return the lower-dimensional grid formed by
    faces on the slice, matching MRST's third output. The result is
    ``(grid, slice_grid)`` or ``(grid, gix, slice_grid)`` when both outputs
    are requested.

    ``only_faces=True`` matches MRST's ``onlyFaces`` option: intersected faces
    are split, but cells are not split and no new slice faces are inserted.
    ``gix_input`` accepts a mapping returned by an earlier call and composes
    the new result back to that mapping's original grid, matching MRST's
    staged-slicing option of the same name.
    """
    if isinstance(coord, (list, tuple)) and coord and not np.isscalar(coord[0]):
        coordinates = list(coord)
        normals = normal if isinstance(normal, (list, tuple)) else [normal] * len(coordinates)
        directions = cut_dir if isinstance(cut_dir, (list, tuple)) else [cut_dir] * len(coordinates)
        mapping = gix_input
        for point, nrm, direction in zip(coordinates, normals, directions):
            g, current = _slice_grid_single(
                g, point, normal=nrm, cut_dir=direction, tol=tol,
                only_faces=only_faces,
            )
            mapping = _compose_mappings(mapping, current)
        return _outputs(g, mapping, return_mapping, return_slice)
    g, mapping = _slice_grid_single(
        g, coord, normal=normal, cut_dir=cut_dir, tol=tol,
        only_faces=only_faces,
    )
    mapping = _compose_mappings(gix_input, mapping)
    return _outputs(g, mapping, return_mapping, return_slice)


def _outputs(g, mapping, return_mapping, return_slice):
    """Assemble optional Python equivalents of MRST's second/third outputs."""
    if return_slice:
        from pymrst.gridprocessing.get_2d_grid_from_faces import (
            get_2d_grid_from_faces,
        )

        slice_faces = np.asarray(mapping.new.faces) == 3
        lower = get_2d_grid_from_faces(g, slice_faces) if slice_faces.any() else None
        if return_mapping:
            return g, mapping, lower
        return g, lower
    return (g, mapping) if return_mapping else g


def _slice_grid_single(g, coord, normal=None, cut_dir=None, tol=1e-9,
                       only_faces=False):
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

    # which side(s) each face reaches, for every face at once - the passes
    # below consult this per face, and doing it in Python dominated the cost
    face_of_node = np.repeat(np.arange(g.faces.num, dtype=np.int64),
                             np.diff(node_pos))
    node_side = side[fnodes]
    has_plus_of = np.bincount(face_of_node, weights=node_side > 0,
                              minlength=g.faces.num) > 0
    has_minus_of = np.bincount(face_of_node, weights=node_side < 0,
                               minlength=g.faces.num) > 0
    on_plane_of = ~(has_plus_of | has_minus_of)          # wholly in the plane
    touches_plane_of = np.bincount(face_of_node, weights=node_side == 0,
                                   minlength=g.faces.num) > 0

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
    for f in np.flatnonzero(has_plus_of & has_minus_of):
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

    # Faces are split for every cell the plane fully traverses, but MRST's
    # splitCells only divides the cell itself when both halves keep at least
    # `griddim` faces. Cells failing that are left whole; when the odd half is
    # a single face lying in the plane, that face becomes a slice face instead
    # - MRST's "cut along face" path.
    split_cell = fully_cut.copy()
    slice_only_faces = set()
    if not only_faces:
        halves = {int(c): ([], []) for c in np.flatnonzero(fully_cut)}
        for f in range(g.faces.num):
            has_plus, has_minus = has_plus_of[f], has_minus_of[f]
            for c in neigh[f]:
                if c < 0 or not fully_cut[c]:
                    continue
                minus, plus = halves[int(c)]
                if has_plus and has_minus:                # duplicated by the cut
                    minus.append(f)
                    plus.append(f)
                elif has_plus:
                    plus.append(f)
                else:                                     # also the all-on-plane case
                    minus.append(f)
        declined = 0
        for c, (minus, plus) in halves.items():
            small, large = (minus, plus) if len(minus) <= len(plus) else (plus, minus)
            if len(small) >= g.griddim:
                continue
            split_cell[c] = False
            if len(small) == 1 and small[0] not in large:
                slice_only_faces.add(small[0])
            else:
                declined += 1
        if declined:
            warnings.warn(
                f"cutting of {declined} out of {len(halves)} cells failed",
                RuntimeWarning, stacklevel=3)

        # a face lying wholly in the plane already *is* the slice there, so it
        # is reported as one - this is what a cut snapped onto a column of
        # existing faces produces, with no cell split at all
        for f in np.flatnonzero(on_plane_of):
            seg = fnodes[node_pos[f]:node_pos[f + 1]]
            if any(c >= 0 and split_cell[c] for c in neigh[f]):
                continue
            if window is not None:
                _, _, L = window
                t = [boundary_t(p) for p in coords[seg]]
                if min(t) < -1e-9 * max(L, 1.0) or max(t) > L + 1e-9 * max(L, 1.0):
                    continue                             # outside the cut segment
            slice_only_faces.add(f)

    cut_ids = np.flatnonzero(split_cell)
    plus_id = np.full(nc0, -1, dtype=np.int64)
    if not only_faces:
        plus_id[cut_ids] = nc0 + np.arange(cut_ids.size)  # + half of cut cell

    def half(c, side_pos):
        if c < 0:
            return -1
        if only_faces:
            return c
        return plus_id[c] if (side_pos and split_cell[c]) else c

    out_nodes, out_np, out_neigh = [], [0], []
    face_parent, face_status = [], []
    # per cut cell, the boundary edges (pairs of on-plane / crossing points) that
    # bound its cut polygon
    cell_cut_edges = {int(c): [] for c in cut_ids}
    plane_rep = {}

    def point_of(node):
        return (coords[node] if node < coords.shape[0]
                else new_coords[node - coords.shape[0]])

    def rep(node):
        """Canonical index for cut-plane points that coincide within `tol`.

        A collapsed (pinched) cell carries several distinct node indices at the
        same coordinate; the cut polygon has to see those as a single corner or
        its boundary edges never close into a ring.
        """
        bucket = np.floor(point_of(node) / tol).astype(np.int64)
        for offset in _BUCKET_NEIGHBOURS:
            other = plane_rep.get(tuple(bucket + offset))
            if other is not None and np.linalg.norm(
                    point_of(other) - point_of(node)) <= tol:
                return other
        plane_rep[tuple(bucket)] = node
        return node

    def record_edge(a, b, bnd):
        ring = []
        for node in bnd:                                 # merge coincident corners
            node = rep(node)
            if node not in ring:
                ring.append(node)
        if len(ring) != 2:                               # not a single boundary edge
            return
        for c in (a, b):
            if c >= 0 and split_cell[c]:
                cell_cut_edges[int(c)].append((ring[0], ring[1]))

    # which faces the cut actually divides, decided for all of them at once so
    # the loop below can hand an untouched face straight through
    left, right = neigh[:, 0], neigh[:, 1]
    splitting_of = (((left >= 0) & fully_cut[np.maximum(left, 0)])
                    | ((right >= 0) & fully_cut[np.maximum(right, 0)]))
    divided_of = has_plus_of & has_minus_of & splitting_of

    for f in range(g.faces.num):
        seg = fnodes[node_pos[f]:node_pos[f + 1]]
        a, b = neigh[f]

        if divided_of[f]:                                # face is truly split
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
            out_nodes.append(np.asarray(minus, dtype=np.int64))
            out_np.append(out_np[-1] + len(minus))
            out_neigh.append([half(a, False), half(b, False)])
            face_parent.append(f)
            face_status.append(2)
            out_nodes.append(np.asarray(plus, dtype=np.int64))
            out_np.append(out_np[-1] + len(plus))
            out_neigh.append([half(a, True), half(b, True)])
            face_parent.append(f)
            face_status.append(2)
        else:                                            # face on one side
            # an all-on-plane face keeps its nodes and joins the minus half;
            # where it is the only face of that half the cell is not split at
            # all and the face itself becomes the slice
            side_pos = has_plus_of[f]                    # False also for all-0
            out_nodes.append(seg)                        # kept as-is, in bulk
            out_np.append(out_np[-1] + seg.size)
            out_neigh.append([half(a, side_pos), half(b, side_pos)])
            face_parent.append(f)
            face_status.append(3 if f in slice_only_faces else 1)
            # a face touching the plane along an edge (two on-plane nodes)
            # still bounds the cut polygon of its cut cells
            if touches_plane_of[f]:
                record_edge(a, b, [int(x) for x in seg[side[seg] == 0]])

    all_coords = np.vstack([coords, np.array(new_coords)]) if new_coords else coords

    # one cut-polygon face per cut cell, between its - and + halves; wind it so
    # its normal points from the minus half to the plus half (along +n)
    split_cell_ids = np.empty(0, dtype=np.int64) if only_faces else cut_ids
    for c in split_cell_ids:
        ring = _order_ring(cell_cut_edges[int(c)])
        if np.dot(_polygon_normal(all_coords[ring]), n) < 0:
            ring = ring[::-1]
        out_nodes.append(np.asarray(ring, dtype=np.int64))
        out_np.append(out_np[-1] + len(ring))
        out_neigh.append([int(c), int(plus_id[c])])
        face_parent.append(-1)
        face_status.append(3)
    neighbors = np.asarray(out_neigh, dtype=np.int64)
    faces_nodes = (np.concatenate(out_nodes) if out_nodes
                   else np.zeros(0, dtype=np.int64)).astype(np.int64)
    faces_pos = np.asarray(out_np, dtype=np.int64)
    num_cells = nc0 + split_cell_ids.size
    cell_faces, cell_face_pos = _rebuild_cell_faces(num_cells, neighbors)

    source_index = (
        np.asarray(g.cells.indexMap, dtype=np.int64)
        if g.cells.indexMap is not None else np.arange(nc0, dtype=np.int64)
    )
    cell_parent = np.concatenate([
        np.arange(nc0, dtype=np.int64), split_cell_ids
    ])
    cell_tags = None
    if g.cells.tags is not None:
        source_tags = {}
        source_cell = g.cell_no()
        for incidence, (cell, face) in enumerate(zip(source_cell, g.cells.faces)):
            source_tags[(int(cell), int(face))] = np.asarray(g.cells.tags)[incidence]
        output_cell = np.repeat(
            np.arange(num_cells, dtype=np.int64), np.diff(cell_face_pos)
        )
        cell_tags = np.asarray([
            source_tags.get(
                (int(cell_parent[cell]), int(face_parent[face])), 0
            )
            for cell, face in zip(output_cell, cell_faces)
        ])
    cell_status = np.ones(num_cells, dtype=np.int64)
    cell_status[split_cell_ids] = 2
    cell_status[nc0:] = 2
    result = Grid(
        cells=Cells(num=int(num_cells), facePos=cell_face_pos, faces=cell_faces,
                    indexMap=source_index[cell_parent], tags=cell_tags),
        faces=Faces(num=int(faces_pos.size - 1), nodePos=faces_pos,
                    nodes=faces_nodes, neighbors=neighbors,
                    tag=(
                        np.asarray(g.faces.tag)[np.maximum(face_parent, 0)].copy()
                        if g.faces.tag is not None else None
                    )),
        nodes=Nodes(num=int(all_coords.shape[0]), coords=all_coords),
        griddim=g.griddim, cartDims=g.cartDims,
        type=list(g.type) + ["sliceGrid"])
    if result.faces.tag is not None:
        result.faces.tag[np.asarray(face_parent) < 0] = 0
    old_cells = np.ones(nc0, dtype=bool)
    old_cells[split_cell_ids] = False
    old_faces = np.ones(g.faces.num, dtype=bool)
    split_parent = np.asarray(face_parent)[np.asarray(face_status) == 2]
    old_faces[np.unique(split_parent)] = False
    mapping = SimpleNamespace(
        old=SimpleNamespace(cells=old_cells, faces=old_faces),
        new=SimpleNamespace(
            cells=cell_status,
            faces=np.asarray(face_status, dtype=np.int64),
            nodes=np.concatenate([
                np.ones(g.nodes.num, dtype=np.int64),
                np.full(all_coords.shape[0] - g.nodes.num, 2, dtype=np.int64),
            ]),
        ),
        parent=SimpleNamespace(
            cells=cell_parent,
            faces=np.asarray(face_parent, dtype=np.int64),
            nodes=np.concatenate([
                np.arange(g.nodes.num, dtype=np.int64),
                np.full(all_coords.shape[0] - g.nodes.num, -1, dtype=np.int64),
            ]),
        ),
    )
    return result, mapping


def _compose_mappings(previous, current):
    """Compose mappings from repeated cuts back to the original grid."""
    if previous is None:
        return current
    cell_parent = current.parent.cells
    face_parent = current.parent.faces
    node_parent = current.parent.nodes
    mapped_cells = previous.parent.cells[cell_parent]
    mapped_faces = np.full(face_parent.shape, -1, dtype=np.int64)
    valid_faces = face_parent >= 0
    mapped_faces[valid_faces] = previous.parent.faces[face_parent[valid_faces]]
    mapped_nodes = np.full(node_parent.shape, -1, dtype=np.int64)
    valid_nodes = node_parent >= 0
    mapped_nodes[valid_nodes] = previous.parent.nodes[node_parent[valid_nodes]]
    old_cells = previous.old.cells.copy()
    changed_cells = previous.parent.cells[~current.old.cells]
    old_cells[np.unique(changed_cells)] = False
    old_faces = previous.old.faces.copy()
    changed_faces = previous.parent.faces[~current.old.faces]
    changed_faces = changed_faces[changed_faces >= 0]
    old_faces[np.unique(changed_faces)] = False
    return SimpleNamespace(
        old=SimpleNamespace(
            cells=old_cells,
            faces=old_faces,
        ),
        new=SimpleNamespace(
            cells=np.maximum(current.new.cells, previous.new.cells[cell_parent]),
            faces=np.where(
                valid_faces,
                np.maximum(current.new.faces, previous.new.faces[face_parent]),
                current.new.faces,
            ),
            nodes=np.where(
                valid_nodes,
                np.maximum(current.new.nodes, previous.new.nodes[node_parent]),
                current.new.nodes,
            ),
        ),
        parent=SimpleNamespace(
            cells=mapped_cells, faces=mapped_faces, nodes=mapped_nodes,
        ),
    )


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
