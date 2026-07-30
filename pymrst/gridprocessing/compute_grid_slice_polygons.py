"""Polygons where a plane cuts each grid cell (MRST `computeGridSlicePolygons`).

Given a 3D grid and a cutting plane, this returns, for every cell the plane
passes through, the polygon of that cell's cross-section: the ordered ring of
points where the plane crosses the cell's edges. It is the geometric core of
`sliceGrid` (which then splits the cells along these polygons) and is directly
useful on its own for cross-section plots and cut-plane extraction.

The plane is given either as a point plus a normal, or as a line segment plus a
"cut direction" it is swept along (the plane then contains the segment and the
cut direction). For each face edge the plane-crossing point is found, points are
grouped by the cells sharing the face, and each cell's points are ordered by
angle about their centroid in the 2D cut-plane frame `(t, p·cutDir)`.

Piecewise-linear paths, face/cell subsets, endpoint snapping, and finite segment
width are supported. Bounding-box selection remains an internal optimisation,
not part of the numerical result.
"""

from typing import Dict, Optional

import numpy as np
from scipy.linalg import null_space


def _face_edges(g, face_ix=None):
    """All faces' edges as `(p1, p2, face_no)` over node coordinates."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    a, b, fno = [], [], []
    faces = (
        np.arange(g.faces.num, dtype=np.int64)
        if face_ix is None else np.asarray(face_ix, dtype=np.int64).ravel()
    )
    for f in faces:
        seg = fnodes[node_pos[f]:node_pos[f + 1]]
        m = seg.size
        for k in range(m):
            a.append(seg[k])
            b.append(seg[(k + 1) % m])
            fno.append(f)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return coords[a], coords[b], np.asarray(fno, dtype=np.int64)


def compute_grid_slice_polygons(g, coord, normal: Optional[np.ndarray] = None,
                                cut_dir: Optional[np.ndarray] = None,
                                tol: float = 1e-6,
                                remove_degenerates: bool = True,
                                remove_repeated: bool = True,
                                face_ix=None, cell_ix=None, radius=None,
                                snap_tol: float = 0.0) -> Dict:
    """Return the per-cell cut polygons of the plane through `coord`.

    `coord` is either a single point (then `normal` is required) or a line
    segment's two endpoints (then `cut_dir` is required). Returns a dict with
    `cell_ix` (the cut cells), `node_pos` (CSR into the vertex arrays),
    `coords_3d` (vertex coordinates) and `coords_2d` (their `(t, p·cutDir)`
    cut-plane coordinates).
    """
    coord = np.atleast_2d(np.asarray(coord, dtype=float))
    if coord.shape[0] > 2:
        pieces = [
            compute_grid_slice_polygons(
                g, coord[k:k + 2], cut_dir=cut_dir, tol=tol,
                remove_degenerates=remove_degenerates,
                remove_repeated=remove_repeated, face_ix=face_ix,
                cell_ix=cell_ix, radius=radius, snap_tol=snap_tol,
            )
            for k in range(coord.shape[0] - 1)
        ]
        return _combine_piecewise(pieces)
    node_xyz = np.asarray(g.nodes.coords, dtype=float)

    if coord.shape[0] == 1:
        assert normal is not None, "a single point needs a plane normal"
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        R = np.linalg.norm(node_xyz.max(0) - node_xyz.min(0))
        basis = null_space(normal[None, :])            # (3, 2) plane basis
        v, cut_dir = basis[:, 0], basis[:, 1]
        seg = np.array([coord[0] - R * v, coord[0] + R * v])
    else:
        assert cut_dir is not None, "a segment needs a cut direction"
        seg = coord[:2]
        cut_dir = np.asarray(cut_dir, dtype=float)
    cut_dir = cut_dir / np.linalg.norm(cut_dir)

    vu = seg[1] - seg[0]
    vu = vu / np.linalg.norm(vu)
    n = null_space(np.vstack([vu, cut_dir]))           # plane normal (3, 1)
    assert n.shape[1] == 1, "segment direction and cut direction are parallel"
    n = n[:, 0]
    d = seg[0].dot(n)

    p1, p2, face_no = _face_edges(g, face_ix)
    v = p2 - p1
    vn = v.dot(n)
    ok = np.abs(vn) > tol * np.linalg.norm(v, axis=1)
    s = np.full(vn.shape, np.nan)
    s[ok] = -(p1[ok].dot(n) - d) / vn[ok]
    if snap_tol > 0:
        s[np.abs(s) <= snap_tol] = 0.0
        s[np.abs(s - 1.0) <= snap_tol] = 1.0
    hit = np.flatnonzero((s >= 0) & (s <= 1))
    if hit.size == 0:
        return {"cell_ix": np.array([], dtype=np.int64),
                "node_pos": np.array([0]), "coords_3d": np.zeros((0, 3)),
                "coords_2d": np.zeros((0, 2))}

    pts = p1[hit] + s[hit, None] * v[hit]
    fno = face_no[hit]

    # each crossing point belongs to both cells sharing the face
    cno = np.asarray(g.faces.neighbors, dtype=np.int64)[fno]
    is_pos = cno >= 0
    cell_of = np.concatenate([cno[is_pos[:, 0], 0], cno[is_pos[:, 1], 1]])
    pix = np.concatenate([np.flatnonzero(is_pos[:, 0]),
                          np.flatnonzero(is_pos[:, 1])])
    pts = pts[pix]

    t = pts.dot(vu) - seg[0].dot(vu)
    p_cut = pts.dot(cut_dir) - seg[0].dot(cut_dir)

    cells = np.unique(cell_of)
    if cell_ix is not None:
        cells = np.intersect1d(cells, np.asarray(cell_ix, dtype=np.int64))
    out_cells, out_pos, out_3d, out_2d = [], [0], [], []
    for c in cells:
        m = cell_of == c
        pp = np.column_stack([t[m], p_cut[m]])
        cent = pp.mean(axis=0)
        theta = np.arctan2(pp[:, 1] - cent[1], pp[:, 0] - cent[0])
        order = np.argsort(theta)
        if remove_repeated:
            # each edge crossing is generated once per adjoining face; drop the
            # consecutive duplicates left after angle-sorting
            pp_o = pp[order]
            same = np.concatenate([[False], np.all(
                np.abs(np.diff(pp_o, axis=0)) < np.sqrt(np.finfo(float).eps),
                axis=1)])
            order = order[~same]
        pp_ordered = pp[order]
        finite_segment = coord.shape[0] == 2
        if finite_segment:
            width = np.inf if radius is None else float(np.asarray(radius).ravel()[0])
            pp_ordered = _clip_rectangle(
                pp_ordered, 0.0, np.linalg.norm(seg[1] - seg[0]), -width, width
            )
            if pp_ordered.size:
                pts_ordered = (
                    seg[0] + pp_ordered[:, :1] * vu
                    + pp_ordered[:, 1:] * cut_dir
                )
            else:
                pts_ordered = np.zeros((0, node_xyz.shape[1]))
        else:
            pts_m = pts[m]
            pts_ordered = pts_m[order]
            if radius is not None:
                radii = np.asarray(radius, dtype=float).ravel()
                r = float(radii[0])
                keep = np.sum((pts_ordered - coord[0]) ** 2, axis=1) <= r*r + tol
                pp_ordered, pts_ordered = pp_ordered[keep], pts_ordered[keep]
        if remove_degenerates and pp_ordered.shape[0] <= g.griddim - 1:
            continue
        out_cells.append(c)
        out_3d.append(pts_ordered)
        out_2d.append(pp_ordered)
        out_pos.append(out_pos[-1] + pp_ordered.shape[0])

    if not out_cells:
        return {"cell_ix": np.array([], dtype=np.int64),
                "node_pos": np.array([0]), "coords_3d": np.zeros((0, 3)),
                "coords_2d": np.zeros((0, 2))}

    return {"cell_ix": np.asarray(out_cells, dtype=np.int64),
            "node_pos": np.asarray(out_pos, dtype=np.int64),
            "coords_3d": np.vstack(out_3d),
            "coords_2d": np.vstack(out_2d)}


def _empty(dim=3):
    return {
        "cell_ix": np.array([], dtype=np.int64),
        "node_pos": np.array([0], dtype=np.int64),
        "coords_3d": np.zeros((0, dim)),
        "coords_2d": np.zeros((0, 2)),
        "segment_ix": np.array([], dtype=np.int64),
    }


def _combine_piecewise(pieces):
    nonempty = [piece for piece in pieces if piece["cell_ix"].size]
    if not nonempty:
        return _empty()
    counts = [np.diff(piece["node_pos"]) for piece in nonempty]
    all_counts = np.concatenate(counts)
    segment_ix = np.concatenate([
        np.full(piece["cell_ix"].size, k, dtype=np.int64)
        for k, piece in enumerate(pieces) if piece["cell_ix"].size
    ])
    return {
        "cell_ix": np.concatenate([p["cell_ix"] for p in nonempty]),
        "node_pos": np.concatenate([[0], np.cumsum(all_counts)]),
        "coords_3d": np.vstack([p["coords_3d"] for p in nonempty]),
        "coords_2d": np.vstack([p["coords_2d"] for p in nonempty]),
        "segment_ix": segment_ix,
    }


def _clip_rectangle(poly, xmin, xmax, ymin, ymax):
    """Sutherland-Hodgman clipping of an ordered 2D polygon."""
    out = np.asarray(poly, dtype=float)
    for axis, bound, keep_less in (
        (0, xmin, False), (0, xmax, True), (1, ymin, False), (1, ymax, True),
    ):
        if out.size == 0:
            break
        result = []
        for p, q in zip(out, np.roll(out, -1, axis=0)):
            pin = p[axis] <= bound if keep_less else p[axis] >= bound
            qin = q[axis] <= bound if keep_less else q[axis] >= bound
            if pin:
                result.append(p)
            if pin != qin:
                alpha = (bound - p[axis]) / (q[axis] - p[axis])
                result.append(p + alpha * (q - p))
        out = np.asarray(result, dtype=float)
    return out.reshape((-1, 2))


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_vertical_plane_cuts_a_column_of_cells():
        """A vertical plane x = 1.5 through a 3x1x2 grid cuts the two middle
        cells (i=1) into rectangles of 4 vertices each."""
        g = compute_geometry(cart_grid([3, 1, 2], [3.0, 1.0, 2.0]))
        poly = compute_grid_slice_polygons(
            g, np.array([[1.5, 0.0, 0.0], [1.5, 1.0, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0]))
        counts = np.diff(poly["node_pos"])
        assert np.all(counts == 4), counts            # rectangular cross-sections
        # every vertex lies on the plane x = 1.5
        assert np.allclose(poly["coords_3d"][:, 0], 1.5)
        print(f"ok test_vertical_plane_cuts_a_column_of_cells "
              f"({poly['cell_ix'].size} cells cut)")

    def test_point_normal_defines_the_same_plane():
        """Point + x-normal gives the x = 1.5 plane too."""
        g = compute_geometry(cart_grid([3, 1, 2], [3.0, 1.0, 2.0]))
        poly = compute_grid_slice_polygons(
            g, np.array([1.5, 0.5, 1.0]), normal=np.array([1.0, 0.0, 0.0]))
        assert np.allclose(poly["coords_3d"][:, 0], 1.5)
        assert poly["cell_ix"].size == 2
        print("ok test_point_normal_defines_the_same_plane")

    def test_polygon_vertices_are_ordered_around_the_ring():
        """The 2D cut-plane vertices of a cell form a simple (non-crossing)
        quadrilateral - consecutive angles increase."""
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        poly = compute_grid_slice_polygons(
            g, np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            cut_dir=np.array([0.0, 0.0, 1.0]))
        p = poly["coords_2d"][poly["node_pos"][0]:poly["node_pos"][1]]
        cent = p.mean(axis=0)
        ang = np.arctan2(p[:, 1] - cent[1], p[:, 0] - cent[0])
        assert np.all(np.diff(ang) >= 0), ang
        print("ok test_polygon_vertices_are_ordered_around_the_ring")

    test_vertical_plane_cuts_a_column_of_cells()
    test_point_normal_defines_the_same_plane()
    test_polygon_vertices_are_ordered_around_the_ring()
    print("\nAll tests passed!")
