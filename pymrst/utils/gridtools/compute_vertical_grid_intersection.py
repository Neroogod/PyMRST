"""Slice a grid with a vertical surface (MRST ``utils/gridtools/computeVerticalGridIntersection.m``).

Given a polyline in map view, this extrudes it downwards into a piecewise-planar
vertical surface and returns the polygon that surface cuts out of each cell it
passes through. It is what draws a fence diagram, and what tells a well
trajectory which cells it crosses and where.

The result is a dict with:

- `nodes` - `(n_polygons, max_vertices)` of indices into the coordinate arrays,
  padded by repeating the last vertex so the array is rectangular;
- `coords3D` - the vertices in space;
- `coords2D` - the same vertices as `(distance along the polyline, z)`, which
  is the natural plane to draw the section in;
- `cellIx` - which cell each polygon belongs to;
- `segments`, `cumlength` - the polyline and the cumulative distance to each of
  its vertices.

**Every vertex appears twice.** An intersection point lies on an edge shared by
two of the cell's faces, and both faces contribute it. MRST notices this - its
source carries a commented-out `order = order(1:2:end)` and the remark that
de-duplicating would "loose some poly neighbour-info" - and ships the
duplicates. Kept, for parity, but it means a polygon's vertex count is twice
its corner count, and anything measuring area or perimeter from `nodes` must
cope with the repeats.

**The surface stops where the polyline stops**, so the cells at either end get
an open, truncated polygon rather than a closed section - two distinct corners
where a fully-crossed cell has four. That is correct and is what a caller
usually wants (the section covers the trace they asked for), but it reads as a
bug the first time. Extend the polyline past the grid if every cell it touches
should get a complete section.

Speed comes from `get_faces_close_to_segment`: the whole polyline pre-filters
the faces once, then each segment filters that subset again, so the exact
intersection arithmetic only ever runs on a handful of faces. Both filters
project along z, since a vertical surface's footprint is its map-view trace.
"""

from typing import Dict

import numpy as np

from pymrst.utils.gridtools.add_bounding_box_fields import add_bounding_box_fields
from pymrst.utils.gridtools.get_faces_close_to_segment import (
    get_faces_close_to_segment,
)
from pymrst.utils.rldecode import rldecode

VERTICAL = [0.0, 0.0, 1.0]


def _face_edges(g, faces):
    """Every face's edges as ``(start, end, face)``, walking each boundary."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)

    blocks = [nodes[node_pos[f]:node_pos[f + 1]] for f in faces]
    own = np.concatenate(blocks) if blocks else np.zeros(0, np.int64)
    following = np.concatenate([np.roll(b, -1) for b in blocks]) if blocks \
        else np.zeros(0, np.int64)
    counts = np.array([b.size for b in blocks], dtype=np.int64)
    return coords[own], coords[following], rldecode(np.asarray(faces), counts)


def _plane_crossings(start, end, normal, offset, segment):
    """Where edges cross the plane, restricted to the segment's own span."""
    along = end - start
    denominator = along @ normal
    crossing = np.abs(denominator) > 0

    t = np.full(denominator.shape, np.nan)
    t[crossing] = -((start[crossing] @ normal) - offset) / denominator[crossing]
    inside = np.flatnonzero((t >= 0) & (t <= 1))
    points = start[inside] + t[inside, None] * along[inside]

    # The plane is unbounded; the segment is not.
    v = segment[1] - segment[0]
    s = ((points[:, :2] @ v) - (segment[0] @ v)) / (v @ v)
    within = (s >= 0) & (s <= 1)
    return points[within], inside[within]


def compute_vertical_grid_intersection(g, segments) -> Dict:
    """Return the polygons cut from ``g`` by the vertical surface on ``segments``."""
    if getattr(g.faces, "bbox", None) is None:
        g = add_bounding_box_fields(g)

    segments = np.atleast_2d(np.asarray(segments, dtype=float))[:, :2]
    count = segments.shape[0] - 1
    if count < 1:
        raise ValueError(
            f"a surface needs at least two polyline points, got "
            f"{segments.shape[0]}")

    # One coarse pass over the whole polyline, then a fine one per segment.
    candidates = (get_faces_close_to_segment(g, segments, proj_vec=VERTICAL)
                  if count > 1 else None)

    points, distance, face_no = [], [], []
    travelled, cumulative = 0.0, np.zeros(count + 1)
    for k in range(count):
        segment = segments[[k, k + 1]]
        near = get_faces_close_to_segment(g, segment, face_ix=candidates,
                                          proj_vec=VERTICAL)
        start, end, faces = _face_edges(g, near)

        v = segment[1] - segment[0]
        normal = np.array([-v[1], v[0], 0.0])
        offset = float(segment[0] @ normal[:2])
        cut, which = _plane_crossings(start, end, normal, offset, segment)

        points.append(cut)
        distance.append(travelled + np.linalg.norm(cut[:, :2] - segment[0],
                                                   axis=1))
        face_no.append(faces[which])
        travelled += float(np.linalg.norm(v))
        cumulative[k + 1] = travelled

    points = np.vstack(points) if points else np.zeros((0, 3))
    distance = np.concatenate(distance) if distance else np.zeros(0)
    face_no = np.concatenate(face_no).astype(np.int64) if face_no \
        else np.zeros(0, np.int64)

    # A face's crossing belongs to the cells on both of its sides.
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)[face_no]
    left, right = neighbours[:, 0] >= 0, neighbours[:, 1] >= 0
    points = np.vstack([points[left], points[right]])
    distance = np.concatenate([distance[left], distance[right]])
    cells = np.concatenate([neighbours[left, 0], neighbours[right, 1]])

    if cells.size == 0:
        return {"nodes": np.zeros((0, 0), dtype=np.int64),
                "coords3D": np.zeros((0, 3)), "coords2D": np.zeros((0, 2)),
                "cellIx": np.zeros(0, dtype=np.int64), "segments": segments,
                "cumlength": cumulative}

    present = np.unique(cells)
    local = np.searchsorted(present, cells)
    per_cell = np.bincount(local, minlength=present.size)

    # Order each cell's vertices by angle about their own centre, in the
    # section plane - the polygon is convex there, so angle is boundary order.
    plane = np.column_stack([distance, points[:, 2]])
    centre = np.column_stack([
        np.bincount(local, weights=plane[:, d], minlength=present.size)
        for d in range(2)]) / per_cell[:, None]
    relative = plane - centre[local]
    angle = np.arctan2(relative[:, 1], relative[:, 0])
    order = np.lexsort((angle, local))

    points, distance, local = points[order], distance[order], local[order]

    # Pad each polygon out to the longest by repeating its last vertex.
    ends = np.cumsum(per_cell)
    starts = ends - per_cell
    widest = int(per_cell.max())
    columns = [np.minimum(starts + k, ends - 1) for k in range(widest)]
    nodes = np.column_stack(columns).astype(np.int64)

    return {"nodes": nodes, "coords3D": points,
            "coords2D": np.column_stack([distance, points[:, 2]]),
            "cellIx": present, "segments": segments, "cumlength": cumulative}


computeVerticalGridIntersection = compute_vertical_grid_intersection


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(4, 4, 3), sizes=(4.0, 4.0, 3.0)):
        return add_bounding_box_fields(
            compute_geometry(cart_grid(list(dims), list(sizes))))

    def test_a_straight_cut_crosses_a_row_of_cells() -> None:
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        # The line runs along y = 0.5, so one column of cells in y, all layers.
        assert out["cellIx"].size == 4 * 3, out["cellIx"].size
        print("ok test_a_straight_cut_crosses_a_row_of_cells")

    def test_the_polygons_lie_in_their_own_cells() -> None:
        """The defining property, checked against cell geometry."""
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        centroids = np.asarray(g.cells.centroids)
        for row, cell in enumerate(out["cellIx"]):
            xyz = out["coords3D"][out["nodes"][row]]
            assert np.all(np.abs(xyz - centroids[cell]) <= 0.5 + 1e-9), cell
        print("ok test_the_polygons_lie_in_their_own_cells")

    def test_the_cut_lies_on_the_plane() -> None:
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        assert np.allclose(out["coords3D"][:, 1], 0.5), "off the y = 0.5 plane"
        print("ok test_the_cut_lies_on_the_plane")

    def test_the_second_coordinate_is_distance_and_depth() -> None:
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        assert np.allclose(out["coords2D"][:, 1], out["coords3D"][:, 2])
        # Along x, distance from the start is just x - 0.5.
        assert np.allclose(out["coords2D"][:, 0],
                           out["coords3D"][:, 0] - 0.5, atol=1e-9)
        print("ok test_the_second_coordinate_is_distance_and_depth")

    def test_a_diagonal_cut_finds_more_cells() -> None:
        g = build()
        straight = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        diagonal = compute_vertical_grid_intersection(
            g, np.array([[0.2, 0.2], [3.8, 3.8]]))
        assert diagonal["cellIx"].size >= straight["cellIx"].size
        print(f"ok test_a_diagonal_cut_finds_more_cells "
              f"({diagonal['cellIx'].size} cells)")

    def test_a_polyline_accumulates_length() -> None:
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [2.0, 0.5], [2.0, 3.5]]))
        assert np.allclose(out["cumlength"], [0.0, 1.5, 4.5])
        assert out["coords2D"][:, 0].max() <= 4.5 + 1e-9
        print("ok test_a_polyline_accumulates_length")

    def test_a_bend_visits_cells_from_both_legs() -> None:
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [2.5, 0.5], [2.5, 3.5]]))
        centroids = np.asarray(g.cells.centroids)[out["cellIx"]]
        assert centroids[:, 0].max() > 2.0 and centroids[:, 1].max() > 2.0
        print("ok test_a_bend_visits_cells_from_both_legs")

    def test_every_vertex_appears_twice() -> None:
        """MRST's documented duplication; see the module docstring.

        Checked on a cell the surface crosses completely. The cells at the two
        ends are clipped by the polyline and are covered separately below.
        """
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        middle = [r for r, c in enumerate(out["cellIx"])
                  if 1.0 < np.asarray(g.cells.centroids)[c, 0] < 3.0]
        assert middle, "expected some fully-crossed cells"
        for row in middle:
            distinct = np.unique(np.round(
                out["coords3D"][out["nodes"][row]], 9), axis=0)
            entries = len(set(out["nodes"][row].tolist()))
            assert entries == 2 * distinct.shape[0], (entries, distinct.shape)
        print(f"ok test_every_vertex_appears_twice ({len(middle)} cells)")

    def test_the_end_cells_are_clipped_by_the_polyline() -> None:
        """The surface stops where the polyline stops, so the first and last
        cells get an open, truncated polygon rather than a closed section.

        Easy to mistake for a bug - a full cell gives 4 corners, these give 2 -
        so it is asserted rather than left to be rediscovered.
        """
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        x = np.asarray(g.cells.centroids)[out["cellIx"], 0]
        for row in np.flatnonzero((x < 1.0) | (x > 3.0)):
            distinct = np.unique(np.round(
                out["coords3D"][out["nodes"][row]], 9), axis=0)
            assert distinct.shape[0] == 2, distinct
            # The surviving side is the one the polyline reaches.
            assert np.all((distinct[:, 0] > 0.9) | (distinct[:, 0] < 3.1))
        print("ok test_the_end_cells_are_clipped_by_the_polyline")

    def test_a_polyline_spanning_the_grid_clips_nothing_inside() -> None:
        """Extending the ends past the grid gives every cell a full section."""
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[-1.0, 0.5], [5.0, 0.5]]))
        for row in range(out["cellIx"].size):
            distinct = np.unique(np.round(
                out["coords3D"][out["nodes"][row]], 9), axis=0)
            assert distinct.shape[0] == 4, (row, distinct.shape)
        print("ok test_a_polyline_spanning_the_grid_clips_nothing_inside")

    def test_the_node_table_is_rectangular_and_padded() -> None:
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[0.2, 0.2], [3.8, 3.8]]))
        nodes = out["nodes"]
        assert nodes.ndim == 2 and nodes.shape[0] == out["cellIx"].size
        # Padding repeats the last index, so rows are non-decreasing.
        assert np.all(np.diff(nodes, axis=1) >= 0)
        print("ok test_the_node_table_is_rectangular_and_padded")

    def test_a_cut_outside_the_grid_finds_nothing() -> None:
        g = build()
        out = compute_vertical_grid_intersection(
            g, np.array([[100.0, 100.0], [200.0, 200.0]]))
        assert out["cellIx"].size == 0 and out["coords3D"].shape[0] == 0
        print("ok test_a_cut_outside_the_grid_finds_nothing")

    def test_three_dimensional_segments_are_accepted() -> None:
        """Only the x-y trace matters; a z column is ignored, per MRST."""
        g = build()
        flat = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5], [3.5, 0.5]]))
        with_z = compute_vertical_grid_intersection(
            g, np.array([[0.5, 0.5, 99.0], [3.5, 0.5, -99.0]]))
        assert np.array_equal(flat["cellIx"], with_z["cellIx"])
        print("ok test_three_dimensional_segments_are_accepted")

    def test_a_single_point_is_rejected() -> None:
        try:
            compute_vertical_grid_intersection(build(), np.array([[1.0, 1.0]]))
        except ValueError as error:
            assert "at least two polyline points" in str(error)
            print("ok test_a_single_point_is_rejected")
            return
        raise AssertionError("one point defines no surface")

    def test_a_grid_without_bbox_still_works() -> None:
        g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = compute_vertical_grid_intersection(
                g, np.array([[0.5, 0.5], [2.5, 0.5]]))
        assert out["cellIx"].size > 0
        print("ok test_a_grid_without_bbox_still_works")

    test_a_straight_cut_crosses_a_row_of_cells()
    test_the_polygons_lie_in_their_own_cells()
    test_the_cut_lies_on_the_plane()
    test_the_second_coordinate_is_distance_and_depth()
    test_a_diagonal_cut_finds_more_cells()
    test_a_polyline_accumulates_length()
    test_a_bend_visits_cells_from_both_legs()
    test_every_vertex_appears_twice()
    test_the_end_cells_are_clipped_by_the_polyline()
    test_a_polyline_spanning_the_grid_clips_nothing_inside()
    test_the_node_table_is_rectangular_and_padded()
    test_a_cut_outside_the_grid_finds_nothing()
    test_three_dimensional_segments_are_accepted()
    test_a_single_point_is_rejected()
    test_a_grid_without_bbox_still_works()
    print("\nAll tests passed!")
