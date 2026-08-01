"""Faces near a line segment (MRST ``utils/gridtools/getFacesCloseToSegment.m``).

A cheap **conservative** filter: it returns every face the segment could
plausibly touch, and some it does not. The point is to cut a grid's worth of
faces down to a handful before running a real intersection test, which costs
far more per face. Treating the result as "the faces the segment crosses" is
the way to misuse it.

Conservative in the direction that matters - it does not miss faces - and the
tests check exactly that, by walking the segment cell by cell and asserting
every genuinely crossed face comes back.

Each face is approximated as its centroid plus `fac` times its bounding-box
extent. `fac = 2/3` is MRST's default; `1/2` would be exact for a Cartesian
grid, and the extra third is slack for faces whose centroid is not their box
centre. That approximation is inherited from `add_bounding_box_fields`, whose
docstring explains why the reconstruction can fall short - which is the reason
the slack exists and the reason it is not a guarantee.

Given more than two points, a line of best fit is taken through them by SVD and
the search radius grows by how far the points stray from it. That makes a
polyline usable as a well trajectory without splitting it into segments.

`proj_vec` collapses the problem along a direction, answering "close in map
view" rather than "close in space" - the usual case being a vertical well seen
from above. A canonical direction drops to the other two coordinates exactly; a
general one projects out the component and inflates the box by `sqrt(3)`, since
a box's extent under an arbitrary rotation is not its extent along the axes.

**MRST's `radius` option is not implemented, because MRST does not implement it
either** - it is declared in the options struct and never read. Accepting a
parameter that silently does nothing is worse than not accepting it.
"""

import warnings

import numpy as np

from pymrst.utils.gridtools.add_bounding_box_fields import add_bounding_box_fields

TOL = float(np.sqrt(np.finfo(float).eps))


def _canonical(direction):
    """The two coordinates left when projecting along an axis, else None."""
    nonzero = np.flatnonzero(np.abs(direction) > 0)
    if nonzero.size != 1:
        return None
    return [d for d in range(direction.size) if d != nonzero[0]]


def get_faces_close_to_segment(g, seg, face_ix=None, fac: float = 2.0 / 3.0,
                               tol: float = TOL, proj_vec=None) -> np.ndarray:
    """Return faces that may lie near ``seg``; a superset, never a subset."""
    seg = np.atleast_2d(np.asarray(seg, dtype=float))
    coords = np.asarray(g.nodes.coords, dtype=float)
    if coords.shape[1] == 3 and seg.shape[1] == 2:
        seg = np.column_stack([seg, np.zeros(seg.shape[0])])
    if seg.shape[0] < 2:
        raise ValueError("a segment needs at least two points")
    if np.allclose(seg, seg[0]):
        raise ValueError("the segment has zero length")
    original = seg

    if getattr(g.faces, "bbox", None) is None:
        warnings.warn(
            "faces.bbox is missing; computing it now. Call "
            "add_bounding_box_fields once if this is in a loop.",
            stacklevel=2)
        g = add_bounding_box_fields(g)

    selection = slice(None) if face_ix is None else np.atleast_1d(
        np.asarray(face_ix, dtype=np.int64))
    bbox = np.asarray(g.faces.bbox, dtype=float)[selection]
    centroids = np.asarray(g.faces.centroids, dtype=float)[selection]

    if proj_vec is not None:
        direction = np.asarray(proj_vec, dtype=float).ravel()
        dims = _canonical(direction)
        if dims is not None:
            bbox, centroids, seg = bbox[:, dims], centroids[:, dims], seg[:, dims]
        else:
            direction = direction / np.linalg.norm(direction)
            # A box's extent under an arbitrary rotation is not its extent
            # along the axes; sqrt(3) bounds the discrepancy in 3-D.
            bbox = np.sqrt(3.0) * bbox
            centroids = centroids - np.outer(centroids @ direction, direction)
            seg = seg - np.outer(seg @ direction, direction)

    # More than two points: fit a line and widen the search by the scatter.
    radius = np.zeros(seg.shape[1])
    if seg.shape[0] > 2:
        middle = seg.mean(axis=0)
        relative = seg - middle
        along = np.linalg.svd(relative, full_matrices=False)[2][0]
        t = relative @ along
        radius = np.max(np.abs(relative - np.outer(t, along)), axis=0)
        seg = np.vstack([middle + t.min() * along, middle + t.max() * along])

    start = seg[0]
    v = seg[1] - seg[0]
    length = np.linalg.norm(v)
    if length == 0:
        raise ValueError(
            f"the segment collapses to a point when projected along "
            f"{np.asarray(proj_vec).tolist()}, so there is no direction left "
            f"to search along; it runs from {original[0].tolist()} to "
            f"{original[-1].tolist()}, which is parallel to the projection. "
            f"MRST divides by zero here and silently returns no faces.")

    offset = centroids - start
    t = (offset @ v) / (v @ v)
    # How far past each end a face's own box could still reach.
    slack = (bbox @ np.abs(fac * v)) / length
    near = np.flatnonzero((t > -slack / length - tol)
                          & (t < 1.0 + slack / length + tol))
    if near.size == 0:
        return np.zeros(0, dtype=np.int64)

    perpendicular = offset[near] - np.outer(t[near], v)
    distance = np.linalg.norm(perpendicular, axis=1)
    # Both sides are multiplied by `distance`, so this compares the distance
    # against the box's reach along that same direction - the unit vector is
    # implied rather than formed, which is why the left side is squared.
    reach = np.sum((fac * bbox[near] + radius) * np.abs(perpendicular), axis=1)
    near = near[distance ** 2 <= reach + tol]

    return (near.astype(np.int64) if face_ix is None
            else np.asarray(face_ix, dtype=np.int64)[near])


getFacesCloseToSegment = get_faces_close_to_segment


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(6, 6, 4), sizes=(6.0, 6.0, 4.0)):
        return add_bounding_box_fields(
            compute_geometry(cart_grid(list(dims), list(sizes))))

    def crossed_faces(g, seg, samples=4000):
        """Faces genuinely crossed, found by walking the segment.

        Sampling densely along the segment and recording which cell each point
        falls in gives the sequence of cells traversed; consecutive distinct
        cells must share the face that was crossed between them.
        """
        from pymrst.utils.gridtools.find_enclosing_cell import find_enclosing_cell
        points = (seg[0] + np.linspace(0, 1, samples)[:, None] * (seg[1] - seg[0]))
        cells = np.asarray(find_enclosing_cell(g, points), dtype=np.int64)
        cells = cells[cells >= 0]
        if cells.size == 0:
            return set()
        path = cells[np.concatenate([[True], np.diff(cells) != 0])]

        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
        crossed = set()
        for a, b in zip(path[:-1], path[1:]):
            shared = np.flatnonzero(
                ((neighbours[:, 0] == a) & (neighbours[:, 1] == b))
                | ((neighbours[:, 0] == b) & (neighbours[:, 1] == a)))
            crossed.update(shared.tolist())
        return crossed

    def test_it_misses_nothing_on_a_diagonal() -> None:
        """The property the filter exists to guarantee."""
        g = build()
        seg = np.array([[0.3, 0.4, 0.5], [5.7, 5.6, 3.5]])
        got = set(get_faces_close_to_segment(g, seg).tolist())
        missing = crossed_faces(g, seg) - got
        assert not missing, sorted(missing)
        print(f"ok test_it_misses_nothing_on_a_diagonal "
              f"({len(got)} returned, {len(crossed_faces(g, seg))} crossed)")

    def test_it_misses_nothing_on_an_axis_aligned_segment() -> None:
        g = build()
        for seg in (np.array([[0.5, 0.5, 0.5], [5.5, 0.5, 0.5]]),
                    np.array([[2.5, 0.2, 1.5], [2.5, 5.8, 1.5]]),
                    np.array([[1.5, 3.5, 0.2], [1.5, 3.5, 3.8]])):
            got = set(get_faces_close_to_segment(g, seg).tolist())
            assert not (crossed_faces(g, seg) - got), seg
        print("ok test_it_misses_nothing_on_an_axis_aligned_segment")

    def test_it_actually_filters() -> None:
        """A superset is only useful if it is much smaller than everything."""
        g = build()
        seg = np.array([[0.5, 0.5, 0.5], [5.5, 0.5, 0.5]])
        got = get_faces_close_to_segment(g, seg)
        assert got.size < g.faces.num / 4, (got.size, g.faces.num)
        print(f"ok test_it_actually_filters ({got.size} of {g.faces.num})")

    def test_a_far_away_segment_returns_nothing() -> None:
        g = build()
        got = get_faces_close_to_segment(
            g, np.array([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]]))
        assert got.size == 0, got
        print("ok test_a_far_away_segment_returns_nothing")

    def test_a_face_subset_is_honoured() -> None:
        g = build()
        seg = np.array([[0.5, 0.5, 0.5], [5.5, 0.5, 0.5]])
        everything = get_faces_close_to_segment(g, seg)
        subset = np.arange(0, g.faces.num, 3)
        got = get_faces_close_to_segment(g, seg, face_ix=subset)
        assert set(got.tolist()) == set(everything.tolist()) & set(subset.tolist())
        print("ok test_a_face_subset_is_honoured")

    def test_the_returned_indices_are_global_when_a_subset_is_given() -> None:
        """The trap: they index the grid, not the subset."""
        g = build()
        subset = np.array([10, 20, 30, 40, 50])
        got = get_faces_close_to_segment(
            g, np.array([[0.5, 0.5, 0.5], [5.5, 0.5, 0.5]]), face_ix=subset)
        assert set(got.tolist()) <= set(subset.tolist()), got
        print("ok test_the_returned_indices_are_global_when_a_subset_is_given")

    def test_a_polyline_widens_the_search() -> None:
        """Extra points off the line grow the radius, so more faces qualify."""
        g = build()
        straight = np.array([[0.5, 3.0, 2.0], [5.5, 3.0, 2.0]])
        bent = np.array([[0.5, 3.0, 2.0], [3.0, 4.2, 2.0], [5.5, 3.0, 2.0]])
        assert (get_faces_close_to_segment(g, bent).size
                > get_faces_close_to_segment(g, straight).size)
        print("ok test_a_polyline_widens_the_search")

    def test_a_larger_factor_returns_more() -> None:
        g = build()
        seg = np.array([[0.5, 0.5, 0.5], [5.5, 0.5, 0.5]])
        small = get_faces_close_to_segment(g, seg, fac=0.5)
        large = get_faces_close_to_segment(g, seg, fac=2.0)
        assert set(small.tolist()) <= set(large.tolist())
        assert large.size > small.size
        print("ok test_a_larger_factor_returns_more")

    def test_projection_along_z_ignores_depth() -> None:
        """Seen from above, a deviated well's depth stops mattering, so faces
        far below the segment's own z range still qualify."""
        g = build()
        seg = np.array([[1.5, 1.5, 0.2], [4.5, 4.5, 0.4]])
        plain = get_faces_close_to_segment(g, seg)
        projected = get_faces_close_to_segment(g, seg, proj_vec=[0.0, 0.0, 1.0])
        assert projected.size > plain.size, (projected.size, plain.size)
        centroids = np.asarray(g.faces.centroids)
        assert centroids[plain, 2].max() < 1.5
        assert centroids[projected, 2].max() > 3.0, "the column should qualify"
        print("ok test_projection_along_z_ignores_depth")

    def test_projecting_a_segment_along_itself_is_rejected() -> None:
        """MRST divides by zero here and quietly returns nothing, which reads
        as "no faces are near" rather than "the question was malformed"."""
        g = build()
        vertical = np.array([[2.5, 2.5, 0.1], [2.5, 2.5, 3.9]])
        try:
            get_faces_close_to_segment(g, vertical, proj_vec=[0.0, 0.0, 1.0])
        except ValueError as error:
            assert "collapses to a point" in str(error)
            print("ok test_projecting_a_segment_along_itself_is_rejected")
            return
        raise AssertionError("a segment parallel to the projection must be "
                             "rejected")

    def test_a_general_projection_direction_works() -> None:
        g = build()
        seg = np.array([[2.5, 2.5, 0.5], [2.5, 2.5, 3.5]])
        got = get_faces_close_to_segment(g, seg, proj_vec=[1.0, 1.0, 1.0])
        assert got.size > 0
        print("ok test_a_general_projection_direction_works")

    def test_it_works_in_two_dimensions() -> None:
        g = add_bounding_box_fields(
            compute_geometry(cart_grid([6, 6], [6.0, 6.0])))
        seg = np.array([[0.5, 0.5], [5.5, 5.5]])
        got = get_faces_close_to_segment(g, seg)
        assert 0 < got.size < g.faces.num
        print("ok test_it_works_in_two_dimensions")

    def test_a_missing_bbox_is_computed_with_a_warning() -> None:
        g = compute_geometry(cart_grid([4, 4], [4.0, 4.0]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = get_faces_close_to_segment(g, np.array([[0.5, 0.5],
                                                          [3.5, 3.5]]))
        assert got.size > 0
        assert any("bbox" in str(w.message) for w in caught), caught
        print("ok test_a_missing_bbox_is_computed_with_a_warning")

    def test_a_zero_length_segment_is_rejected() -> None:
        g = build()
        try:
            get_faces_close_to_segment(g, np.array([[1.0, 1.0, 1.0],
                                                    [1.0, 1.0, 1.0]]))
        except ValueError as error:
            assert "zero length" in str(error)
            print("ok test_a_zero_length_segment_is_rejected")
            return
        raise AssertionError("a degenerate segment must be rejected")

    def test_a_single_point_is_rejected() -> None:
        try:
            get_faces_close_to_segment(build(), np.array([[1.0, 1.0, 1.0]]))
        except ValueError as error:
            assert "at least two points" in str(error)
            print("ok test_a_single_point_is_rejected")
            return
        raise AssertionError("a one-point segment must be rejected")

    test_it_misses_nothing_on_a_diagonal()
    test_it_misses_nothing_on_an_axis_aligned_segment()
    test_it_actually_filters()
    test_a_far_away_segment_returns_nothing()
    test_a_face_subset_is_honoured()
    test_the_returned_indices_are_global_when_a_subset_is_given()
    test_a_polyline_widens_the_search()
    test_a_larger_factor_returns_more()
    test_projection_along_z_ignores_depth()
    test_projecting_a_segment_along_itself_is_rejected()
    test_a_general_projection_direction_works()
    test_it_works_in_two_dimensions()
    test_a_missing_bbox_is_computed_with_a_warning()
    test_a_zero_length_segment_is_rejected()
    test_a_single_point_is_rejected()
    print("\nAll tests passed!")
