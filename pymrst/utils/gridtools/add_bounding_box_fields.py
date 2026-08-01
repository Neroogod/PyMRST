"""Per-cell and per-face bounding boxes (MRST ``utils/gridtools/addBoundingBoxFields.m``).

A cheap rejection test. Before running real geometry against every face - is
this segment near it, does this cell contain that point - you can throw out
almost all candidates by comparing boxes, which costs a few subtractions each.

**What is stored is the box's *size*, not the box.** `faces.bbox[f]` holds
`max - min` of the face's node coordinates along each axis, so it says how big
the face is but not where it is. MRST's naming invites the other reading, and
the distinction matters: it takes a second quantity to place the box.

MRST's own consumer, `getFacesCloseToSegment`, supplies that second quantity as
`faces.centroids` and treats the face as spanning `centroid ± bbox/2`. **That
is an approximation**, because a centroid is not a box centre - for an L-shaped
or strongly skewed face the true box is off to one side, and the reconstructed
box can miss part of it. It is the right kind of approximation for a rejection
test *if* it errs outward, and it does not always. Recorded here rather than
in the caller, because the caller inherits the problem from this
representation.

`extent=False` returns the honest `(minimum, maximum)` pair instead, which is
what most callers want and what makes an overlap test exact. The MRST-shaped
`bbox` field is still written either way, so a port of `getFacesCloseToSegment`
finds what it expects.
"""

import copy

import numpy as np

from pymrst.utils.gridtools.grid_cell_nodes import grid_cell_nodes


def _blocks(coords, index, pos):
    """Minimum and maximum of ``coords[index]`` over each packed block."""
    if np.any(np.diff(pos) < 1):
        raise ValueError(
            "a bounding box needs at least one node per entry; the grid has "
            "an empty block in nodePos/facePos")
    taken = coords[index]
    starts = np.asarray(pos[:-1], dtype=np.int64)
    return (np.minimum.reduceat(taken, starts, axis=0),
            np.maximum.reduceat(taken, starts, axis=0))


def add_bounding_box_fields(g, cells: bool = False, faces: bool = True,
                            extent: bool = True):
    """Return a copy of ``g`` with ``bbox`` on its faces and/or cells.

    ``extent=True`` (MRST's behaviour) stores the box size, ``max - min``;
    ``extent=False`` also stores ``bboxMin`` and ``bboxMax``. See the module
    docstring - the two are not interchangeable.
    """
    g = copy.deepcopy(g)
    coords = np.asarray(g.nodes.coords, dtype=float)

    def attach(holder, index, pos):
        low, high = _blocks(coords, index, pos)
        holder.bbox = high - low
        if not extent:
            holder.bboxMin, holder.bboxMax = low, high

    if faces:
        attach(g.faces, np.asarray(g.faces.nodes, dtype=np.int64),
               np.asarray(g.faces.nodePos, dtype=np.int64))
    if cells:
        # cellNodes, not cells.faces: a cell's box is over its nodes.
        nodes, pos = grid_cell_nodes(g)
        attach(g.cells, np.asarray(nodes, dtype=np.int64),
               np.asarray(pos, dtype=np.int64))
    return g


addBoundingBoxFields = add_bounding_box_fields


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(3, 2, 2), sizes=(3.0, 2.0, 2.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_faces_are_boxed_by_default_and_cells_are_not() -> None:
        g = add_bounding_box_fields(build())
        assert getattr(g.faces, "bbox", None) is not None
        assert getattr(g.cells, "bbox", None) is None
        print("ok test_faces_are_boxed_by_default_and_cells_are_not")

    def test_a_unit_cube_cell_has_unit_extent() -> None:
        g = add_bounding_box_fields(build(), cells=True)
        assert np.allclose(g.cells.bbox, 1.0), g.cells.bbox
        print("ok test_a_unit_cube_cell_has_unit_extent")

    def test_a_cartesian_face_is_flat_along_its_own_axis() -> None:
        """Each face spans zero in exactly one direction - that is what makes
        it a face rather than a cell."""
        g = add_bounding_box_fields(build())
        flat = np.count_nonzero(np.asarray(g.faces.bbox) == 0.0, axis=1)
        assert np.all(flat == 1), np.unique(flat)
        print("ok test_a_cartesian_face_is_flat_along_its_own_axis")

    def test_the_box_really_contains_the_nodes() -> None:
        """The defining property, checked against the nodes themselves."""
        g = add_bounding_box_fields(build(), cells=True, extent=False)
        coords = np.asarray(g.nodes.coords, dtype=float)
        pos = np.asarray(g.faces.nodePos, dtype=np.int64)
        nodes = np.asarray(g.faces.nodes, dtype=np.int64)
        for f in range(g.faces.num):
            own = coords[nodes[pos[f]:pos[f + 1]]]
            assert np.all(own >= g.faces.bboxMin[f] - 1e-12)
            assert np.all(own <= g.faces.bboxMax[f] + 1e-12)
        print("ok test_the_box_really_contains_the_nodes")

    def test_the_extent_is_the_difference_of_the_bounds() -> None:
        g = add_bounding_box_fields(build(), cells=True, extent=False)
        for holder in (g.faces, g.cells):
            assert np.allclose(holder.bbox, holder.bboxMax - holder.bboxMin)
        print("ok test_the_extent_is_the_difference_of_the_bounds")

    def test_the_extent_alone_does_not_locate_the_box() -> None:
        """Why `extent=False` exists: two grids far apart share every extent."""
        near = add_bounding_box_fields(build())
        far = build()
        far.nodes.coords = np.asarray(far.nodes.coords, dtype=float) + 1000.0
        far = add_bounding_box_fields(compute_geometry(far))
        assert np.allclose(near.faces.bbox, far.faces.bbox)
        print("ok test_the_extent_alone_does_not_locate_the_box")

    def test_centroid_plus_half_extent_can_miss_the_true_box() -> None:
        """The approximation MRST's consumer relies on; see the docstring.

        A triangle is the simplest witness: its centroid sits a third of the
        way up, so the box built around it does not cover the triangle.
        """
        low = np.array([0.0, 0.0])
        high = np.array([3.0, 3.0])
        corners = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        centroid = corners.mean(axis=0)
        bbox = high - low
        assert np.any(centroid + bbox / 2 < high - 1e-12), \
            "the reconstructed box was expected to fall short"
        print("ok test_centroid_plus_half_extent_can_miss_the_true_box")

    def test_it_works_in_two_dimensions() -> None:
        g = add_bounding_box_fields(build((3, 3), (3.0, 3.0)), cells=True)
        assert np.asarray(g.faces.bbox).shape == (g.faces.num, 2)
        assert np.allclose(g.cells.bbox, 1.0)
        print("ok test_it_works_in_two_dimensions")

    def test_a_stretched_grid_gives_the_stretched_extent() -> None:
        g = add_bounding_box_fields(build((2, 2, 2), (4.0, 6.0, 10.0)),
                                    cells=True)
        assert np.allclose(g.cells.bbox, [2.0, 3.0, 5.0]), g.cells.bbox[0]
        print("ok test_a_stretched_grid_gives_the_stretched_extent")

    def test_the_input_is_not_modified() -> None:
        g = build()
        add_bounding_box_fields(g, cells=True)
        assert getattr(g.faces, "bbox", None) is None
        print("ok test_the_input_is_not_modified")

    def test_it_survives_an_unstructured_grid() -> None:
        """Faces with differing node counts - the packed-block path."""
        from pymrst.gridprocessing import triangle_grid
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        g = add_bounding_box_fields(
            compute_geometry(triangle_grid(points, triangles)),
            cells=True, extent=False)
        assert np.all(g.cells.bbox >= 0) and np.all(g.faces.bbox >= 0)
        assert np.allclose(g.cells.bboxMax - g.cells.bboxMin, g.cells.bbox)
        print("ok test_it_survives_an_unstructured_grid")

    test_faces_are_boxed_by_default_and_cells_are_not()
    test_a_unit_cube_cell_has_unit_extent()
    test_a_cartesian_face_is_flat_along_its_own_axis()
    test_the_box_really_contains_the_nodes()
    test_the_extent_is_the_difference_of_the_bounds()
    test_the_extent_alone_does_not_locate_the_box()
    test_centroid_plus_half_extent_can_miss_the_true_box()
    test_it_works_in_two_dimensions()
    test_a_stretched_grid_gives_the_stretched_extent()
    test_the_input_is_not_modified()
    test_it_survives_an_unstructured_grid()
    print("\nAll tests passed!")
