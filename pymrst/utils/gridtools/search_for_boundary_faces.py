"""Boundary faces in an index box (MRST ``utils/gridtools/searchForBoundaryFaces.m``).

Boundary faces on one named side of a logically Cartesian grid, restricted to a
box of logical indices.

**Not a duplicate of `boundary_face_indices`, though the two are easy to
confuse.** That one takes the two ranges *in the plane of the side*, so which
axes `i1` and `i2` refer to depends on which side you asked for. This one takes
`i1`, `i2`, `i3` as the `I`, `J` and `K` ranges always, whatever the side.
Asking for the west faces of a 3-D grid with `i3=[0]` selects the top layer's
west faces here; the same call to `boundary_face_indices` means something else.
Both exist in MRST, and this keeps them distinct rather than folding one into
the other.

**Ranges are 0-based**, like every other index in PyMRST and unlike MRST's.
`i3=[0]` is the first layer.

The side names follow MRST's six alias groups, and one of them is a trap worth
saying out loud: **`top` means `ZMin` and `bottom` means `ZMax`**. That is the
depth convention - z increases downwards - so the "top" of the reservoir is at
the small end of z. `upper`/`lower` are the same two groups.

Restriction is by the cell's logical position through `cells.indexMap`, so it
stays correct on a grid with inactive cells removed, where the cell number and
the logical index have parted company.
"""

import numpy as np

from pymrst.utils.gridtools.boundary_face_indices import SIDE_TAG


def search_for_boundary_faces(g, direction: str, i1=None, i2=None,
                              i3=None) -> np.ndarray:
    """Return boundary faces on ``direction``, inside the ``i1/i2/i3`` box."""
    key = direction.strip().lower()
    if key not in SIDE_TAG:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of "
            f"{sorted(set(SIDE_TAG))}")
    tag = SIDE_TAG[key]

    if g.cartDims is None or getattr(g.cells, "tags", None) is None:
        raise ValueError(
            "search_for_boundary_faces needs a logically Cartesian grid "
            "(cartDims and cells.tags)")

    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    index_map = np.asarray(g.cells.indexMap, dtype=np.int64)
    # MATLAB's ind2sub on cartDims is column-major: the first axis runs fastest.
    ijk = np.unravel_index(index_map, dims, order="F")

    inside = np.ones(index_map.size, dtype=bool)
    for axis, wanted in enumerate((i1, i2, i3)):
        if wanted is None:
            continue
        pick = np.zeros(dims[axis], dtype=bool)
        wanted = np.atleast_1d(np.asarray(wanted, dtype=np.int64))
        if wanted.size and (wanted.min() < 0 or wanted.max() >= dims[axis]):
            raise ValueError(
                f"index range for axis {axis} runs outside 0..{dims[axis] - 1}; "
                f"note PyMRST indexes from 0 where MRST indexes from 1")
        pick[wanted] = True
        inside &= pick[ijk[axis]]

    faces = np.asarray(g.cells.faces, dtype=np.int64)
    faces = faces[:, 0] if faces.ndim > 1 else faces
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(
        np.asarray(g.cells.facePos, dtype=np.int64)))
    boundary = np.any(np.asarray(g.faces.neighbors, dtype=np.int64) < 0, axis=1)

    chosen = (boundary[faces] & (np.asarray(g.cells.tags) == tag)
              & inside[cell_no])
    return faces[chosen]


searchForBoundaryFaces = search_for_boundary_faces


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.boundary_face_indices import boundary_face_indices

    def build(dims=(4, 3, 2), sizes=(4.0, 3.0, 2.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_each_side_gives_the_expected_count() -> None:
        g = build()
        for side, expected in (("west", 3 * 2), ("east", 3 * 2),
                               ("south", 4 * 2), ("north", 4 * 2),
                               ("top", 4 * 3), ("bottom", 4 * 3)):
            got = search_for_boundary_faces(g, side)
            assert got.size == expected, (side, got.size, expected)
        print("ok test_each_side_gives_the_expected_count")

    def test_the_faces_really_are_on_the_boundary() -> None:
        g = build()
        neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
        for side in ("west", "east", "south", "north", "top", "bottom"):
            faces = search_for_boundary_faces(g, side)
            assert np.all(np.any(neighbours[faces] < 0, axis=1)), side
        print("ok test_the_faces_really_are_on_the_boundary")

    def test_west_faces_sit_at_the_low_x_end() -> None:
        g = build()
        centroids = np.asarray(g.faces.centroids, dtype=float)
        west = search_for_boundary_faces(g, "west")
        east = search_for_boundary_faces(g, "east")
        assert np.allclose(centroids[west, 0], 0.0)
        assert np.allclose(centroids[east, 0], 4.0)
        print("ok test_west_faces_sit_at_the_low_x_end")

    def test_top_is_zmin_and_bottom_is_zmax() -> None:
        """The depth convention, which is the easiest thing here to get wrong."""
        g = build()
        centroids = np.asarray(g.faces.centroids, dtype=float)
        assert np.allclose(centroids[search_for_boundary_faces(g, "top"), 2], 0.0)
        assert np.allclose(centroids[search_for_boundary_faces(g, "bottom"), 2], 2.0)
        assert np.array_equal(search_for_boundary_faces(g, "top"),
                              search_for_boundary_faces(g, "zmin"))
        print("ok test_top_is_zmin_and_bottom_is_zmax")

    def test_the_aliases_agree() -> None:
        g = build()
        for group in (("west", "xmin", "left"), ("east", "xmax", "right"),
                      ("south", "ymin", "back"), ("north", "ymax", "front"),
                      ("top", "zmin", "upper"), ("bottom", "zmax", "lower")):
            first = search_for_boundary_faces(g, group[0])
            for alias in group[1:]:
                assert np.array_equal(search_for_boundary_faces(g, alias), first)
        print("ok test_the_aliases_agree")

    def test_a_layer_restriction_selects_that_layer() -> None:
        g = build()
        centroids = np.asarray(g.faces.centroids, dtype=float)
        top_layer = search_for_boundary_faces(g, "west", i3=[0])
        assert top_layer.size == 3, top_layer.size
        assert np.all(centroids[top_layer, 2] < 1.0)
        print("ok test_a_layer_restriction_selects_that_layer")

    def test_the_restrictions_are_axial_not_in_plane() -> None:
        """The distinction from `boundary_face_indices`; see the docstring.

        `i1` restricts `I` even when asking for a west face, where every
        selected face already has `I = 0`. Restricting to `I = 1` therefore
        gives nothing - which is the point: the ranges are not in-plane.
        """
        g = build()
        assert search_for_boundary_faces(g, "west", i1=[0]).size == 6
        assert search_for_boundary_faces(g, "west", i1=[1]).size == 0
        print("ok test_the_restrictions_are_axial_not_in_plane")

    def test_it_agrees_with_boundary_face_indices_unrestricted() -> None:
        """Different interfaces, same faces when nothing is restricted."""
        g = build()
        for side in ("west", "east", "south", "north", "top", "bottom"):
            assert np.array_equal(np.sort(search_for_boundary_faces(g, side)),
                                  np.sort(boundary_face_indices(g, side))), side
        print("ok test_it_agrees_with_boundary_face_indices_unrestricted")

    def test_combined_restrictions_intersect() -> None:
        g = build()
        got = search_for_boundary_faces(g, "top", i1=[0, 1], i2=[2])
        assert got.size == 2, got.size
        print("ok test_combined_restrictions_intersect")

    def test_it_works_in_two_dimensions() -> None:
        g = build((4, 3), (4.0, 3.0))
        assert search_for_boundary_faces(g, "west").size == 3
        assert search_for_boundary_faces(g, "north").size == 4
        print("ok test_it_works_in_two_dimensions")

    def test_it_follows_index_map_on_a_grid_with_holes() -> None:
        """Restriction is by logical position, so removing cells must not
        shift which layer `i3=[0]` means."""
        from pymrst.gridprocessing import remove_cells
        g = build()
        holed = compute_geometry(remove_cells(build(), np.array([5, 6])))
        centroids = np.asarray(holed.faces.centroids, dtype=float)
        faces = search_for_boundary_faces(holed, "west", i3=[0])
        assert faces.size and np.all(centroids[faces, 2] < 1.0)
        print("ok test_it_follows_index_map_on_a_grid_with_holes")

    def test_an_unknown_direction_is_rejected() -> None:
        try:
            search_for_boundary_faces(build(), "sideways")
        except ValueError as error:
            assert "unknown direction" in str(error)
            print("ok test_an_unknown_direction_is_rejected")
            return
        raise AssertionError("an unknown direction must be rejected")

    def test_an_out_of_range_index_is_rejected() -> None:
        """A 1-based index passed by habit is the likely cause."""
        try:
            search_for_boundary_faces(build(), "west", i3=[2])
        except ValueError as error:
            assert "MRST indexes from 1" in str(error)
            print("ok test_an_out_of_range_index_is_rejected")
            return
        raise AssertionError("an out-of-range index must be rejected")

    test_each_side_gives_the_expected_count()
    test_the_faces_really_are_on_the_boundary()
    test_west_faces_sit_at_the_low_x_end()
    test_top_is_zmin_and_bottom_is_zmax()
    test_the_aliases_agree()
    test_a_layer_restriction_selects_that_layer()
    test_the_restrictions_are_axial_not_in_plane()
    test_it_agrees_with_boundary_face_indices_unrestricted()
    test_combined_restrictions_intersect()
    test_it_works_in_two_dimensions()
    test_it_follows_index_map_on_a_grid_with_holes()
    test_an_unknown_direction_is_rejected()
    test_an_out_of_range_index_is_rejected()
    print("\nAll tests passed!")
