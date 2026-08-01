"""Faces on the boundary of a cell subset (MRST ``plotting/boundaryFaces.m``).

A face is on the boundary of a subset when exactly one of its two neighbours is
in the subset. With no subset given the subset is every cell, so the result is
the grid's outer boundary.

Note the definition is *exclusive or*, not "has a neighbour outside": an
interior face between two selected cells is not a boundary face, and neither is
one between two unselected cells. That is what makes the function work for an
arbitrary subset and not only for the whole grid.

**MRST files this under `plotting/`; this port does not**, and the reason is
concrete rather than aesthetic. `pymrst/plotting/__init__.py` imports the
PyVista and matplotlib stack, so a counterpart placed there would drag a
rendering toolchain into `conserve_flux` - a numerical routine that draws
nothing. It sits beside `rldecode`, `mcolon` and `boundary_face_indices`
instead, which is what it is: a grid query.

This is the one place in the `vem` port where MRST's layout is deliberately
not followed. The parity ledger keys on the filename, so `boundaryFaces.m`
still resolves to `boundary_faces.py`.
"""

from typing import Optional, Tuple, Union

import numpy as np


def boundary_faces(g, subset=None, with_cells: bool = False):
    """Faces bounding ``subset``; also the owning cell when ``with_cells``."""
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)

    present = np.zeros(g.cells.num, dtype=bool)
    if subset is None:
        present[:] = True
    else:
        subset = np.atleast_1d(np.asarray(subset))
        if subset.dtype == bool:
            if subset.size != g.cells.num:
                raise ValueError(
                    f"logical subset has {subset.size} entries but the grid "
                    f"has {g.cells.num} cells")
            present = subset.copy()
        else:
            present[subset.astype(np.int64)] = True

    # A neighbour index of -1 marks the outside, which is never "present".
    def inside(column):
        return (column >= 0) & present[np.maximum(column, 0)]

    left, right = inside(neighbours[:, 0]), inside(neighbours[:, 1])
    faces = np.flatnonzero(left ^ right)
    if not with_cells:
        return faces

    # The owning cell is whichever side is in the subset.
    owners = np.where(left[faces], neighbours[faces, 0], neighbours[faces, 1])
    return faces, owners


boundaryFaces = boundary_faces


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_the_whole_grid_gives_its_outer_boundary() -> None:
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        faces = boundary_faces(g)
        outer = np.flatnonzero(
            np.any(np.asarray(g.faces.neighbors) < 0, axis=1))
        assert np.array_equal(faces, outer), (faces, outer)
        print("ok test_the_whole_grid_gives_its_outer_boundary")

    def test_a_two_by_two_grid_has_eight_boundary_faces() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        assert boundary_faces(g).size == 8, boundary_faces(g).size
        print("ok test_a_two_by_two_grid_has_eight_boundary_faces")

    def test_a_subset_boundary_includes_interior_faces() -> None:
        """The point of the subset form: the cut runs through the grid."""
        g = compute_geometry(cart_grid([3, 1], [3.0, 1.0]))
        faces = boundary_faces(g, [0])
        assert faces.size == 4, faces.size          # 3 outer + 1 cut
        interior = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
        assert interior[faces].sum() == 1, "one face cuts into the grid"
        print("ok test_a_subset_boundary_includes_interior_faces")

    def test_a_face_between_two_selected_cells_is_not_a_boundary() -> None:
        g = compute_geometry(cart_grid([3, 1], [3.0, 1.0]))
        faces = boundary_faces(g, [0, 1])
        interior = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
        assert interior[faces].sum() == 1, "only the cut, not the shared face"
        print("ok test_a_face_between_two_selected_cells_is_not_a_boundary")

    def test_the_empty_subset_has_no_boundary() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        assert boundary_faces(g, []).size == 0
        print("ok test_the_empty_subset_has_no_boundary")

    def test_a_logical_subset_is_accepted() -> None:
        g = compute_geometry(cart_grid([3, 1], [3.0, 1.0]))
        mask = np.zeros(g.cells.num, dtype=bool)
        mask[0] = True
        assert np.array_equal(boundary_faces(g, mask), boundary_faces(g, [0]))
        print("ok test_a_logical_subset_is_accepted")

    def test_the_owning_cell_is_the_one_inside_the_subset() -> None:
        g = compute_geometry(cart_grid([3, 1], [3.0, 1.0]))
        faces, owners = boundary_faces(g, [1], with_cells=True)
        assert np.all(owners == 1), owners
        print("ok test_the_owning_cell_is_the_one_inside_the_subset")

    def test_a_mis_sized_logical_subset_is_rejected() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        try:
            boundary_faces(g, np.ones(3, dtype=bool))
        except ValueError as error:
            assert "4 cells" in str(error)
            print("ok test_a_mis_sized_logical_subset_is_rejected")
            return
        raise AssertionError("a mis-sized mask must be rejected")

    def test_it_works_in_three_dimensions() -> None:
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        assert boundary_faces(g).size == 24, boundary_faces(g).size
        print("ok test_it_works_in_three_dimensions")

    test_the_whole_grid_gives_its_outer_boundary()
    test_a_two_by_two_grid_has_eight_boundary_faces()
    test_a_subset_boundary_includes_interior_faces()
    test_a_face_between_two_selected_cells_is_not_a_boundary()
    test_the_empty_subset_has_no_boundary()
    test_a_logical_subset_is_accepted()
    test_the_owning_cell_is_the_one_inside_the_subset()
    test_a_mis_sized_logical_subset_is_rejected()
    test_it_works_in_three_dimensions()
    print("\nAll tests passed!")
