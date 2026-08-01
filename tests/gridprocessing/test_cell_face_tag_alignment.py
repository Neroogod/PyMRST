"""Cell-face tags must stay aligned with `cells.faces`.

`cells.tags` is parallel to `cells.faces`: entry i says which side of the cell
half-face i is on. Any operation that reorders or rebuilds `cells.faces` has to
carry the tags with it, or the two silently desynchronise and every routine
that selects a boundary by side - `pside`, `boundary_face_indices`, and through
them the whole vemmech boundary-condition stack - starts returning the wrong
faces.

This is a regression test for two real defects: `_sort_cell_faces` permuting
faces without their tags, and `triangulate_faces` dropping them entirely. Both
produced valid-looking grids and plausible-looking answers.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.pebi import _sort_cell_faces
from pymrst.gridprocessing.triangulate_faces import triangulate_faces
from pymrst.utils.gridtools import boundary_face_indices

SIDES_2D = ("XMin", "XMax", "YMin", "YMax")
SIDES_3D = ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax")


def _side_position(grid, side: str):
    """Mean coordinate of a side, along the axis that side is normal to."""
    faces = boundary_face_indices(grid, side)
    axis = SIDES_3D.index(side) // 2
    return faces.size, float(grid.faces.centroids[faces][:, axis].mean())


def _assert_sides_are_where_they_claim(grid, sides):
    """Every named side must lie at the extreme of its own axis."""
    coords = np.asarray(grid.nodes.coords, dtype=float)
    for side in sides:
        axis = SIDES_3D.index(side) // 2
        count, position = _side_position(grid, side)
        assert count > 0, f"{side} is empty"
        expected = coords[:, axis].min() if side.endswith("Min") \
            else coords[:, axis].max()
        assert np.isclose(position, expected), (
            f"{side} sits at {position}, expected {expected}"
        )


@pytest.mark.parametrize("dims,size,sides", [
    ([6, 6], [15.0, 15.0], SIDES_2D),
    ([4, 4, 4], [1.0, 1.0, 1.0], SIDES_3D),
])
def test_plain_grid_sides_are_correct(dims, size, sides):
    _assert_sides_are_where_they_claim(compute_geometry(cart_grid(dims, size)),
                                       sides)


def test_sort_cell_faces_preserves_tag_alignment():
    """The defect: sorting permuted cells.faces and left cells.tags behind."""
    grid = compute_geometry(cart_grid([6, 6], [15.0, 15.0]))
    before = {side: _side_position(grid, side) for side in SIDES_2D}
    grid = compute_geometry(_sort_cell_faces(grid))
    after = {side: _side_position(grid, side) for side in SIDES_2D}
    assert before == after, f"sorting moved the sides: {before} -> {after}"
    _assert_sides_are_where_they_claim(grid, SIDES_2D)


def test_sort_cell_faces_preserves_alignment_after_triangulation():
    """The combination that actually broke: split some faces, then sort."""
    grid = compute_geometry(cart_grid([6, 6], [15.0, 15.0]))
    tags = np.asarray(grid.cells.tags, dtype=np.int64)
    horizontal = np.unique(
        np.asarray(grid.cells.faces, dtype=np.int64)[np.isin(tags, (2, 3))])
    grid = compute_geometry(triangulate_faces(grid, horizontal))
    grid = compute_geometry(_sort_cell_faces(grid))
    _assert_sides_are_where_they_claim(grid, SIDES_2D)
    # the split sides carry twice the faces, the untouched ones are unchanged
    assert _side_position(grid, "YMin")[0] == 2 * _side_position(grid, "XMin")[0]


@pytest.mark.parametrize("dims,size,sides", [
    ([4, 4], [1.0, 1.0], SIDES_2D),
    ([3, 3, 3], [1.0, 1.0, 1.0], SIDES_3D),
])
def test_triangulation_keeps_every_side_populated(dims, size, sides):
    """The other defect: triangulation zero-filled the tags, emptying pside."""
    grid = compute_geometry(cart_grid(dims, size))
    counts = {side: boundary_face_indices(grid, side).size for side in sides}
    grid = compute_geometry(triangulate_faces(grid, np.arange(grid.faces.num)))
    _assert_sides_are_where_they_claim(grid, sides)
    for side in sides:
        after = boundary_face_indices(grid, side).size
        assert after > 0, f"{side} lost all its faces"
        assert after % counts[side] == 0


def test_tags_stay_the_same_length_as_cell_faces():
    grid = compute_geometry(cart_grid([4, 4], [1.0, 1.0]))
    for step in (lambda g: triangulate_faces(g, np.arange(g.faces.num)),
                 _sort_cell_faces):
        grid = compute_geometry(step(grid))
        assert grid.cells.tags is not None
        assert np.asarray(grid.cells.tags).size == np.asarray(grid.cells.faces).size
