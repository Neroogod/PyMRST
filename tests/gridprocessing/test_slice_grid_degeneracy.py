"""sliceGrid where the cut plane coincides with a face.

Two cases MRST handles through its `topologicalCellSplit` path: a pinched cell
whose collapsed face lies in the plane (still split), and a cut snapped onto an
existing column of faces (nothing split, but the faces are the slice).
"""

import warnings

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid

PLANE_POINT = np.array([0.5, 0.5, 0.5])
PLANE_NORMAL = np.array([1.0, 0.0, 0.0])


def _pinched_cube():
    """A unit cube whose top face is collapsed onto the plane x=0.5.

    The plane then contains that whole face while still passing through the
    cell interior - the case that used to raise NotImplementedError.
    """
    grid = cart_grid([1, 1, 1], [1.0, 1.0, 1.0])
    grid.nodes.coords[grid.nodes.coords[:, 2] > 0.5, 0] = 0.5
    with warnings.catch_warnings():                      # the collapsed face
        warnings.simplefilter("ignore")
        return compute_geometry(grid)


def _snapped_cut(grid, **kwargs):
    return slice_grid(grid, np.array([[1.0, -1.0, 0.0], [1.0, 2.0, 0.0]]),
                      cut_dir=np.array([0.0, 0.0, 1.0]), **kwargs)


def _outward_normal_residual(grid):
    """max |sum of outward face normals| over cells - zero for closed cells."""
    cell_no = grid.cell_no()
    sign = np.where(grid.faces.neighbors[grid.cells.faces, 0] == cell_no, 1.0, -1.0)
    residual = np.zeros((grid.cells.num, 3))
    np.add.at(residual, cell_no, sign[:, None] * grid.faces.normals[grid.cells.faces])
    return np.abs(residual).max()


def test_plane_containing_a_collapsed_face_splits_the_cell():
    grid = _pinched_cube()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut = compute_geometry(
            slice_grid(grid, PLANE_POINT, normal=PLANE_NORMAL))
    assert cut.cells.num == 2
    assert np.allclose(cut.cells.volumes, 0.25)
    assert np.isclose(cut.cells.volumes.sum(), grid.cells.volumes.sum())


def test_split_of_a_pinched_cell_is_a_closed_valid_grid():
    grid = _pinched_cube()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut = compute_geometry(
            slice_grid(grid, PLANE_POINT, normal=PLANE_NORMAL))
    assert np.all(cut.cells.volumes > 1e-12)
    assert _outward_normal_residual(cut) < 1e-12
    cell_no = cut.cell_no()
    incidence = set(zip(cell_no.tolist(), np.asarray(cut.cells.faces).tolist()))
    for face in range(cut.faces.num):
        for cell in cut.faces.neighbors[face]:
            if cell >= 0:
                assert (int(cell), face) in incidence


def test_pinched_cut_inserts_exactly_one_slice_face():
    grid = _pinched_cube()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, gix = slice_grid(grid, PLANE_POINT, normal=PLANE_NORMAL,
                            return_mapping=True)
    assert np.count_nonzero(gix.new.faces == 3) == 1


def test_snapped_cut_marks_the_coincident_face_as_the_slice():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    cut, gix = _snapped_cut(grid, return_mapping=True)
    assert cut.cells.num == 2                            # nothing is split
    slice_faces = np.flatnonzero(gix.new.faces == 3)
    assert slice_faces.size == 1
    cut = compute_geometry(cut)
    assert np.allclose(cut.faces.centroids[slice_faces[0]], [1.0, 0.5, 0.5])


def test_snapped_cut_returns_the_lower_dimensional_slice_grid():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    _, cross_section = _snapped_cut(grid, return_slice=True)
    assert cross_section is not None
    assert cross_section.cells.num == 1
    assert np.allclose(np.unique(cross_section.nodes.coords[:, 0]), [1.0])


def test_clean_cut_marks_no_face_as_coincident():
    """The new in-plane marking must not fire on an ordinary interior cut."""
    grid = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
    _, gix = slice_grid(
        grid, np.array([[1.5, -1.0, 0.0], [1.5, 3.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]), return_mapping=True)
    # only the inserted cut polygons are slices, one per split cell
    assert np.count_nonzero(gix.new.faces == 3) == np.count_nonzero(
        gix.new.cells == 2) // 2


def test_only_faces_mode_marks_no_slice_faces():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    _, gix = _snapped_cut(grid, only_faces=True, return_mapping=True)
    assert not np.any(gix.new.faces == 3)


@pytest.mark.parametrize("offset", [0.0, 0.25, -0.25])
def test_pinched_cut_conserves_volume_for_shifted_collapse(offset):
    """The collapse line may sit anywhere; only offset 0 puts it in the plane."""
    grid = cart_grid([1, 1, 1], [1.0, 1.0, 1.0])
    grid.nodes.coords[grid.nodes.coords[:, 2] > 0.5, 0] = 0.5 + offset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = compute_geometry(grid)
        cut = compute_geometry(
            slice_grid(grid, PLANE_POINT, normal=PLANE_NORMAL))
    assert np.isclose(cut.cells.volumes.sum(), grid.cells.volumes.sum())
    assert np.all(cut.cells.volumes > 1e-12)
