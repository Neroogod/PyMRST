"""Boundary-face selection, ported from MRST `boundaryFaceIndices`."""

from typing import Optional, Sequence

import numpy as np

SIDE_TAG = {
    "left": 0,
    "west": 0,
    "xmin": 0,
    "right": 1,
    "east": 1,
    "xmax": 1,
    "back": 2,
    "south": 2,
    "ymin": 2,
    "front": 3,
    "north": 3,
    "ymax": 3,
    "top": 4,
    "upper": 4,
    "zmin": 4,
    "bottom": 5,
    "lower": 5,
    "zmax": 5,
}


def boundary_face_indices(
    g,
    side: str,
    i1: Optional[Sequence[int]] = None,
    i2: Optional[Sequence[int]] = None,
    search_range: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return boundary faces on one side of a logically Cartesian grid."""
    tag = _side_tag(side)
    if g.cells.tags is None or g.cartDims is None:
        raise ValueError(
            "boundary_face_indices needs a logically Cartesian grid "
            "(cells.tags and cartDims)"
        )
    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    nx, ny, _ = dims
    normal_axis = tag // 2
    if normal_axis >= int(g.griddim):
        raise ValueError(f"side {side!r} is unsupported for a {g.griddim}-D grid")
    other_axes = [axis for axis in range(3) if axis != normal_axis]
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    faces = g.cells.faces[g.cells.tags == tag]
    cells = cell_no[g.cells.tags == tag]
    outer = g.faces.neighbors[faces].min(axis=1) < 0
    faces, cells = faces[outer], cells[outer]
    return _restrict_faces(
        g, faces, cells, dims, nx, ny, other_axes, normal_axis,
        i1, i2, search_range,
    )


def _side_tag(side: str) -> int:
    key = side.strip().lower()
    if key not in SIDE_TAG:
        raise ValueError(f"unknown side '{side}'; expected one of {sorted(set(SIDE_TAG))}")
    return SIDE_TAG[key]


def _restrict_faces(
    g,
    faces: np.ndarray,
    cells: np.ndarray,
    dims: list[int],
    nx: int,
    ny: int,
    other_axes: list[int],
    normal_axis: int,
    i1: Optional[Sequence[int]],
    i2: Optional[Sequence[int]],
    search_range: Optional[Sequence[int]],
) -> np.ndarray:
    if i1 is None and i2 is None and search_range is None:
        return np.unique(faces)
    gids = cells if g.cells.indexMap is None else np.asarray(g.cells.indexMap)[cells]
    ijk = np.column_stack([gids % nx, (gids // nx) % ny, gids // (nx * ny)])
    keep = np.ones(faces.size, dtype=bool)
    for index_range, axis, label in (
        (i1, other_axes[0], "i1"), (i2, other_axes[1], "i2"),
        (search_range, normal_axis, "range"),
    ):
        if index_range is None:
            continue
        values = np.asarray(index_range, dtype=np.int64).ravel()
        if values.size == 0:
            continue
        if np.any((values < 0) | (values >= dims[axis])):
            raise IndexError(f"{label} is outside the logical grid dimensions")
        keep &= np.isin(ijk[:, axis], values)
    return np.unique(faces[keep])


if __name__ == "__main__":

    def test_boundary_face_indices_finds_each_side() -> None:
        from pymrst.gridprocessing import cart_grid, compute_geometry

        g = compute_geometry(cart_grid([4, 3, 2], [40.0, 30.0, 20.0]))
        expected = {"xmin": 6, "xmax": 6, "ymin": 8, "ymax": 8, "zmin": 12, "zmax": 12}
        seen = set()
        for side, count in expected.items():
            faces = boundary_face_indices(g, side)
            assert faces.size == count
            assert np.all(g.faces.neighbors[faces].min(axis=1) < 0)
            assert not (set(faces.tolist()) & seen)
            seen |= set(faces.tolist())
        boundary = np.flatnonzero(g.faces.neighbors.min(axis=1) < 0)
        assert seen == set(boundary.tolist())
        assert np.array_equal(boundary_face_indices(g, "west"), boundary_face_indices(g, "xmin"))
        print("ok test_boundary_face_indices_finds_each_side")

    test_boundary_face_indices_finds_each_side()
    print("\nAll tests passed!")
