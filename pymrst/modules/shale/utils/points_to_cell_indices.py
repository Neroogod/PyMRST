"""Map physical points to Cartesian cell indices (MRST ``Pts2CellIdx``)."""

import numpy as np

from pymrst.gridprocessing.grid import Grid


def points_to_cell_indices(points, grid: Grid, *, perturb: float = 1.0000001) -> np.ndarray:
    """Return zero-based global cell indices with the i-index varying fastest."""
    if not hasattr(grid, "cartDims"):
        raise ValueError("points_to_cell_indices requires a Cartesian/tensor grid")
    dims = tuple(int(v) for v in grid.cartDims)
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != len(dims):
        raise ValueError(f"points must have shape (n, {len(dims)})")
    pts = pts * float(perturb)

    axes = [np.unique(grid.nodes.coords[:, axis]) for axis in range(len(dims))]
    ijk = np.empty_like(pts, dtype=np.int64)
    for axis, coordinates in enumerate(axes):
        # MATLAB finds the first grid coordinate strictly greater than p.
        ijk[:, axis] = np.searchsorted(coordinates, pts[:, axis], side="right") - 1
        bad = (ijk[:, axis] < 0) | (ijk[:, axis] >= dims[axis])
        if np.any(bad):
            raise ValueError(f"point lies outside grid along axis {axis}")

    strides = np.cumprod((1,) + dims[:-1], dtype=np.int64)
    return ijk @ strides


__all__ = ["points_to_cell_indices"]
