"""Validate markCutGrids against MATLAB/MRST."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry, mark_cut_grids


HERE = Path(__file__).resolve().parent


def main() -> int:
    path = HERE / "ref_grid_mark_cut_grids.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    grid = compute_geometry(cart_grid([4, 2, 1]))
    status = np.asarray(ref["face_status"], dtype=np.int64).ravel()
    checks = {
        "component labels": (
            mark_cut_grids(grid, status) + 1,
            np.asarray(ref["labels"]).ravel(),
        ),
        "legacy left flood": (
            mark_cut_grids(grid, status, legacy=True, start=0),
            np.asarray(ref["legacy_left"]).ravel(),
        ),
        "legacy right flood": (
            mark_cut_grids(
                grid, status, legacy=True, start=grid.cells.centroids[3]
            ),
            np.asarray(ref["legacy_right"]).ravel(),
        ),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        passed = np.array_equal(actual, expected)
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
