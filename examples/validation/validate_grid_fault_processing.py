"""Validate processFaults/removeFaultBdryFaces against MATLAB/MRST."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import (
    cart_grid,
    process_faults,
    remove_fault_boundary_faces,
)


HERE = Path(__file__).resolve().parent


def _names(value) -> list[str]:
    return [str(item).strip() for item in np.atleast_1d(value).tolist()]


def _packed(faults):
    counts = np.array([fault.numf for fault in faults], dtype=np.int64)
    positions = np.concatenate([[1], 1 + np.cumsum(counts)])
    faces = (
        np.concatenate([fault.faces + 1 for fault in faults])
        if faults else np.zeros(0, dtype=np.int64)
    )
    return counts, positions, faces


def main() -> int:
    path = HERE / "ref_grid_fault_processing.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    geometry = {
        "FAULTS": [
            ["F_MAIN", 2, 2, 1, 2, 1, 2, "X+"],
            ["F_MAIN", 2, 2, 3, 3, 1, 2, "X+"],
            ["BOUND", 1, 1, 1, 3, 1, 2, "X-"],
        ],
        "MULTFLT": [["F_MAIN", 0.25], ["BOUND", 0.50]],
    }
    grid = cart_grid([3, 3, 2])
    faults, ids = process_faults(grid, geometry)
    interior, keep = remove_fault_boundary_faces(faults, grid)
    numf, pos, faces = _packed(faults)
    inumf, ipos, ifaces = _packed(interior)

    checks = {
        "fault names": (
            np.asarray([fault.name for fault in faults]),
            np.asarray(_names(ref["fault_names"])),
        ),
        "fault counts": (numf, np.atleast_1d(ref["fault_numf"])),
        "fault multipliers": (
            np.array([fault.multiplier for fault in faults]),
            np.atleast_1d(ref["fault_mult"]),
        ),
        "fault positions": (pos, np.atleast_1d(ref["fault_pos"])),
        "fault faces": (faces, np.atleast_1d(ref["fault_faces"])),
        "fault ids": (
            np.array([ids["BOUND"] + 1, ids["F_MAIN"] + 1]),
            np.array([ref["id_bound"], ref["id_main"]]),
        ),
        "keep mask": (keep, np.atleast_1d(ref["keep"]).astype(bool)),
        "interior names": (
            np.asarray([fault.name for fault in interior]),
            np.asarray(_names(ref["interior_names"])),
        ),
        "interior counts": (inumf, np.atleast_1d(ref["interior_numf"])),
        "interior multipliers": (
            np.array([fault.multiplier for fault in interior]),
            np.atleast_1d(ref["interior_mult"]),
        ),
        "interior positions": (ipos, np.atleast_1d(ref["interior_pos"])),
        "interior faces": (ifaces, np.atleast_1d(ref["interior_faces"])),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
        passed = actual.shape == expected.shape and np.array_equal(actual, expected)
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        if not passed:
            print(f"     PyMRST={actual!r}\n     MRST={expected!r}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
