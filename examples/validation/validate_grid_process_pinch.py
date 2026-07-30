"""Validate processPINCH against MATLAB/MRST."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import process_grdecl, process_pinch


HERE = Path(__file__).resolve().parent


def _grdecl():
    nx, ny, nz = 2, 2, 3
    coord = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord.extend([float(i), float(j), 0.0,
                          float(i), float(j), 15.0])
    z = np.empty((2 * nx, 2 * ny, 2 * nz), dtype=float)
    for k in range(nz):
        z[:, :, 2 * k] = 5.0 * k
        z[:, :, 2 * k + 1] = 5.0 * (k + 1)
    actnum = np.ones((nx, ny, nz), dtype=np.int64)
    actnum[:, :, 1] = 0
    return {
        "cartDims": (nx, ny, nz),
        "COORD": np.asarray(coord),
        "ZCORN": z.ravel(order="F"),
        "ACTNUM": actnum.ravel(order="F"),
        "MULTZ": np.array([
            0.8, 0.7, 0.6, 0.5,
            0.3, 0.4, 0.2, 0.1,
            0.9, 0.95, 0.85, 0.88,
        ]),
        "PINCH": [5.0, "GAP", 0, "TOPBOT", "TOP"],
    }


def main() -> int:
    path = HERE / "ref_grid_process_pinch.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    grdecl = _grdecl()
    grid = process_grdecl(grdecl)
    top = process_pinch(grdecl, grid)
    grdecl["PINCH"][4] = "ALL"
    all_layers = process_pinch(grdecl, grid)
    grdecl["PINCH"][0] = 4.99
    excluded = process_pinch(grdecl, grid)

    checks = {
        "TOP cells": (top["cells"] + 1, ref["top_cells"]),
        "TOP faces": (top["faces"] + 1, ref["top_faces"]),
        "TOP multipliers": (top["mult"], ref["top_mult"]),
        "ALL cells": (all_layers["cells"] + 1, ref["all_cells"]),
        "ALL faces": (all_layers["faces"] + 1, ref["all_faces"]),
        "ALL multipliers": (all_layers["mult"], ref["all_mult"]),
        "threshold exclusion": (
            np.array(excluded is None),
            np.array(ref["excluded_is_empty"]).astype(bool),
        ),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual).squeeze()
        expected = np.asarray(expected).squeeze()
        passed = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-13, atol=1e-13
        )
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        if not passed:
            print(f"     PyMRST={actual!r}\n     MRST={expected!r}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
