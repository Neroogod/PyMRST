"""Validate foundational gridprocessing helpers against MATLAB/MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import (
    build_corner_pt_pillars,
    cart2active,
    check_and_repair_zcorn,
)


HERE = Path(__file__).resolve().parent


def main() -> int:
    reference_path = HERE / "ref_grid_corner_point_foundations.mat"
    if not reference_path.exists():
        print(f"{reference_path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(reference_path, squeeze_me=True)
    grdecl = {
        "COORD": np.array([
            0.0, 0.0, 0.0, 10.0, 20.0, 10.0,
            4.0, 5.0, 3.0, 4.0, 5.0, 3.0,
        ]),
        "ZCORN": np.array([2.0, 8.0]),
    }
    x, y, z = build_corner_pt_pillars(grdecl)
    sx, sy, sz = build_corner_pt_pillars(grdecl, scale=True)
    grid = SimpleNamespace(
        cartDims=(2, 2, 2),
        cells=SimpleNamespace(indexMap=np.array([0, 2, 3, 6])),
    )
    active = cart2active(grid, np.asarray(ref["cart_query"], dtype=int) - 1)
    actual = {
        "pillar_x": x,
        "pillar_y": y,
        "pillar_z": z,
        "scaled_x": sx,
        "scaled_y": sy,
        "scaled_z": sz,
        "active_cells": active + 1,
        "repaired_cell": check_and_repair_zcorn(
            ref["cell_inversion"], [1, 1, 1]
        ),
        "repaired_layers": check_and_repair_zcorn(
            ref["layer_overlap"], [1, 1, 2]
        ),
        "inactive_layers": check_and_repair_zcorn(
            ref["layer_overlap"], [1, 1, 2], active=[1, 0]
        ),
    }

    failed = False
    for name, value in actual.items():
        expected = np.asarray(ref[name])
        value = np.asarray(value)
        same_shape = value.squeeze().shape == expected.squeeze().shape
        error = (
            float(np.max(np.abs(value.squeeze() - expected.squeeze())))
            if same_shape else float("inf")
        )
        passed = same_shape and np.allclose(
            value.squeeze(), expected.squeeze(), rtol=1e-13, atol=1e-13
        )
        print(f"{'ok  ' if passed else 'FAIL'} {name}: max abs {error:.3e}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
