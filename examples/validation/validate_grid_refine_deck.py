"""Validate refineDeck against MATLAB/MRST."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import refine_deck


HERE = Path(__file__).resolve().parent


def _deck():
    nx, ny, nz = 2, 1, 2
    xyz = np.zeros((6, nx + 1, ny + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            xyz[:, i, j] = [i, j, 0.0, i, j, nz]
    z = np.empty((2 * nx, 2 * ny, 2 * nz))
    for k in range(nz):
        z[:, :, 2 * k] = k
        z[:, :, 2 * k + 1] = k + 1
    return {
        "RUNSPEC": {"DIMENS": np.array([2, 1, 2])},
        "GRID": {
            "cartDims": np.array([2, 1, 2]),
            "COORD": xyz.ravel(order="F"),
            "ZCORN": z.ravel(order="F"),
            "PORO": np.array([0.1, 0.2, 0.3, 0.4]),
        },
        "SOLUTION": {
            "PRESSURE": np.array([100, 200, 300, 400]),
            "SWAT": np.array([0.1, 0.2, 0.3, 0.4]),
        },
        "REGIONS": {"SATNUM": np.array([1, 2, 3, 4])},
        "PROPS": {"SWL": np.array([0.01, 0.02, 0.03, 0.04])},
        "SCHEDULE": {"control": {
            "WELSPECS": np.array([
                ["WZ", "G", 1, 1], ["WX", "G", 2, 1],
            ], dtype=object),
            "COMPDAT": np.array([
                ["WZ", 1, 1, 1, 2, "OPEN", 0, 12, 0, 18, 0, 0, "Z"],
                ["WX", 2, 1, 1, 1, "OPEN", 0, 20, 0, 30, 0, 0, "X"],
            ], dtype=object),
        }},
    }


def _directions(value):
    return np.array([
        str(item).strip() for item in np.atleast_1d(value).tolist()
    ])


def main() -> int:
    path = HERE / "ref_grid_refine_deck.mat"
    if not path.exists():
        print(f"{path.name} missing (run the matching MATLAB dump)")
        return 1
    ref = loadmat(path, squeeze_me=True)
    refined = refine_deck(_deck(), [2, 3, 2])
    control = refined["SCHEDULE"]["control"]
    compdat = control["COMPDAT"]
    defaulted = refine_deck(_deck(), [2, 1, 1], default_well=True)
    default_compdat = defaulted["SCHEDULE"]["control"]["COMPDAT"]
    checks = {
        "RUNSPEC dimensions": (
            refined["RUNSPEC"]["DIMENS"], ref["refined_dims"],
        ),
        "GRID dimensions": (
            refined["GRID"]["cartDims"], ref["refined_grid_dims"],
        ),
        "PRESSURE": (
            refined["SOLUTION"]["PRESSURE"], ref["refined_pressure"],
        ),
        "SWAT": (refined["SOLUTION"]["SWAT"], ref["refined_swat"]),
        "SATNUM": (refined["REGIONS"]["SATNUM"], ref["refined_satnum"]),
        "SWL": (refined["PROPS"]["SWL"], ref["refined_swl"]),
        "GRID PORO": (refined["GRID"]["PORO"], ref["refined_poro"]),
        "WELSPECS ij": (
            control["WELSPECS"][:, 2:4].astype(float),
            ref["refined_welspec_ij"],
        ),
        "COMPDAT ijk": (
            compdat[:, 1:5].astype(float), ref["refined_compdat_ijk"],
        ),
        "COMPDAT trans": (
            compdat[:, 7].astype(float), ref["refined_compdat_trans"],
        ),
        "COMPDAT KH": (
            compdat[:, 9].astype(float), ref["refined_compdat_kh"],
        ),
        "COMPDAT direction": (
            _directions(compdat[:, 12]),
            _directions(ref["refined_compdat_dir"]),
        ),
        "default trans": (
            default_compdat[:, 7] == "-1",
            np.asarray(ref["default_trans"]).astype(bool),
        ),
        "default KH": (
            default_compdat[:, 9] == "-1",
            np.asarray(ref["default_kh"]).astype(bool),
        ),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
        if actual.dtype.kind in "USO" or expected.dtype.kind in "USO":
            passed = actual.shape == expected.shape and np.array_equal(
                actual.astype(str), expected.astype(str)
            )
        else:
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
