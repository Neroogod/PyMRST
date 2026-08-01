"""Check `sort_horizons` and `refine_grdecl_layers` against MRST.

Reference: scripts/matlab/ref_horizons.mat. Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_horizons_reference.m')"

`refine_grdecl_layers` is compared on its full `ZCORN` array, entry for entry -
there is no numbering ambiguity here, since a corner-point deck is indexed by
logical position rather than by any traversal order. The `(2, 3)` MATLAB band
is `(1, 2)` here; the conversion is the thing most likely to be wrong, so the
cases deliberately include a band at the very start and one covering the whole
grid, where an off-by-one would run off an end rather than land somewhere
plausible.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing.refine_grdecl import grdecl_xyz                  # noqa: E402
from pymrst.utils.gridtools.refine_grdecl_layers import (                   # noqa: E402
    refine_grdecl_layers,
)
from pymrst.utils.gridtools.sort_horizons import sort_horizons              # noqa: E402


def box(nx=3, ny=3, nz=4):
    xyz = np.zeros((6, nx + 1, ny + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            xyz[:, i, j] = [i, j, 0.0, i, j, float(nz)]
    zc = np.zeros((2 * nx, 2 * ny, 2 * nz))
    for k in range(nz):
        zc[:, :, 2 * k] = k
        zc[:, :, 2 * k + 1] = k + 1
    return {"cartDims": np.array([nx, ny, nz]),
            "COORD": xyz.ravel(order="F"), "ZCORN": zc.ravel(order="F")}


def surface(z, n=3):
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    return {"x": x, "y": y, "z": np.asarray(z, dtype=float)}


# MATLAB band (1-based, inclusive) -> PyMRST band (0-based, inclusive)
CASES = {"band23": ((1, 2), 3), "single3": ((2, 2), 5),
         "first": ((0, 0), 2), "all": ((0, 3), 2)}


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_horizons.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []

    _, order = sort_horizons([surface(np.full((3, 3), v))
                              for v in (30.0, 10.0, 20.0)])
    ok = np.array_equal(order, np.asarray(reference["sorted_ix"],
                                          dtype=np.int64).ravel() - 1)
    print(f"  {'ok ' if ok else 'FAIL'} sortHorizons: order")
    if not ok:
        failures.append("sortHorizons order")

    _, tilt = sort_horizons([surface(np.full((3, 3), 50.0)),
                             surface(np.tile([0.0, 100.0, 100.0], (3, 1)))])
    ok = np.array_equal(tilt, np.asarray(reference["tilt_ix"],
                                         dtype=np.int64).ravel() - 1)
    print(f"  {'ok ' if ok else 'FAIL'} sortHorizons: min decides, not mean")
    if not ok:
        failures.append("sortHorizons tilt")

    for name, (band, ref) in CASES.items():
        expected = reference[name]
        got = refine_grdecl_layers(box(), band, ref)

        same_dims = tuple(np.asarray(got["cartDims"], dtype=int)) == tuple(
            np.asarray(expected.cartDims, dtype=int).ravel())
        print(f"  {'ok ' if same_dims else 'FAIL'} {name}: cartDims "
              f"{tuple(int(v) for v in np.asarray(got['cartDims']))}")
        if not same_dims:
            failures.append(f"{name} cartDims")
            continue

        _, mine = grdecl_xyz(got)
        theirs = np.asarray(expected.zcorn, dtype=float)
        ok = mine.shape == theirs.shape and np.allclose(mine, theirs)
        print(f"  {'ok ' if ok else 'FAIL'} {name}: ZCORN {mine.shape}")
        if not ok:
            failures.append(f"{name} ZCORN")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("sort_horizons and refine_grdecl_layers match MRST.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
