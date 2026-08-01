"""Check `convert_horizons_to_grid` against MRST (scripts/matlab/ref_convert_horizons.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; cd('.../scripts/matlab'); dump_convert_horizons_reference"

(The dump defines a local function, so MATLAB needs it run as a script on the
path rather than through `run`.)

`COORD`, `ZCORN` and `ACTNUM` are compared **entry for entry**. A corner-point
deck is indexed by logical position, so unlike the unstructured-grid validators
there is no numbering ambiguity to work around - if the arrays differ anywhere,
the port is wrong.

Non-finite entries are compared as such rather than numerically: the deck
carries `inf` where a horizon had no data, and `np.allclose` says nothing
useful about those. The check is that the two agree on *which* entries are
infinite and on the values of all the rest.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.utils.gridtools.convert_horizons_to_grid import (              # noqa: E402
    convert_horizons_to_grid,
)


def horizon(n, x0, x1, y0, y1, f):
    x, y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n),
                       indexing="ij")
    return {"x": x, "y": y, "z": f(x, y)}


FLAT0 = horizon(8, 0, 1, 0, 1, lambda x, y: np.zeros_like(x))
FLAT5 = horizon(8, 0, 1, 0, 1, lambda x, y: np.full_like(x, 5.0))
FLAT10 = horizon(8, 0, 1, 0, 1, lambda x, y: np.full_like(x, 10.0))
SLOPE = horizon(8, 0, 1, 0, 1, lambda x, y: 5.0 * x)
OFFSET = horizon(8, 0.5, 1.5, 0, 1, lambda x, y: np.full_like(x, 10.0))

CASES = {
    "box": ([FLAT0, FLAT10], {}),
    "layers3": ([FLAT0, FLAT10], {"layers": 3}),
    "slope": ([SLOPE, FLAT10], {}),
    "multi": ([FLAT0, FLAT5, FLAT10], {"layers": [2, 3]}),
    "dims": ([FLAT0, FLAT10], {"dims": [20, 15]}),
    "offset": ([FLAT0, OFFSET], {}),
}


def same(a, b):
    """Equal entry for entry, treating non-finite values as labels."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    finite = np.isfinite(a)
    if not np.array_equal(finite, np.isfinite(b)):
        return False, (f"{int(np.count_nonzero(finite != np.isfinite(b)))} "
                       f"entries differ in finiteness")
    if not np.allclose(a[finite], b[finite], rtol=1e-10, atol=1e-10):
        worst = np.max(np.abs(a[finite] - b[finite]))
        return False, f"max difference {worst:.3e}"
    return True, f"{a.size} entries"


def main() -> int:
    reference = loadmat(
        Path(__file__).parent / "matlab" / "ref_convert_horizons.mat",
        squeeze_me=True, struct_as_record=False)
    failures = []
    for name, (horizons, options) in CASES.items():
        ref = reference[name]
        deck = convert_horizons_to_grid(horizons, **options)

        ok = tuple(int(v) for v in deck["cartDims"]) == tuple(
            int(v) for v in np.atleast_1d(np.asarray(ref.cartDims)))
        print(f"  {'ok ' if ok else 'FAIL'} {name}: cartDims "
              f"{tuple(int(v) for v in deck['cartDims'])}")
        if not ok:
            failures.append(f"{name} cartDims")
            continue

        for field, mine, theirs in (("COORD", deck["COORD"], ref.COORD),
                                    ("ZCORN", deck["ZCORN"], ref.ZCORN),
                                    ("ACTNUM", deck["ACTNUM"], ref.ACTNUM)):
            ok, detail = same(mine, theirs)
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {field} ({detail})")
            if not ok:
                failures.append(f"{name} {field}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("convert_horizons_to_grid matches MRST's convertHorizonsToGrid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
