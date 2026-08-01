"""Check `make_layered_horizon_grid` against MRST (scripts/matlab/ref_layered_horizon.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; cd('.../scripts/matlab'); dump_layered_horizon_reference"

Compared **order-independently**: counts, total volume, and the sorted node
depths and cell volumes. The extrusion inherits the areal grid's node and cell
numbering, which differs between the two codes, and `remove_short_edges` then
renumbers again after collapsing - so an entry-wise comparison would fail on
numbering alone. Sorted volumes and depths pin the geometry regardless.

The `wedge` case is the one that matters most: a horizon that touches the one
above it at `x = 0`, so cells collapse there and the grid comes back smaller
than `n * layers`. That exercises the collapse path and the cell map together,
and both codes must lose the same number of cells.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry               # noqa: E402
from pymrst.utils.gridtools.make_layered_horizon_grid import (              # noqa: E402
    make_layered_horizon_grid,
)


def cloud(f, n=9):
    x, y = np.meshgrid(np.linspace(-0.1, 1.1, n), np.linspace(-0.1, 1.1, n))
    return {"x": x, "y": y, "z": np.asarray(f(x, y), dtype=float)}


FLAT0 = cloud(lambda x, y: np.zeros_like(x))
FLAT5 = cloud(lambda x, y: np.full_like(x, 5.0))
FLAT9 = cloud(lambda x, y: np.full_like(x, 9.0))
FLAT10 = cloud(lambda x, y: np.full_like(x, 10.0))
SLOPE = cloud(lambda x, y: 5.0 * x)

CASES = {
    "layers3": ([FLAT0, FLAT10], {"layers": 3}),
    "one": ([FLAT0, FLAT10], {}),
    "multi": ([FLAT0, FLAT5, FLAT9], {"layers": [2, 3]}),
    "slope": ([SLOPE, FLAT10], {}),
    "wedge": ([FLAT0, SLOPE], {}),
}


def main() -> int:
    reference = loadmat(
        Path(__file__).parent / "matlab" / "ref_layered_horizon.mat",
        squeeze_me=True, struct_as_record=False)
    base = compute_geometry(cart_grid([4, 4], [1.0, 1.0]))
    failures = []

    for name, (horizons, options) in CASES.items():
        ref = reference[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, cellmap = make_layered_horizon_grid(base, horizons, **options)
        g = compute_geometry(g)

        checks = [
            ("cell count", g.cells.num == int(ref.numCells),
             f"{g.cells.num} vs {int(ref.numCells)}"),
            ("node count", g.nodes.num == int(ref.numNodes),
             f"{g.nodes.num} vs {int(ref.numNodes)}"),
            ("total volume", np.isclose(np.sum(g.cells.volumes),
                                        float(ref.totVol), rtol=1e-9),
             f"{np.sum(g.cells.volumes):.6f} vs {float(ref.totVol):.6f}"),
            ("sorted depths", np.allclose(
                np.sort(np.asarray(g.nodes.coords)[:, 2]),
                np.asarray(ref.zsorted, dtype=float).ravel(), atol=1e-9), ""),
            ("sorted volumes", np.allclose(
                np.sort(np.asarray(g.cells.volumes)),
                np.asarray(ref.volsorted, dtype=float).ravel(), atol=1e-9), ""),
            ("layer histogram", np.array_equal(
                np.bincount(np.asarray(g.cells.layer, dtype=np.int64)),
                np.bincount(np.asarray(ref.layer, dtype=np.int64).ravel() - 1)),
             ""),
            ("cellmap size", np.asarray(cellmap).size
             == np.asarray(ref.cellmap).size, ""),
        ]
        for label, ok, detail in checks:
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {label}"
                  f"{'  ' + detail if detail else ''}")
            if not ok:
                failures.append(f"{name} {label}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("make_layered_horizon_grid matches MRST's makeLayeredHorizonGrid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
