"""Phase 421: DG cell dimensions against MRST (computeCellDimensions).

Pins PyMRST's `compute_cell_dimensions` against MRST's `computeCellDimensions`:
the per-cell reference-frame span `dx` and bounding box `xMin`/`xMax`, on a
Cartesian and a perturbed 2-D grid and a 3-D grid. These set the coordinate
transform the DG operators use to map each cell onto [-1, 1]^dim.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.dg.discretizations.dg_discretization import compute_cell_dimensions

REF = Path(__file__).with_name("ref_p421_cell_dims.mat")


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _cart2():
    return compute_geometry(cart_grid([3, 4], [6.0, 8.0]))


def _pert2():
    g = cart_grid([4, 3], [4.0, 3.0])
    c = np.asarray(g.nodes.coords, dtype=float)
    L = 4.0
    c[:, 0] += 0.2 * np.sin(np.pi * c[:, 0] / L) * np.sin(np.pi * c[:, 1] / 3.0)
    g.nodes.coords = c
    return compute_geometry(g)


def _cart3():
    return compute_geometry(cart_grid([2, 3, 2], [4.0, 3.0, 2.0]))


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase421_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    cases = [("2-D Cartesian", _cart2(), "cart2"),
             ("2-D perturbed", _pert2(), "pert2"),
             ("3-D Cartesian", _cart3(), "cart3")]

    failed = 0
    print("compute_cell_dimensions vs MRST:")
    for name, g, key in cases:
        compute_cell_dimensions(g)
        d_dx = np.max(np.abs(g.cells.dx - ref[key + "_dx"]))
        d_min = np.max(np.abs(g.cells.xMin - ref[key + "_xmin"]))
        d_max = np.max(np.abs(g.cells.xMax - ref[key + "_xmax"]))
        failed += _check(f"{name}: dx, xMin, xMax match",
                         max(d_dx, d_min, d_max) < 1e-12,
                         f"max|d| {max(d_dx, d_min, d_max):.1e}")

    print("\nDG cell dimensions match MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
