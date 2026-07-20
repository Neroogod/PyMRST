"""Compare PyMRST's coarsening and upscaling against MATLAB.

The partition, the volume-weighted porosity, and the flow-based
(sealed-sides) permeability upscaling of a heterogeneous 12x12x6 model
into 3x3x2 coarse blocks.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.upscaling import (
    generate_coarse_grid,
    partition_cart_grid,
    upscale_perm_fixed,
    upscale_poro,
)
from pymrst.params import make_rock

HERE = Path(__file__).parent


def main() -> int:
    path = HERE / "ref_p12_upscaling.mat"
    if not path.exists():
        print("ref_p12_upscaling.mat missing (run dump_phase12_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    failed = 0

    def chk(name, a, b, rtol):
        nonlocal failed
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        scale = max(np.abs(b).max(), 1e-300)
        diff = float(np.abs(a - b).max()) if a.shape == b.shape else np.inf
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name:26s} rel diff = "
              f"{diff / scale:.2e}  (tol {rtol:.0e})")
        failed += 0 if ok else 1

    fine = [int(v) for v in np.atleast_1d(ref["fine"])]
    coarse = [int(v) for v in np.atleast_1d(ref["coarse"])]

    g = compute_geometry(cart_grid(fine, [120.0, 120.0, 60.0]))
    nc = g.cells.num
    kvals = (1 + (13 * np.arange(nc)) % 200) * 9.869232667160129e-16
    poro = 0.1 + 0.2 * (np.arange(nc) % 5) / 4.0
    rock = make_rock(g, kvals[:, None], poro[:, None])

    part = partition_cart_grid(fine, coarse)
    ml_part = np.asarray(ref["partition"], np.int64).ravel() - 1
    same = np.array_equal(part, ml_part)
    print(f"  {'ok  ' if same else 'FAIL'} {'partition':26s} "
          f"{'identical' if same else 'DIFFERS'}")
    failed += 0 if same else 1

    cg = generate_coarse_grid(g, part)
    assert cg.n_blocks == int(np.prod(coarse))

    chk("upscaled porosity", upscale_poro(g, rock, cg), ref["poroU"], 1e-12)

    kk = upscale_perm_fixed(g, rock, cg)
    ml_k = np.atleast_2d(np.asarray(ref["K"], float))
    chk("upscaled permeability", kk, ml_k, 1e-9)

    print("\nUpscaling matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
