"""Phase 376: accumulateCartesianMultipliers against MRST (Phase 25 leftovers).

Pins `accumulate_cartesian_multipliers` against MRST's
`accumulateCartesianMultipliers` on a 3x2x2 grid with deterministic per-cell
MULTX/MULTY/MULTZ arrays. The per-face multiplier vector is compared number for
number (cartGrid face ordering matches between the two codes).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.solvers.accumulate_cartesian_multipliers import (
    accumulate_cartesian_multipliers,
)

REF = Path(__file__).with_name("ref_p376_multipliers.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase376_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
    mult = {"x": ref["multX"].ravel(), "y": ref["multY"].ravel(),
            "z": ref["multZ"].ravel()}
    face = accumulate_cartesian_multipliers(g, mult)

    failed = 0
    print("accumulateCartesianMultipliers (3x2x2, MULTX/Y/Z) vs MRST:")
    failed += _check("same number of faces", face.size == ref["facemult"].size,
                     f"{face.size} vs {ref['facemult'].size}")
    err = np.max(np.abs(face - ref["facemult"].ravel()))
    failed += _check("per-face multipliers match", err < 1e-12,
                     f"max|dm| = {err:.3e}")

    print("\naccumulateCartesianMultipliers matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
