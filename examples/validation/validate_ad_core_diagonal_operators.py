"""Direct MATLAB/MRST validation of diagonal backend grid operators."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.backends.diagonal.operators import (
    discreteDivergence,
    faceAverage,
    singlePointUpwind,
    twoPointGradient,
)


def main() -> int:
    ref = loadmat(Path(__file__).with_name("ref_ad_core_diagonal_operators.mat"))
    neighbors = ref["N"].astype(np.int64) - 1
    cells = ref["cells"].ravel()
    flux = ref["flux"].ravel()
    flag = ref["flag"].ravel().astype(bool)
    options = SimpleNamespace(N=neighbors, nc=cells.size)

    comparisons = {
        "twoPointGradient": (twoPointGradient(neighbors, cells), ref["grad"]),
        "faceAverage": (faceAverage(neighbors, cells), ref["avg"]),
        "singlePointUpwind": (
            singlePointUpwind(flag, neighbors, cells),
            ref["upwind"],
        ),
        "discreteDivergence": (
            discreteDivergence(None, flux, options),
            ref["div"],
        ),
        "discreteDivergence + accumulation": (
            discreteDivergence(np.ones(cells.size), flux, options),
            ref["div_acc"],
        ),
    }

    failed = 0
    print("Diagonal backend operators vs MATLAB/MRST:")
    for name, (actual, expected) in comparisons.items():
        error = np.max(np.abs(np.asarray(actual).ravel() - expected.ravel()))
        ok = error < 1e-14
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d| = {error:.3e}")
    print("\nOperators match MATLAB/MRST!" if not failed
          else f"\n{failed} comparisons FAILED")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
