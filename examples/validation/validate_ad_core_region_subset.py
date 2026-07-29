"""Direct MATLAB/MRST validation of evaluateFunctionCellSubsetReg."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.statefunctions.flowprops.hysteresis import (
    evaluateFunctionCellSubsetReg,
)


def main() -> int:
    ref = loadmat(Path(__file__).with_name("ref_ad_core_region_subset.mat"))
    funcs = [lambda v: 2 * v, lambda v: 10 + v]
    x = ref["x"].ravel()
    normal = evaluateFunctionCellSubsetReg(
        None, funcs, ref["regions"].ravel(), x
    )
    offset = evaluateFunctionCellSubsetReg(
        None, funcs, ref["regions_offset"].ravel(), x
    )
    e1 = np.max(np.abs(normal - ref["v"].ravel()))
    e2 = np.max(np.abs(offset - ref["v_offset"].ravel()))
    print("Regional subset evaluation vs MATLAB/MRST:")
    print(f"  {'ok  ' if e1 < 1e-14 else 'FAIL'} base regions: {e1:.3e}")
    print(f"  {'ok  ' if e2 < 1e-14 else 'FAIL'} offset regions: {e2:.3e}")
    return int(e1 >= 1e-14 or e2 >= 1e-14)


if __name__ == "__main__":
    raise SystemExit(main())
