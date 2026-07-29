"""Direct MATLAB/MRST validation of relperm-derived saturation limits."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.initialization import (
    getMinMaxPhaseSaturationsFromRelPerm,
)


def main() -> int:
    ref = loadmat(Path(__file__).with_name("ref_ad_core_relperm_limits.mat"))
    fluid = SimpleNamespace(
        krW=[
            lambda s: np.clip((s - 0.2) / 0.7, 0, 1),
            lambda s: np.clip((s - 0.1) / 0.8, 0, 1),
        ],
        krO=[
            lambda s: np.clip((s - 0.25) / 0.65, 0, 1),
            lambda s: np.clip((s - 0.15) / 0.75, 0, 1),
        ],
    )
    model = SimpleNamespace(
        fluid=fluid, water=True, oil=True, gas=False,
        get_phase_index=lambda phase: {"W": 0, "O": 1}[phase],
    )
    s_min, s_max = getMinMaxPhaseSaturationsFromRelPerm(
        model, tol=0.05, region=2
    )
    e_min = np.max(np.abs(s_min - ref["s_min"].ravel()))
    e_max = np.max(np.abs(s_max - ref["s_max"].ravel()))
    print("Relperm saturation limits vs MATLAB/MRST:")
    print(f"  {'ok  ' if e_min < 1e-14 else 'FAIL'} s_min: {e_min:.3e}")
    print(f"  {'ok  ' if e_max < 1e-14 else 'FAIL'} s_max: {e_max:.3e}")
    return int(e_min >= 1e-14 or e_max >= 1e-14)


if __name__ == "__main__":
    raise SystemExit(main())
