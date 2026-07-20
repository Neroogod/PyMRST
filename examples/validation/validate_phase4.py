"""Compare the PyMRST compressible AD simulator against MATLAB MRST.

Both sides run flowSolverTutorialAD.m's model with identical parameters,
initial state, and Newton settings, so histories should agree to direct-
solver round-off.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from single_phase_compressible_ad import run_simulation  # noqa: E402
from pymrst.utils.units import BARSA  # noqa: E402


def main() -> int:
    path = HERE / "ref_p4_ad.mat"
    if not path.exists():
        print("ref_p4_ad.mat missing (run dump_phase4_reference.m in MATLAB)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    out = run_simulation()

    checks = [
        ("p_init", out["p_init"], ref["p_init"], 1e-9 * 250 * BARSA),
        ("WI", out["WI"], ref["WI"], 1e-12 * float(np.max(ref["WI"]))),
        ("dZ", out["dZ"], ref["dZ"], 1e-12 * 50),
        ("p_hist (53 steps)", out["p_hist"].ravel(),
         np.asarray(ref["p_hist"]).ravel(), 1e-8 * 250 * BARSA),
        ("bhp_hist", out["bhp_hist"], ref["bhp_hist"], 1e-8 * 250 * BARSA),
        ("qs_hist", out["qs_hist"], ref["qs_hist"],
         1e-8 * float(np.abs(ref["qs_hist"]).max())),
    ]
    failed = 0
    for name, py, ml, tol in checks:
        diff = float(np.max(np.abs(np.asarray(py, float).ravel()
                                   - np.asarray(ml, float).ravel())))
        ok = diff <= tol
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = "
              f"{diff:.3e} (tol {tol:.1e})")
        failed += 0 if ok else 1

    print("\nAll compared arrays match!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
