"""Phase 76: vemmech calibration calculate_stress vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.calibration import (
    calculate_stress,
    identify_boundary_elements,
    set_bcfun,
)
from pymrst.modules.vemmech.vem import create_augmented_grid

HERE = Path(__file__).parent
REF = HERE / "ref_p76_vemmech_calculate_stress.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def compare(name, py, ml, rtol=1e-12, atol=1e-8) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ml))) if ml.size else 0.0, 1.0)
    diff = 0.0 if py.size == ml.size == 0 else float(np.max(np.abs(py - ml)))
    ok = py.shape == ml.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase76_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True, struct_as_record=False)

    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(g, binfo, _arr(ref, "bcond"))
    u = _arr(ref, "u").ravel()
    E = _arr(ref, "E").ravel()
    nu = _arr(ref, "nu").ravel()
    efun = lambda controls: E
    nufun = lambda controls: nu

    py = calculate_stress(u, _arr(ref, "dd").ravel(), g, bcfun, efun, nufun)

    failures = 0
    print("Phase 76: vemmech calibration calculate_stress vs MATLAB MRST")
    failures += compare("num controls", [num_bc], [_arr(ref, "num_bc")],
                        rtol=0, atol=0)
    failures += compare("stress vector", py, _arr(ref, "stress_vec"),
                        rtol=1e-11, atol=1e-7)
    failures += compare("stress matrix", py.reshape(g.cells.num, 6),
                        _arr(ref, "stress"), rtol=1e-11, atol=1e-7)

    if failures:
        print(f"\nvemmech calculate_stress validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST vemmech calibration calculate_stress.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
