"""Phase 75: vemmech calibration BC helper vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.calibration import (
    identify_boundary_elements,
    set_bcfun,
)
from pymrst.modules.vemmech.vem import create_augmented_grid

HERE = Path(__file__).parent
REF = HERE / "ref_p75_vemmech_bcfun.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _disp_matrix(tensor, g, order="nd"):
    if order == "nd":
        return _dense(tensor.as_matrix([["n"], ["d"]], True,
                                       [g.nodes.num, g.griddim]))
    return _dense(tensor.as_matrix([["d"], ["n"]], True,
                                   [g.griddim, g.nodes.num]))


def compare(name, py, ml, rtol=1e-12, atol=1e-12) -> int:
    py = np.asarray(py, dtype=float)
    ml = np.asarray(ml, dtype=float)
    same_nan = np.array_equal(np.isnan(py), np.isnan(ml))
    pyv = np.nan_to_num(py, nan=0.0).ravel()
    mlv = np.nan_to_num(ml, nan=0.0).ravel()
    scale = max(float(np.max(np.abs(mlv))) if mlv.size else 0.0, 1.0)
    diff = 0.0 if pyv.size == mlv.size == 0 else float(np.max(np.abs(pyv - mlv)))
    ok = py.shape == ml.shape and same_nan and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase75_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True, struct_as_record=False)

    g = create_augmented_grid(
        compute_geometry(cart_grid([2, 2, 3], [2.0, 2.0, 3.0]))
    )
    binfo = identify_boundary_elements(g)

    failures = 0
    print("Phase 75: vemmech calibration BC helper vs MATLAB MRST")

    bcfun5, num5 = set_bcfun(g, binfo, _arr(ref, "bcond5"))
    bc5 = bcfun5(_arr(ref, "u5").ravel())
    failures += compare("5-row num controls", [num5], [_arr(ref, "num5")],
                        rtol=0, atol=0)
    failures += compare("5-row disp_bc n,d", _disp_matrix(bc5.disp_bc, g),
                        _arr(ref, "disp5_nd"))
    failures += compare("5-row disp_bc d,n", _disp_matrix(bc5.disp_bc, g, "dn"),
                        _arr(ref, "disp5_dn"))

    bcfun3, num3 = set_bcfun(g, binfo, _arr(ref, "bcond3"))
    bc3 = bcfun3(_arr(ref, "u3").ravel())
    failures += compare("3-row num controls", [num3], [_arr(ref, "num3")],
                        rtol=0, atol=0)
    failures += compare("3-row disp_bc n,d", _disp_matrix(bc3.disp_bc, g),
                        _arr(ref, "disp3_nd"))

    if failures:
        print(f"\nvemmech BC helper validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST vemmech calibration BC helper.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
