"""Phase 78: numeric VEM_assemble_AD / VEM_linElast_AD path."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import (
    SparseMultiArray,
    vem_assemble_ad,
    vem_lin_elast_ad,
)
from pymrst.modules.vemmech.calibration import identify_boundary_elements, set_bcfun
from pymrst.modules.vemmech.vem import create_augmented_grid

HERE = Path(__file__).parent
REF = HERE / "ref_p78_vemmech_ad_path.mat"


def _arr(ref, key):
    val = ref[key]
    return val.toarray() if hasattr(val, "toarray") else np.asarray(val, dtype=float)


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _field(x, name):
    return getattr(x, name)


def compare(name, py, ml, rtol=1e-11, atol=1e-8) -> int:
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
            f"Missing {REF.name}. Run dump_phase78_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True, struct_as_record=False)

    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(g, binfo, _arr(ref, "bcond"))
    bc = bcfun(_arr(ref, "u").ravel())
    load = SparseMultiArray(_arr(ref, "load_mat"), ["cn", "d"])
    E = _arr(ref, "E").ravel()
    nu = _arr(ref, "nu").ravel()

    S, op = vem_assemble_ad(g, E, nu)
    uu, extra = vem_lin_elast_ad(g, E, nu, bc, load)
    _, op_reuse = vem_assemble_ad(g, E, nu, extra=op)
    uu_ns, extra_ns = vem_lin_elast_ad(g, E, nu, bc, load, extra=op,
                                       no_solve=True)

    failures = 0
    print("Phase 78: numeric VEM_assemble_AD / VEM_linElast_AD path")
    failures += compare("num controls", [num_bc], [_arr(ref, "num_bc")],
                        rtol=0, atol=0)
    failures += compare("assemble S", _dense(S), _arr(ref, "S"))
    failures += compare("assemble reuse S", _dense(op_reuse.S), _arr(ref, "S"))
    failures += compare("D matrix", _dense(op.D), _dense(_field(ref["op"], "D")))
    failures += compare("displacement", uu, _arr(ref, "uu"), rtol=1e-10)
    failures += compare("extra S", _dense(extra.S), _arr(ref, "S"))
    ref_disc = _field(ref["extra"], "disc")
    failures += compare("disc V_dir", extra.disc["V_dir"],
                        _field(ref_disc, "V_dir"))
    failures += compare("disc divrhs", extra.disc["divrhs"],
                        _field(ref_disc, "divrhs"))
    ns_flat = uu_ns.ravel()
    u_flat = uu.ravel()
    isdir = extra_ns.disc["isdirdofs"]
    failures += compare("no_solve dir values", ns_flat[isdir], u_flat[isdir],
                        rtol=0, atol=0)
    if not np.isnan(ns_flat[~isdir]).all():
        print("  FAIL no_solve free values: expected NaNs")
        failures += 1
    else:
        print("  ok   no_solve free values: NaNs present")

    if failures:
        print(f"\nvemmech AD path validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches the numeric VEM AD assembly/solve path.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
