"""Phase 77: vemmech calibration Gaussian objective vs MATLAB MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.calibration import (
    gaussian_objfun,
    identify_boundary_elements,
    set_bcfun,
    set_material_fun,
    setup_objective_fun,
    theta_diff,
)
from pymrst.modules.vemmech.vem import create_augmented_grid

HERE = Path(__file__).parent
REF = HERE / "ref_p77_vemmech_objective.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def _field(obj, name):
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return obj[name]
    return getattr(obj, name)


def _props(mat_struct):
    return SimpleNamespace(
        young=np.asarray(_field(mat_struct, "young"), dtype=float).ravel(),
        poisson=np.asarray(_field(mat_struct, "poisson"), dtype=float).ravel(),
        rho=np.asarray(_field(mat_struct, "rho"), dtype=float).ravel(),
    )


def compare(name, py, ml, rtol=1e-12, atol=1e-10) -> int:
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
            f"Missing {REF.name}. Run dump_phase77_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True, struct_as_record=False)

    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(g, binfo, _arr(ref, "bcond"))
    props = _props(ref["props"])
    matcond = _arr(ref, "matcond").reshape(-1, 4)
    mfuns, num_m = set_material_fun(g, props, matcond, num_bc)

    u = _arr(ref, "u").ravel()
    data = ref["data"]
    args = (
        u, _arr(ref, "dd").ravel(), None, g, data, bcfun, num_bc,
        _arr(ref, "mat_facs").ravel(), None, mfuns.efun, mfuns.nufun,
        _arr(ref, "u_init").ravel(), _arr(ref, "u_sigma").ravel(),
        _arr(ref, "model_sigma").ravel(), float(ref["scaling"])
    )
    val, terms, simvals, datapts, all_simvals = gaussian_objfun(
        *args, ignore_prior=False
    )
    match_val, match_terms, *_ = gaussian_objfun(*args, ignore_prior=True)
    setup = setup_objective_fun(
        g, data, bcfun, mfuns, num_bc, _arr(ref, "mat_facs").ravel(),
        _arr(ref, "u_init").ravel(), _arr(ref, "u_sigma").ravel(),
        _arr(ref, "model_sigma").ravel(), float(ref["scaling"]), False
    )
    setup_val = setup.ofun(u, _arr(ref, "dd").ravel())[0]
    setup_match_val = setup.matchfun(u, _arr(ref, "dd").ravel())[0]

    failures = 0
    print("Phase 77: vemmech calibration Gaussian objective vs MATLAB MRST")
    failures += compare("theta_diff", theta_diff([0, 2.8], [3.0, 0.2]),
                        [0.14159265358979312, 0.541592653589793])
    failures += compare("num controls material", num_m, _arr(ref, "num_mctrls"),
                        rtol=0, atol=0)
    failures += compare("objective value", [val], [ref["val"]])
    failures += compare("match value", [match_val], [ref["match_val"]])
    failures += compare("setup ofun", [setup_val], [ref["val"]])
    failures += compare("setup matchfun", [setup_match_val], [ref["match_val"]])

    ref_terms = ref["terms"]
    failures += compare("terms r_theta", terms.r_theta,
                        _field(ref_terms, "r_theta"))
    failures += compare("terms r_Smax", terms.r_Smax,
                        _field(ref_terms, "r_Smax"))
    failures += compare("terms r_Smin", terms.r_Smin,
                        _field(ref_terms, "r_Smin"))
    failures += compare("terms r_Sv", terms.r_Sv, _field(ref_terms, "r_Sv"))
    failures += compare("terms r_prior", terms.r_prior,
                        _field(ref_terms, "r_prior"))
    failures += compare("match terms r_prior", match_terms.r_prior,
                        _field(ref["match_terms"], "r_prior"))

    ref_sim = ref["simvals"]
    failures += compare("simvals Smax", simvals.Smax, _field(ref_sim, "Smax"))
    failures += compare("simvals Smin", simvals.Smin, _field(ref_sim, "Smin"))
    failures += compare("simvals Sv", simvals.Sv, _field(ref_sim, "Sv"))
    failures += compare("simvals theta", simvals.theta,
                        _field(ref_sim, "theta"))
    failures += compare("datapts theta", datapts.theta,
                        _field(ref["datapts"], "theta"))
    failures += compare("all simvals Smax", all_simvals.Smax,
                        _field(ref["all_simvals"], "Smax"))

    if failures:
        print(f"\nvemmech objective validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST vemmech calibration Gaussian objective.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
