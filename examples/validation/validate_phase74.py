"""Phase 74: vemmech calibration helpers vs MATLAB MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.calibration import (
    identify_boundary_elements,
    set_material_fun,
    sigma_to_smax_smin_theta,
)
from pymrst.modules.vemmech.vem import create_augmented_grid

HERE = Path(__file__).parent
REF = HERE / "ref_p74_vemmech_calibration.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _mat_cell_array(x):
    arr = np.asarray(x, dtype=object).ravel()
    return [np.asarray(v, dtype=float).ravel() for v in arr]


def compare(name, py, ml, rtol=1e-12, atol=1e-12) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ml))) if ml.size else 0.0, 1.0)
    diff = 0.0 if py.size == ml.size == 0 else float(np.max(np.abs(py - ml)))
    ok = py.shape == ml.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def compare_side_lists(name, py_lists, ml_obj) -> int:
    failures = 0
    ml_lists = _mat_cell_array(ml_obj)
    for i, (py, ml) in enumerate(zip(py_lists, ml_lists), start=1):
        failures += compare(f"{name} side {i}", np.asarray(py) + 1, ml,
                            rtol=0, atol=0)
    return failures


def _load_matrix(tensor, g):
    return _dense(tensor.as_matrix([["cn"], ["d"]], True,
                                   [g.cells.nodes.size, g.griddim]))


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase74_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True, struct_as_record=False)

    g = create_augmented_grid(
        compute_geometry(cart_grid([2, 2, 3], [2.0, 2.0, 3.0]))
    )
    props = SimpleNamespace(
        young=_arr(ref, "props_young").ravel(),
        poisson=_arr(ref, "props_poisson").ravel(),
        rho=_arr(ref, "props_rho").ravel(),
    )

    failures = 0
    print("Phase 74: vemmech calibration helpers vs MATLAB MRST")

    smax, smin, theta = sigma_to_smax_smin_theta(
        _arr(ref, "o_xx"), _arr(ref, "o_yy"), _arr(ref, "o_xy")
    )
    failures += compare("Sigma Smax", smax, _arr(ref, "Smax"))
    failures += compare("Sigma Smin", smin, _arr(ref, "Smin"))
    failures += compare("Sigma theta", theta, _arr(ref, "theta"))

    binfo = identify_boundary_elements(g)
    failures += compare_side_lists("boundary faces", binfo.side_faces,
                                   ref["side_faces"])
    failures += compare_side_lists("boundary nodes", binfo.side_nodes,
                                   ref["side_nodes"])

    funs0, num0 = set_material_fun(g, props, None, 0)
    failures += compare("num controls none", num0, _arr(ref, "num0"),
                        rtol=0, atol=0)
    failures += compare("E no controls", funs0.efun([]), _arr(ref, "E0"))
    failures += compare("nu no controls", funs0.nufun([]), _arr(ref, "nu0"))
    failures += compare("load no controls", _load_matrix(funs0.loadfun([]), g),
                        _arr(ref, "L0"))

    matcond = _arr(ref, "matcond")
    u = _arr(ref, "u").ravel()
    funs, num = set_material_fun(g, props, matcond, 0)
    failures += compare("num controls", num, _arr(ref, "num"),
                        rtol=0, atol=0)
    failures += compare("E controls", funs.efun(u), _arr(ref, "E"))
    failures += compare("nu controls", funs.nufun(u), _arr(ref, "nu"))
    failures += compare("load controls", _load_matrix(funs.loadfun(u), g),
                        _arr(ref, "L"))

    if failures:
        print(f"\nvemmech calibration validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST vemmech calibration helpers.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
