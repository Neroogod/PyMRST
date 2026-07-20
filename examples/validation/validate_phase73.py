"""Phase 73: vemmech AD tensor utilities vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import (
    SparseMultiArray,
    enu2c_ad,
    face_node_tensor,
    kron_adi,
)
from pymrst.modules.vemmech.vem import enu2c

HERE = Path(__file__).parent
REF = HERE / "ref_p73_vemmech_ad_utils.mat"


def _arr(ref, key):
    return np.asarray(ref[key], dtype=float)


def compare(name, py, ml, rtol=1e-12, atol=1e-12) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ml))), 1.0)
    diff = float(np.max(np.abs(py - ml)))
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase73_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    t = SparseMultiArray([2, 3], [[1, 2], [2, 1]], ["i", "j"])

    g = compute_geometry(cart_grid([1, 1, 1], [1.0, 2.0, 3.0]))
    vals = np.arange(1, g.faces.nodes.size + 1, dtype=float)
    E = np.full(g.cells.num, 1e9)
    nu = np.full(g.cells.num, 0.25)
    C_ad = enu2c_ad(E, nu, g)
    C = enu2c(E, nu)

    failures = 0
    print("Phase 73: vemmech AD tensor utilities vs MATLAB MRST")
    failures += compare("SparseMultiArray matrix",
                        _dense(t.as_matrix([["i"], ["j"]], True, [2, 2])),
                        _arr(ref, "t_mat"), rtol=0, atol=0)
    failures += compare("SparseMultiArray vector ji",
                        t.as_full_vector(["j", "i"], [2, 2]),
                        _arr(ref, "t_vec_ji"), rtol=0, atol=0)
    failures += compare("kron_ADI numeric", kron_adi([2, 3], [5, 7, 11]),
                        _arr(ref, "kron_ref"), rtol=0, atol=0)

    fnt = face_node_tensor(g, "f", "n")
    failures += compare("face_node_tensor indicator",
                        _dense(fnt.as_matrix([["f"], ["n"]], True,
                                             [g.faces.num, g.nodes.num])),
                        _arr(ref, "fnt_mat"), rtol=0, atol=0)
    fnt_val = face_node_tensor(g, "f", "n", values=vals)
    failures += compare("face_node_tensor values",
                        _dense(fnt_val.as_matrix([["f"], ["n"]], True,
                                                 [g.faces.num, g.nodes.num])),
                        _arr(ref, "fnt_val_mat"), rtol=0, atol=0)
    fnt_b = face_node_tensor(g, "f", "n", boundary_only=True)
    failures += compare("face_node_tensor boundary",
                        _dense(fnt_b.as_matrix([["f"], ["n"]], True,
                                               [g.faces.num, g.nodes.num])),
                        _arr(ref, "fnt_b_mat"), rtol=0, atol=0)
    failures += compare("Enu2C_AD vector vs MRST Enu2C",
                        C_ad.as_full_vector(["i", "j", "c"],
                                            [6, 6, g.cells.num]),
                        _arr(ref, "C"), rtol=1e-14)

    if failures:
        print(f"\nvemmech AD utility validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST vemmech AD tensor utilities.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
