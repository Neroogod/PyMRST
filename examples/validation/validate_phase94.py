"""Phase 94: ADI coefficient propagation in vemmech sparse tensors."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import (
    SparseMultiArray,
    enu2c_ad,
    face_node_tensor,
    kron_adi,
)
from pymrst.utils.adi import ADI, init_variables_adi, value


def compare(name, py, ref, rtol=1e-10, atol=1e-10) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0)
    diff = 0.0 if py.size == ref.size == 0 else float(np.max(np.abs(py - ref)))
    ok = py.shape == ref.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def _adi_jac(x: ADI, block: int = 0):
    return x.jac[block].toarray()


def _numeric_enu_vector(E, nu, g):
    return value(enu2c_ad(E, nu, g).as_full_vector(["i", "j", "c"],
                                                   [6, 6, g.cells.num]))


def main() -> int:
    failures = 0
    print("Phase 94: ADI coefficient propagation in vemmech sparse tensors")

    (x,) = init_variables_adi(np.array([2.0, 3.0]))
    a = SparseMultiArray(x, [[1], [2]], ["i"])
    b = SparseMultiArray([5.0, 7.0], [[1, 1], [2, 1]], ["i", "j"])
    contracted = (a * b).as_full_vector(["j"], [1])
    failures += compare("contracted value", contracted.val, [31.0])
    failures += compare("contracted jacobian", _adi_jac(contracted),
                        [[5.0, 7.0]])

    doubled = (a + a).as_full_vector(["i"], [2])
    failures += compare("duplicate add value", doubled.val, [4.0, 6.0])
    failures += compare("duplicate add jacobian", _adi_jac(doubled),
                        [[2.0, 0.0], [0.0, 2.0]])

    semi = a.semiproduct(SparseMultiArray([10.0], [[1]], ["j"]))
    semi_vec = semi.as_full_vector(["i", "j"], [2, 1])
    failures += compare("semiproduct value", semi_vec.val, [20.0, 30.0])
    failures += compare("semiproduct jacobian", _adi_jac(semi_vec),
                        [[10.0, 0.0], [0.0, 10.0]])

    kroned = kron_adi(x, [10.0, 20.0])
    failures += compare("kron_ADI value", kroned.val,
                        [20.0, 40.0, 30.0, 60.0])
    failures += compare("kron_ADI jacobian", _adi_jac(kroned),
                        [[10.0, 0.0], [20.0, 0.0],
                         [0.0, 10.0], [0.0, 20.0]])

    g = compute_geometry(cart_grid([1, 1, 1], [1.0, 2.0, 3.0]))
    (fn_vals,) = init_variables_adi(
        np.linspace(1.0, float(g.faces.nodes.size), g.faces.nodes.size)
    )
    fnt = face_node_tensor(g, "f", "n", values=fn_vals)
    fnt_vec = fnt.as_full_vector(["f", "n"], [g.faces.num, g.nodes.num])
    fnt_ref = face_node_tensor(
        g, "f", "n", values=np.linspace(1.0, float(g.faces.nodes.size),
                                       g.faces.nodes.size)
    )
    fnt_ref_vec = fnt_ref.as_full_vector(["f", "n"],
                                         [g.faces.num, g.nodes.num])
    rows = ((fnt.ixs[:, 0] - 1)
            + (fnt.ixs[:, 1] - 1) * g.faces.num).astype(int)
    cols = (fnt.coefs.val - 1).astype(int)
    expected_fnt_jac = np.zeros((g.faces.num * g.nodes.num,
                                 g.faces.nodes.size))
    expected_fnt_jac[rows, cols] = 1.0
    failures += compare("face_node_tensor ADI values",
                        fnt_vec.val, fnt_ref_vec)
    failures += compare("face_node_tensor ADI jacobian",
                        fnt_vec.jac[0].toarray(), expected_fnt_jac)

    E0 = np.array([1.2e9])
    nu0 = np.array([0.23])
    E, nu = init_variables_adi(E0, nu0)
    C = enu2c_ad(E, nu, g).as_full_vector(["i", "j", "c"],
                                          [6, 6, g.cells.num])
    failures += compare("Enu2C_AD value", C.val,
                        _numeric_enu_vector(E0, nu0, g), rtol=1e-13)

    hE = 1e-5 * E0[0]
    hnu = 1e-6
    fd_E = (_numeric_enu_vector(E0 + hE, nu0, g)
            - _numeric_enu_vector(E0 - hE, nu0, g)) / (2.0 * hE)
    fd_nu = (_numeric_enu_vector(E0, nu0 + hnu, g)
             - _numeric_enu_vector(E0, nu0 - hnu, g)) / (2.0 * hnu)
    failures += compare("Enu2C_AD d/dE", C.jac[0].toarray(), fd_E[:, None],
                        rtol=1e-8, atol=1e-8)
    failures += compare("Enu2C_AD d/dnu", C.jac[1].toarray(), fd_nu[:, None],
                        rtol=1e-8, atol=1e-2)

    if failures:
        print(f"\nvemmech ADI tensor propagation failed ({failures} checks).")
    else:
        print("\nPyMRST propagates first-order ADI coefficients through the "
              "eager vemmech tensor substrate.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
