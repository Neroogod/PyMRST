"""Phase 95: lazy tensor graph and exact second-order AD substrate."""

import numpy as np
import scipy.sparse as sp

from pymrst.modules.vemmech.ad_version import (
    SparseMultiArray,
    lazy_sparse_tensor,
    tbinaryop,
    tcontract,
)
from pymrst.utils.adi import ad2_matmul, ad2_sum, hessian_ad2, init_variables_adi


def compare(name, py, ref, rtol=1e-11, atol=1e-11) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0)
    diff = 0.0 if py.size == ref.size == 0 else float(np.max(np.abs(py - ref)))
    ok = py.shape == ref.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    failures = 0
    print("Phase 95: lazy tensor graph and exact second-order AD substrate")

    a = SparseMultiArray([2.0, 3.0], [[1, 1], [1, 2]], ["i", "j"])
    b = SparseMultiArray([5.0, 7.0], [[1, 1], [2, 1]], ["j", "k"])
    c = SparseMultiArray([11.0], [[1, 1]], ["i", "k"])
    lazy_expr = tcontract(a, b) + lazy_sparse_tensor(c)
    eager = (a * b) + c
    failures += compare("lazy contraction value",
                        lazy_expr.as_full_vector(["i", "k"], [1, 1]),
                        eager.as_full_vector(["i", "k"], [1, 1]))
    graph = lazy_expr.graph()
    failures += compare("lazy graph root is add", [graph["op"] == "add"], [1],
                        rtol=0, atol=0)
    failures += compare("lazy graph child is contraction",
                        [graph["args"][0]["op"] == "mul"], [1],
                        rtol=0, atol=0)

    (x,) = init_variables_adi(np.array([2.0, 3.0]))
    ad_tensor = SparseMultiArray(x, [[1], [2]], ["j"])
    lazy_ad = tbinaryop("*", ad_tensor, b).as_full_vector(["k"], [1])
    failures += compare("lazy ADI contraction value", lazy_ad.val, [31.0])
    failures += compare("lazy ADI contraction jacobian",
                        lazy_ad.jac[0].toarray(), [[5.0, 7.0]])

    semi = tbinaryop("^", SparseMultiArray([2.0], [[1]], ["i"]),
                     SparseMultiArray([3.0], [[1]], ["j"]))
    failures += compare("lazy semiproduct",
                        semi.as_full_vector(["i", "j"], [1, 1]), [6.0])

    def scalar_fun(z):
        q = z[0] * z[1] + z[0] ** 3 + 2.0 * z[1] ** 2
        projected = ad2_matmul(sp.csr_array([[2.0]]), q)
        return ad2_sum(projected)

    val, grad, hess = hessian_ad2(scalar_fun, np.array([2.0, 3.0]))
    failures += compare("ADI2 scalar value", [val], [64.0])
    failures += compare("ADI2 exact gradient", grad, [30.0, 28.0])
    failures += compare("ADI2 exact Hessian", hess,
                        [[24.0, 2.0], [2.0, 8.0]])

    if failures:
        print(f"\nLazy tensor / exact Hessian validation failed "
              f"({failures} checks).")
    else:
        print("\nPyMRST has a pure-Python lazy tensor graph and exact "
              "second-order AD substrate.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
