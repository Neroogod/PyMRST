"""Package self-tests for `pymrst.utils.adi`."""

import numpy as np
import scipy.sparse as sp

from pymrst.utils.adi import (
    ad_exp,
    ad_log,
    ad_matmul,
    ad_max,
    ad_min,
    ad_sum,
    combine_equations,
    hessian_ad2,
    init_variables_adi,
)


def _fd_check(fn, x0: np.ndarray, tol: float = 3e-6) -> None:
    """Compare an ADI Jacobian against central differences."""
    (x,) = init_variables_adi(x0)
    jac = fn(x).jac[0].toarray()
    step = 1e-7 * max(1.0, np.abs(x0).max())
    fd = np.zeros_like(jac)
    for col in range(x0.size):
        basis = np.zeros_like(x0)
        basis[col] = step
        fd[:, col] = (fn(x0 + basis) - fn(x0 - basis)) / (2 * step)
    assert np.allclose(jac, fd, rtol=tol, atol=tol), (
        f"max diff {np.abs(jac - fd).max():.2e}"
    )


def test_arithmetic_fd() -> None:
    """Composite arithmetic expression matches finite differences."""
    x0 = np.array([0.6, 1.2, 2.5, 0.3])

    def fn(x):
        return (
            2.0 * x**2
            - x * ad_exp(-x)
            + 1.0 / (x + 3.0)
            - ad_log(x) * 0.1
            + np.arange(1.0, 5.0) * x
        )

    _fd_check(fn, x0)
    print("ok test_arithmetic_fd")


def test_scalar_broadcast() -> None:
    """Scalar ADI variables broadcast against vectors."""
    p, b = init_variables_adi(np.array([1.0, 2.0]), 3.0)
    result = p + b * np.array([10.0, 20.0])
    assert np.allclose(result.val, [31.0, 62.0])
    assert np.allclose(result.jac[1].toarray().ravel(), [10.0, 20.0])
    reverse = b - p
    assert np.allclose(reverse.val, [2.0, 1.0])
    assert np.allclose(reverse.jac[0].toarray(), -np.eye(2))
    print("ok test_scalar_broadcast")


def test_indexing_and_setitem() -> None:
    """Fancy indexing extracts rows; setitem replaces value and Jacobian rows."""
    p, q = init_variables_adi(np.array([1.0, 2.0, 3.0, 4.0]), 7.0)
    sub = p[np.array([1, 3])]
    assert np.allclose(sub.val, [2.0, 4.0])
    assert np.allclose(sub.jac[0].toarray(), [[0, 1, 0, 0], [0, 0, 0, 1]])
    eq = p * 2.0
    eq[np.array([0, 2])] = eq[np.array([0, 2])] - q
    assert np.allclose(eq.val, [2.0 - 7.0, 4.0, 6.0 - 7.0, 8.0])
    assert np.allclose(eq.jac[1].toarray().ravel(), [-1, 0, -1, 0])
    assert np.allclose(eq.jac[0].toarray(), np.diag([2.0, 2.0, 2.0, 2.0]))
    print("ok test_indexing_and_setitem")


def test_matmul_div_sum() -> None:
    """Sparse products, division, and sum propagate Jacobians."""
    (x,) = init_variables_adi(np.array([1.0, 2.0, 4.0]))
    mat = sp.csr_array(np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]))
    result = ad_matmul(mat, x**2)
    assert np.allclose(result.val, [1.0 - 4.0, 4.0 - 16.0])
    assert np.allclose(result.jac[0].toarray(), [[2.0, -4.0, 0.0], [0.0, 4.0, -8.0]])
    summed = ad_sum(1.0 / x)
    assert np.isclose(summed.val[0], 1.75)
    assert np.allclose(summed.jac[0].toarray().ravel(), [-1.0, -0.25, -1.0 / 16.0])
    print("ok test_matmul_div_sum")


def test_combine_equations() -> None:
    """Equation combination assembles one full block Jacobian."""
    p, b = init_variables_adi(np.array([1.0, 2.0]), 5.0)
    eq = combine_equations(p * 3.0 + b, b * 2.0)
    assert len(eq.jac) == 1
    assert eq.jac[0].shape == (3, 3)
    assert np.allclose(eq.jac[0].toarray(), [[3, 0, 1], [0, 3, 1], [0, 0, 2]])
    assert np.allclose(eq.val, [8.0, 11.0, 10.0])
    print("ok test_combine_equations")


def test_max_min() -> None:
    """Branching min/max use the winning branch derivative."""
    (x,) = init_variables_adi(np.array([-2.0, 3.0]))
    maxed = ad_max(0.0, x)
    assert np.allclose(maxed.val, [0.0, 3.0])
    assert np.allclose(maxed.jac[0].toarray(), [[0, 0], [0, 1]])
    mined = ad_min(x, 1.0)
    assert np.allclose(mined.val, [-2.0, 1.0])
    assert np.allclose(mined.jac[0].toarray(), [[1, 0], [0, 0]])
    print("ok test_max_min")


def test_second_order_hessian() -> None:
    """ADI2 computes exact Hessians for scalar expressions."""
    def fun(x):
        return x[0] * x[1] + x[0] ** 3 + 2.0 * x[1] ** 2

    val, grad, hess = hessian_ad2(fun, np.array([2.0, 3.0]))
    assert np.isclose(val, 32.0)
    assert np.allclose(grad, [15.0, 14.0])
    assert np.allclose(hess, [[12.0, 1.0], [1.0, 4.0]])
    print("ok test_second_order_hessian")


test_arithmetic_fd()
test_scalar_broadcast()
test_indexing_and_setitem()
test_matmul_div_sum()
test_combine_equations()
test_max_min()
test_second_order_hessian()
print("\nAll tests passed!")
