"""Introduction to Automatic Differentiation (MRST ``tutorialAD.m``).

Four demonstrations, in the order MRST presents them:

1. AD reproduces a derivative worked out by hand, for `z = 3*exp(-x*y)`.
2. AD assembles a linear system: three equations in three unknowns, where the
   Jacobian *is* the coefficient matrix and one solve gives the answer.
3. Newton's method on the Rosenbrock gradient, with AD supplying the Jacobian.
4. The same Newton solve with hand-coded derivatives, to show the answers agree.

MRST's version also times vectorised against looped AD to argue that
element-by-element loops are ruinous in MATLAB. That argument transfers to
Python but the numbers do not, so the timing table is replaced by
``compare_vectorised_and_looped``, which measures the same contrast here and is
reported rather than asserted: timings are machine-dependent and have no place
in a test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.utils.adi import ad_exp, init_variables_adi, vertcat_adi


def manual_vs_automatic(x0: float = 1.0, y0: float = 2.0):
    """`z = 3*exp(-x*y)`, differentiated by hand and by AD.

    By hand, `dz/dx = -3y*exp(-xy)` and `dz/dy = -3x*exp(-xy)`, so the value
    and both partials are `[1, -y, -x] * 3*exp(-x*y)`. Returns
    ``(manual, automatic)`` as three-element arrays holding value, `d/dx` and
    `d/dy`.
    """
    manual = np.array([1.0, -y0, -x0]) * 3.0 * np.exp(-x0 * y0)
    x, y = init_variables_adi(np.array([float(x0)]), np.array([float(y0)]))
    z = 3.0 * ad_exp(-(x * y))
    automatic = np.array([
        float(z.val[0]),
        float(z.jac[0].toarray()[0, 0]),
        float(z.jac[1].toarray()[0, 0]),
    ])
    return manual, automatic


def solve_linear_system_with_ad():
    """Three linear equations, solved through the AD Jacobian.

        3x0 + 2x1 - 4x2 + 5 = 0
        1x0 - 4x1 + 2x2 + 1 = 0
       -2x0 - 2x1 + 4x2 - 6 = 0

    Initialising the unknowns at zero makes each residual's value the constant
    term and its Jacobian the coefficient matrix, so `u = -J \\ r` solves the
    system in one step. Returns ``(u, jac, residual)``.
    """
    x = init_variables_adi(np.zeros(3))[0]
    a = np.array([[3.0, 2.0, -4.0],
                  [1.0, -4.0, 2.0],
                  [-2.0, -2.0, 4.0]])
    b = np.array([5.0, 1.0, -6.0])
    eqs = [_row(a[i], x) + b[i] for i in range(3)]
    eq = vertcat_adi(*eqs)
    jac = eq.jac[0]
    u = spla.spsolve(jac.tocsc(), -np.asarray(eq.val, dtype=float))
    return np.asarray(u, dtype=float), jac, np.asarray(eq.val, dtype=float)


def _row(coeffs, x):
    """`coeffs . x` for a length-3 AD vector, built from scalar slices."""
    return coeffs[0] * x[0] + coeffs[1] * x[1] + coeffs[2] * x[2]


@dataclass
class NewtonResult:
    x: np.ndarray
    iterations: int
    residual_norm: float


def rosenbrock_newton_ad(a: float = 1.0, b: float = 100.0,
                         x0=(-0.5, 4.0), tol: float = 1e-6,
                         max_it: int = 100) -> NewtonResult:
    """Find the stationary point of the Rosenbrock function using AD.

    The equations are the two components of the gradient,

        f0 = 2(a - x0) - 4b*x0*(x1 - x0^2)
        f1 = 2b*(x1 - x0^2)

    whose root is `(a, a^2)`. Newton needs the Jacobian, and AD supplies it
    without anyone writing the four partials out.
    """
    xv = np.asarray(x0, dtype=float).copy()
    incr = np.full(2, np.inf)
    it = 0
    while np.linalg.norm(incr) > tol and it < max_it:
        x = init_variables_adi(xv)[0]
        f0 = 2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] ** 2)
        f1 = 2.0 * b * (x[1] - x[0] ** 2)
        eq = vertcat_adi(f0, f1)
        incr = spla.spsolve(eq.jac[0].tocsc(), -np.asarray(eq.val, dtype=float))
        xv = xv + incr
        it += 1
    return NewtonResult(x=xv, iterations=it, residual_norm=float(np.linalg.norm(incr)))


def rosenbrock_newton_manual(a: float = 1.0, b: float = 100.0,
                             x0=(-0.5, 4.0), tol: float = 1e-6,
                             max_it: int = 100) -> NewtonResult:
    """The same Newton solve with the Jacobian written out by hand."""
    xv = np.asarray(x0, dtype=float).copy()
    incr = np.full(2, np.inf)
    it = 0
    while np.linalg.norm(incr) > tol and it < max_it:
        x, y = xv
        f = np.array([2.0 * (a - x) - 4.0 * b * x * (y - x ** 2),
                      2.0 * b * (y - x ** 2)])
        j = np.array([[-2.0 - 4.0 * b * (y - x ** 2) + 8.0 * b * x ** 2,
                       -4.0 * b * x],
                      [-4.0 * b * x, 2.0 * b]])
        incr = np.linalg.solve(j, -f)
        xv = xv + incr
        it += 1
    return NewtonResult(x=xv, iterations=it, residual_norm=float(np.linalg.norm(incr)))


def compare_vectorised_and_looped(n: int = 512) -> dict[str, float]:
    """Time `z = x*y` under AD, vectorised against element by element.

    MRST makes this point with a table over `n = 2^0 .. 2^13`. One size is
    enough to show the contrast, and the result is returned rather than
    asserted, since timings vary by machine and belong in a report, not a test.
    """
    xv, yv = np.random.default_rng(0).random(n), np.random.default_rng(1).random(n)

    t0 = time.perf_counter()
    _ = xv * yv
    t_analytic = time.perf_counter() - t0

    x, y = init_variables_adi(xv, yv)
    t0 = time.perf_counter()
    _ = x * y
    t_vector = time.perf_counter() - t0

    x, y = init_variables_adi(xv, yv)
    t0 = time.perf_counter()
    for k in range(n):
        _ = x[k] * y[k]
    t_loop = time.perf_counter() - t0

    return {"n": float(n), "analytic": t_analytic,
            "vectorised": t_vector, "looped": t_loop}


if __name__ == "__main__":
    def test_ad_reproduces_the_hand_derivative():
        manual, automatic = manual_vs_automatic()
        assert np.allclose(manual, automatic, rtol=1e-14), (manual, automatic)
        print("ok test_ad_reproduces_the_hand_derivative")

    def test_ad_jacobian_is_the_coefficient_matrix():
        """Initialised at zero, the Jacobian is exactly the coefficient matrix.

        That is what makes the one-step solve legitimate rather than a
        coincidence, so it is worth asserting directly.
        """
        u, jac, res = solve_linear_system_with_ad()
        a = np.array([[3.0, 2.0, -4.0], [1.0, -4.0, 2.0], [-2.0, -2.0, 4.0]])
        assert np.allclose(jac.toarray(), a), jac.toarray()
        assert np.allclose(res, [5.0, 1.0, -6.0]), res
        assert np.allclose(a @ u + np.array([5.0, 1.0, -6.0]), 0.0, atol=1e-12)
        print("ok test_ad_jacobian_is_the_coefficient_matrix")

    def test_rosenbrock_finds_the_known_root():
        """The gradient vanishes at `(a, a^2)`, which is (1, 1) by default."""
        out = rosenbrock_newton_ad()
        assert np.allclose(out.x, [1.0, 1.0], atol=1e-8), out.x
        assert out.iterations < 100, out.iterations
        print(f"ok test_rosenbrock_finds_the_known_root ({out.iterations} iterations)")

    def test_ad_and_hand_coded_jacobians_agree():
        """Both routes must reach the same root, and in the same way.

        Newton's path is determined by the Jacobian, so matching iteration
        counts is evidence the AD Jacobian equals the hand-written one rather
        than merely being good enough to converge.
        """
        auto = rosenbrock_newton_ad()
        hand = rosenbrock_newton_manual()
        assert np.allclose(auto.x, hand.x, atol=1e-10), (auto.x, hand.x)
        assert auto.iterations == hand.iterations, (auto.iterations, hand.iterations)
        print("ok test_ad_and_hand_coded_jacobians_agree")

    def test_a_different_rosenbrock_root():
        out = rosenbrock_newton_ad(a=2.0, b=50.0, x0=(0.0, 0.0))
        assert np.allclose(out.x, [2.0, 4.0], atol=1e-8), out.x
        print("ok test_a_different_rosenbrock_root")

    def test_timing_comparison_runs():
        t = compare_vectorised_and_looped(n=128)
        assert t["n"] == 128 and t["looped"] > 0.0
        print(f"ok test_timing_comparison_runs "
              f"(vectorised {t['vectorised']:.2e}s, looped {t['looped']:.2e}s)")

    test_ad_reproduces_the_hand_derivative()
    test_ad_jacobian_is_the_coefficient_matrix()
    test_rosenbrock_finds_the_known_root()
    test_ad_and_hand_coded_jacobians_agree()
    test_a_different_rosenbrock_root()
    test_timing_comparison_runs()
    print("\nAll tests passed!")
