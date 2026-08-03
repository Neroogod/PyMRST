"""ADI variable class, ported from MRST `ADI.m`."""

from typing import Sequence, Tuple

import numpy as np
import scipy.sparse as sp


class ADI:
    """Value vector plus a list of sparse Jacobian blocks."""

    __slots__ = ("val", "jac")
    __array_ufunc__ = None

    def __init__(self, val: np.ndarray, jac: Sequence[sp.csr_array]):
        self.val = np.atleast_1d(np.asarray(val, dtype=np.float64)).ravel()
        self.jac = [j.tocsr() for j in jac]

    def __len__(self) -> int:
        return self.val.size

    def __repr__(self) -> str:
        cols = [j.shape[1] for j in self.jac]
        return f"ADI(n={len(self)}, blocks={cols})"

    def copy(self) -> "ADI":
        return ADI(self.val.copy(), [j.copy() for j in self.jac])

    def _expand(self, n: int) -> "ADI":
        """Broadcast a scalar ADI to length `n` by row repetition."""
        assert len(self) == 1, "only scalar ADI can broadcast"
        return ADI(np.full(n, self.val[0]), [_row_repeat(j, n) for j in self.jac])

    @staticmethod
    def _align(u: "ADI", v: "ADI") -> Tuple["ADI", "ADI"]:
        if len(u) == len(v):
            return u, v
        if len(u) == 1:
            return u._expand(len(v)), v
        if len(v) == 1:
            return u, v._expand(len(u))
        raise ValueError(f"incompatible ADI lengths {len(u)}, {len(v)}")

    def __add__(self, other) -> "ADI":
        if isinstance(other, ADI):
            u, v = self._align(self, other)
            return ADI(u.val + v.val, [a + b for a, b in zip(u.jac, v.jac)])
        values = np.atleast_1d(np.asarray(other, dtype=np.float64)).ravel()
        u = self if values.size == 1 or values.size == len(self) or len(self) > 1 else self._expand(values.size)
        return ADI(u.val + values, [j.copy() for j in u.jac])

    __radd__ = __add__

    def __neg__(self) -> "ADI":
        return ADI(-self.val, [-j for j in self.jac])

    def __sub__(self, other) -> "ADI":
        return self + (-other if isinstance(other, ADI) else -np.asarray(other))

    def __rsub__(self, other) -> "ADI":
        return (-self) + other

    def __mul__(self, other) -> "ADI":
        if isinstance(other, ADI):
            u, v = self._align(self, other)
            jac = [
                _diag_mul(v.val, a) + _diag_mul(u.val, b)
                for a, b in zip(u.jac, v.jac)
            ]
            return ADI(u.val * v.val, jac)
        values = np.atleast_1d(np.asarray(other, dtype=np.float64)).ravel()
        if values.size == 1:
            return ADI(self.val * values[0], [j * values[0] for j in self.jac])
        u = self if values.size == len(self) or len(self) > 1 else self._expand(values.size)
        return ADI(u.val * values, [_diag_mul(values, j) for j in u.jac])

    __rmul__ = __mul__

    def __truediv__(self, other) -> "ADI":
        if isinstance(other, ADI):
            return self * other ** -1.0
        return self * (1.0 / np.asarray(other, dtype=np.float64))

    def __rtruediv__(self, other) -> "ADI":
        return self ** -1.0 * other

    def __pow__(self, k) -> "ADI":
        power = float(k)
        derivative = power * self.val ** (power - 1.0)
        return ADI(self.val**power, [_diag_mul(derivative, j) for j in self.jac])

    def __rmatmul__(self, mat) -> "ADI":
        from pymrst.utils.adi.operators import ad_matmul

        return ad_matmul(mat, self)

    def __lt__(self, other):
        from pymrst.utils.adi.value import value

        return self.val < value(other)

    def __le__(self, other):
        from pymrst.utils.adi.value import value

        return self.val <= value(other)

    def __gt__(self, other):
        from pymrst.utils.adi.value import value

        return self.val > value(other)

    def __ge__(self, other):
        from pymrst.utils.adi.value import value

        return self.val >= value(other)

    def __getitem__(self, idx) -> "ADI":
        selected = np.atleast_1d(np.arange(len(self))[idx])
        return ADI(self.val[selected], [j[selected] for j in self.jac])

    def __setitem__(self, idx, val) -> None:
        selected = np.atleast_1d(np.arange(len(self))[idx])
        keep = np.ones(len(self))
        keep[selected] = 0.0
        if isinstance(val, ADI):
            new_val = val if len(val) == selected.size else val._expand(selected.size)
            projector = sp.csr_array(
                (np.ones(selected.size), (selected, np.arange(selected.size))),
                shape=(len(self), selected.size),
            )
            self.jac = [
                _diag_mul(keep, old) + projector @ new
                for old, new in zip(self.jac, new_val.jac)
            ]
            self.val[selected] = new_val.val
        else:
            self.jac = [_diag_mul(keep, old) for old in self.jac]
            self.val[selected] = val


def _row_repeat(jac: sp.csr_array, n: int) -> sp.csr_array:
    """Repeat the single row of a `(1, m)` Jacobian block `n` times."""
    repeater = sp.csr_array(
        (np.ones(n), (np.arange(n), np.zeros(n, dtype=np.int64))), shape=(n, 1)
    )
    return (repeater @ jac).tocsr()


def _diag_mul(d: np.ndarray, jac: sp.csr_array) -> sp.csr_array:
    """Return `diag(d) @ jac` without forming the diagonal matrix."""
    out = jac.tocsr(copy=True)
    out.data *= np.repeat(d, np.diff(out.indptr))
    return out


if __name__ == "__main__":
    from pymrst.utils.adi.init_variables_adi import init_variables_adi

    def _numerical_jacobian(f, x, h=1e-7):
        """Central-difference Jacobian of `f` at `x`.

        The reference every test below compares against. A hand-written
        analytic check would only restate the chain rule this module
        implements; a finite difference is derived independently of it, so a
        wrong derivative shows up as a mismatch rather than agreeing with
        itself.
        """
        x = np.asarray(x, dtype=float)
        base = np.atleast_1d(f(x))
        jac = np.zeros((base.size, x.size))
        for j in range(x.size):
            step = np.zeros_like(x)
            step[j] = h * max(1.0, abs(x[j]))
            jac[:, j] = (np.atleast_1d(f(x + step)) - np.atleast_1d(f(x - step))) / (2 * step[j])
        return jac

    def _check(name, expression, plain, x, rtol=1e-6):
        (u,) = init_variables_adi(np.asarray(x, dtype=float))
        result = expression(u)
        analytic = result.jac[0].toarray()
        numeric = _numerical_jacobian(plain, x)
        assert analytic.shape == numeric.shape, (name, analytic.shape, numeric.shape)
        scale = np.maximum(np.abs(numeric), 1e-8)
        worst = float(np.max(np.abs(analytic - numeric) / scale))
        assert worst < rtol, f"{name}: worst relative error {worst:.3e}"
        assert np.allclose(result.val, plain(np.asarray(x, dtype=float))), name
        return worst

    def test_arithmetic_derivatives_match_finite_differences():
        """Every operator, against an independently derived reference."""
        x = np.array([0.7, 1.3, 2.5])
        cases = [
            ("add", lambda u: u + u, lambda v: v + v),
            ("sub", lambda u: u - 3.0 * u, lambda v: v - 3.0 * v),
            ("neg", lambda u: -u, lambda v: -v),
            ("mul by scalar", lambda u: 2.5 * u, lambda v: 2.5 * v),
            ("product rule", lambda u: u * u, lambda v: v * v),
            ("quotient rule", lambda u: u / (u + 1.0), lambda v: v / (v + 1.0)),
            ("reciprocal", lambda u: 1.0 / u, lambda v: 1.0 / v),
            ("power", lambda u: u ** 3, lambda v: v ** 3),
            ("fractional power", lambda u: u ** 0.5, lambda v: v ** 0.5),
            ("nested", lambda u: (u * u + 1.0) / (u ** 0.5), lambda v: (v * v + 1.0) / (v ** 0.5)),
        ]
        worst = max(_check(name, expr, plain, x) for name, expr, plain in cases)
        print(f"ok test_arithmetic_derivatives_match_finite_differences "
              f"({len(cases)} operators, worst relative error {worst:.2e})")

    def test_the_product_rule_is_not_just_elementwise():
        """u*u must differentiate to 2u, not to u.

        Multiplying values and Jacobians elementwise gives the right *value*
        and a Jacobian that is wrong by a factor of two. Nothing raises.
        """
        x = np.array([3.0])
        (u,) = init_variables_adi(x)
        assert np.allclose((u * u).jac[0].toarray(), [[6.0]]), (u * u).jac[0].toarray()
        print("ok test_the_product_rule_is_not_just_elementwise")

    def test_blocks_stay_separate():
        """Two primary variables keep independent Jacobian blocks."""
        u, v = init_variables_adi(np.array([2.0, 3.0]), np.array([5.0, 7.0]))
        w = u * v
        assert len(w.jac) == 2
        assert np.allclose(w.jac[0].toarray(), np.diag([5.0, 7.0]))   # d(uv)/du = v
        assert np.allclose(w.jac[1].toarray(), np.diag([2.0, 3.0]))   # d(uv)/dv = u
        print("ok test_blocks_stay_separate")

    def test_scalar_broadcasts_without_losing_derivatives():
        """A length-1 ADI meeting a vector must expand, not silently truncate.

        Both are declared in one `init_variables_adi` call. Declaring them
        separately gives each a single block sized to itself, and combining
        them then fails with `inconsistent shapes` from scipy - which is the
        API working as intended, not a defect: variables that appear in the
        same expression have to share a block layout.
        """
        a, b = init_variables_adi(np.array([4.0]), np.array([1.0, 2.0, 3.0]))
        out = a * b
        assert len(out) == 3, len(out)
        assert np.allclose(out.val, [4.0, 8.0, 12.0])
        # d(ab)/da = b, in a's block; d(ab)/db = a, in b's block.
        assert out.jac[0].shape == (3, 1), out.jac[0].shape
        assert np.allclose(out.jac[0].toarray().ravel(), [1.0, 2.0, 3.0])
        assert np.allclose(out.jac[1].toarray(), np.diag([4.0, 4.0, 4.0]))
        print("ok test_scalar_broadcasts_without_losing_derivatives")

    def test_matrix_product_carries_the_jacobian():
        """Via `ad_matmul`, which is the supported spelling.

        `mat @ adi` does *not* reach `__rmatmul__`: scipy's sparse `__matmul__`
        runs first and tries to coerce the ADI into an array rather than
        returning NotImplemented, so it dies inside numpy. `__array_ufunc__ =
        None` blocks numpy ufuncs but not scipy's matmul dispatch. That is why
        `pymrst.utils.adi.operators.ad_matmul` exists, and callers must use it.
        """
        from pymrst.utils.adi.operators import ad_matmul

        (u,) = init_variables_adi(np.array([1.0, 2.0, 3.0]))
        m = sp.csr_array(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))
        out = ad_matmul(m, u)
        assert np.allclose(out.val, [7.0, 6.0])
        assert np.allclose(out.jac[0].toarray(), m.toarray())
        # And the direct spelling is genuinely unsupported, not merely unused.
        try:
            m @ u
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("mat @ adi now works; ad_matmul may be redundant")
        print("ok test_matrix_product_carries_the_jacobian (via ad_matmul)")

    def test_indexing_selects_rows_of_the_jacobian_too():
        """A slice that keeps the values and drops the derivatives is silent."""
        (u,) = init_variables_adi(np.array([1.0, 2.0, 3.0, 4.0]))
        w = (u * u)[1:3]
        assert np.allclose(w.val, [4.0, 9.0])
        assert w.jac[0].shape == (2, 4), w.jac[0].shape
        assert np.allclose(w.jac[0].toarray(), [[0, 4, 0, 0], [0, 0, 6, 0]])
        print("ok test_indexing_selects_rows_of_the_jacobian_too")

    def test_comparisons_use_values_not_objects():
        (u,) = init_variables_adi(np.array([1.0, 5.0]))
        assert list(u > 2.0) == [False, True]
        assert list(u <= 5.0) == [True, True]
        print("ok test_comparisons_use_values_not_objects")

    def test_copy_is_deep_enough_to_be_safe():
        (u,) = init_variables_adi(np.array([1.0, 2.0]))
        w = (u * 3.0).copy()
        before = w.jac[0].toarray().copy()
        u.val[0] = 99.0
        assert np.allclose(w.jac[0].toarray(), before)
        print("ok test_copy_is_deep_enough_to_be_safe")

    test_arithmetic_derivatives_match_finite_differences()
    test_the_product_rule_is_not_just_elementwise()
    test_blocks_stay_separate()
    test_scalar_broadcasts_without_losing_derivatives()
    test_matrix_product_carries_the_jacobian()
    test_indexing_selects_rows_of_the_jacobian_too()
    test_comparisons_use_values_not_objects()
    test_copy_is_deep_enough_to_be_safe()
    print("\nAll tests passed!")
