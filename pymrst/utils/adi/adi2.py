"""Second-order automatic differentiation helpers.

This is a compact forward-mode AD class for exact Hessians of vector
expressions with respect to one flattened control vector. It complements the
MRST-style first-order ``ADI`` class without changing its public contract.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp


class ADI2:
    """Value, Jacobian, and exact per-output Hessian blocks."""

    __slots__ = ("val", "jac", "hess")
    __array_ufunc__ = None

    def __init__(self, val, jac, hess: Sequence[sp.csr_array]):
        self.val = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
        self.jac = jac.tocsr()
        self.hess = [h.tocsr() for h in hess]
        if self.jac.shape[0] != self.val.size:
            raise ValueError("ADI2 Jacobian row count must match value size")
        if len(self.hess) != self.val.size:
            raise ValueError("ADI2 must have one Hessian block per value")

    def __len__(self) -> int:
        return self.val.size

    def __repr__(self) -> str:
        return f"ADI2(n={len(self)}, controls={self.jac.shape[1]})"

    def copy(self) -> "ADI2":
        return ADI2(self.val.copy(), self.jac.copy(),
                    [h.copy() for h in self.hess])

    @property
    def num_controls(self) -> int:
        return self.jac.shape[1]

    def _expand(self, n: int) -> "ADI2":
        if len(self) != 1:
            raise ValueError("only scalar ADI2 can broadcast")
        repeater = sp.csr_array(
            (np.ones(n), (np.arange(n), np.zeros(n, dtype=np.int64))),
            shape=(n, 1),
        )
        return ADI2(
            np.full(n, self.val[0]),
            repeater @ self.jac,
            [self.hess[0].copy() for _ in range(n)],
        )

    @staticmethod
    def _align(u, v) -> tuple["ADI2", "ADI2"]:
        if not isinstance(u, ADI2):
            u = _as_adi2_constant(u, v.num_controls)
        if not isinstance(v, ADI2):
            v = _as_adi2_constant(v, u.num_controls)
        if u.num_controls != v.num_controls:
            raise ValueError("ADI2 operands must use the same controls")
        if len(u) == len(v):
            return u, v
        if len(u) == 1:
            return u._expand(len(v)), v
        if len(v) == 1:
            return u, v._expand(len(u))
        raise ValueError(f"incompatible ADI2 lengths {len(u)}, {len(v)}")

    def __add__(self, other) -> "ADI2":
        u, v = ADI2._align(self, other)
        return ADI2(u.val + v.val, u.jac + v.jac,
                    [a + b for a, b in zip(u.hess, v.hess)])

    __radd__ = __add__

    def __neg__(self) -> "ADI2":
        return ADI2(-self.val, -self.jac, [-h for h in self.hess])

    def __sub__(self, other) -> "ADI2":
        return self + (-other if isinstance(other, ADI2) else -np.asarray(other))

    def __rsub__(self, other) -> "ADI2":
        return (-self) + other

    def __mul__(self, other) -> "ADI2":
        u, v = ADI2._align(self, other)
        jac = _diag_mul(v.val, u.jac) + _diag_mul(u.val, v.jac)
        hess = []
        for i in range(len(u)):
            gu = _row_dense(u.jac, i)
            gv = _row_dense(v.jac, i)
            cross = sp.csr_array(np.outer(gu, gv) + np.outer(gv, gu))
            hess.append(v.val[i] * u.hess[i] + u.val[i] * v.hess[i] + cross)
        return ADI2(u.val * v.val, jac, hess)

    __rmul__ = __mul__

    def __truediv__(self, other) -> "ADI2":
        return self * (other ** -1.0 if isinstance(other, ADI2)
                       else np.asarray(other, dtype=float) ** -1.0)

    def __rtruediv__(self, other) -> "ADI2":
        return (self ** -1.0) * other

    def __pow__(self, k) -> "ADI2":
        power = float(k)
        first = power * self.val ** (power - 1.0)
        second = power * (power - 1.0) * self.val ** (power - 2.0)
        jac = _diag_mul(first, self.jac)
        hess = []
        for i in range(len(self)):
            g = _row_dense(self.jac, i)
            hess.append(first[i] * self.hess[i]
                        + sp.csr_array(second[i] * np.outer(g, g)))
        return ADI2(self.val ** power, jac, hess)

    def __rmatmul__(self, mat) -> "ADI2":
        return ad2_matmul(mat, self)

    def __getitem__(self, idx) -> "ADI2":
        selected = np.atleast_1d(np.arange(len(self))[idx])
        return ADI2(self.val[selected], self.jac[selected],
                    [self.hess[int(i)].copy() for i in selected])


def init_variables_adi2(values) -> ADI2:
    """Declare one flattened second-order primary variable."""
    val = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    n = val.size
    return ADI2(val, sp.eye_array(n, format="csr"),
                [sp.csr_array((n, n)) for _ in range(n)])


def ad2_matmul(mat, x):
    """Matrix product for ``ADI2`` or plain arrays."""
    if not isinstance(x, ADI2):
        return mat @ x
    mat = mat.tocsr() if hasattr(mat, "tocsr") else sp.csr_array(mat)
    val = mat @ x.val
    jac = mat @ x.jac
    hess = []
    for row in range(mat.shape[0]):
        start, end = mat.indptr[row], mat.indptr[row + 1]
        cols = mat.indices[start:end]
        data = mat.data[start:end]
        acc = sp.csr_array((x.num_controls, x.num_controls))
        for coef, col in zip(data, cols):
            acc = acc + coef * x.hess[int(col)]
        hess.append(acc.tocsr())
    return ADI2(val, jac, hess)


def ad2_linsolve(mat, x):
    """Solve ``mat @ y = x`` for an ADI2 right-hand side."""
    if not isinstance(x, ADI2):
        return np.linalg.solve(np.asarray(mat, dtype=float), x)
    mat_csc = mat.tocsc() if hasattr(mat, "tocsc") else sp.csc_array(mat)
    val = sp.linalg.spsolve(mat_csc, x.val)
    inv = sp.csr_array(sp.linalg.spsolve(mat_csc, sp.eye_array(mat_csc.shape[0]).toarray()))
    jac = inv @ x.jac
    hess = []
    for row in range(inv.shape[0]):
        start, end = inv.indptr[row], inv.indptr[row + 1]
        cols = inv.indices[start:end]
        data = inv.data[start:end]
        acc = sp.csr_array((x.num_controls, x.num_controls))
        for coef, col in zip(data, cols):
            acc = acc + coef * x.hess[int(col)]
        hess.append(acc.tocsr())
    return ADI2(val, jac, hess)


def ad2_vertcat(*parts):
    """Stack ADI2/numeric vectors vertically."""
    ads = [p for p in parts if isinstance(p, ADI2)]
    if not ads:
        return np.concatenate([
            np.atleast_1d(np.asarray(p, dtype=float)).ravel()
            for p in parts
        ])
    nctrl = ads[0].num_controls
    full = [_as_adi2_constant(p, nctrl) if not isinstance(p, ADI2) else p
            for p in parts]
    val = np.concatenate([p.val for p in full])
    jac = sp.vstack([p.jac for p in full], format="csr")
    hess = [h for p in full for h in p.hess]
    return ADI2(val, jac, hess)


def ad2_scatter(indices, values, size: int):
    """Scatter vector values into a zero vector of length ``size``."""
    indices = np.asarray(indices, dtype=np.int64).ravel()
    if not isinstance(values, ADI2):
        out = np.zeros(size)
        out[indices] += np.asarray(values, dtype=float).ravel()
        return out
    mat = sp.csr_array(
        (np.ones(indices.size), (indices, np.arange(indices.size))),
        shape=(size, indices.size),
    )
    return ad2_matmul(mat, values)


def ad2_sum(x):
    """Sum to a scalar while preserving exact Hessian."""
    if not isinstance(x, ADI2):
        return np.sum(x)
    ones = sp.csr_array(np.ones((1, len(x))))
    hess = sp.csr_array((x.num_controls, x.num_controls))
    for h in x.hess:
        hess = hess + h
    return ADI2(np.array([x.val.sum()]), ones @ x.jac, [hess.tocsr()])


def ad2_sin(x):
    return _ad2_unary(x, np.sin, np.cos, lambda v: -np.sin(v))


def ad2_cos(x):
    return _ad2_unary(x, np.cos, lambda v: -np.sin(v), lambda v: -np.cos(v))


def ad2_atan2(y, x):
    """Element-wise ``atan2(y, x)`` for ADI2 operands."""
    if not isinstance(y, ADI2) and not isinstance(x, ADI2):
        return np.arctan2(y, x)
    if not isinstance(y, ADI2):
        y = _as_adi2_constant(y, x.num_controls)
    if not isinstance(x, ADI2):
        x = _as_adi2_constant(x, y.num_controls)
    y, x = ADI2._align(y, x)
    val = np.arctan2(y.val, x.val)
    r2 = x.val * x.val + y.val * y.val
    fx = -y.val / r2
    fy = x.val / r2
    jac = _diag_mul(fy, y.jac) + _diag_mul(fx, x.jac)
    hess = []
    for i in range(len(val)):
        gx = _row_dense(x.jac, i)
        gy = _row_dense(y.jac, i)
        xx = x.val[i]
        yy = y.val[i]
        rr = r2[i]
        rr2 = rr * rr
        fxx = 2.0 * xx * yy / rr2
        fyy = -2.0 * xx * yy / rr2
        fxy = (yy * yy - xx * xx) / rr2
        hess.append(
            fx[i] * x.hess[i] + fy[i] * y.hess[i]
            + sp.csr_array(
                fxx * np.outer(gx, gx)
                + fxy * (np.outer(gx, gy) + np.outer(gy, gx))
                + fyy * np.outer(gy, gy)
            )
        )
    return ADI2(val, jac, hess)


def value_jac_hess(x):
    """Return ``(value, jacobian, hessian)`` for an ADI2 expression."""
    if isinstance(x, ADI2):
        hess = x.hess[0].toarray() if len(x) == 1 else [
            h.toarray() for h in x.hess
        ]
        return x.val.copy(), x.jac.toarray(), hess
    val = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    return val, np.zeros((val.size, 0)), np.zeros((0, 0))


def hessian_ad2(fun: Callable[[ADI2], ADI2], x):
    """Evaluate a scalar function and return exact value, gradient, Hessian."""
    y = fun(init_variables_adi2(x))
    if not isinstance(y, ADI2):
        y = _as_adi2_constant(y, np.asarray(x, dtype=float).size)
    if len(y) != 1:
        raise ValueError("hessian_ad2 expects a scalar-output function")
    return float(y.val[0]), y.jac.toarray().ravel(), y.hess[0].toarray()


def _as_adi2_constant(values, nctrl: int) -> ADI2:
    val = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    return ADI2(val, sp.csr_array((val.size, nctrl)),
                [sp.csr_array((nctrl, nctrl)) for _ in range(val.size)])


def _ad2_unary(x, f0, f1, f2):
    if not isinstance(x, ADI2):
        return f0(x)
    val = f0(x.val)
    first = f1(x.val)
    second = f2(x.val)
    jac = _diag_mul(first, x.jac)
    hess = []
    for i in range(len(x)):
        g = _row_dense(x.jac, i)
        hess.append(first[i] * x.hess[i]
                    + sp.csr_array(second[i] * np.outer(g, g)))
    return ADI2(val, jac, hess)


def _diag_mul(d, mat):
    out = mat.tocsr(copy=True)
    out.data *= np.repeat(np.asarray(d, dtype=float).ravel(),
                          np.diff(out.indptr))
    return out


def _row_dense(mat, row: int):
    return np.asarray(mat[row:row + 1].todense()).ravel()


if __name__ == "__main__":
    def test_exact_hessian_quadratic_product():
        def fun(x):
            return x[0] * x[1] + x[0] ** 3 + 2.0 * x[1] ** 2

        val, grad, hess = hessian_ad2(fun, np.array([2.0, 3.0]))
        assert np.isclose(val, 32.0)
        assert np.allclose(grad, [15.0, 14.0])
        assert np.allclose(hess, [[12.0, 1.0], [1.0, 4.0]])
        print("ok test_exact_hessian_quadratic_product")

    test_exact_hessian_quadratic_product()
    print("\nAll tests passed!")
