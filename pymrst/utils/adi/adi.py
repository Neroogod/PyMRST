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
