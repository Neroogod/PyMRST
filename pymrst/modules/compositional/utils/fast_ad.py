"""Small dense AD class (MRST ``FastAD.m`` layout)."""

from __future__ import annotations

import numpy as np


def value(v):
    """Return the numeric value of a FastAD object or raw value."""
    return v.val if isinstance(v, FastAD) else v


class FastAD:
    """Very small dense AD object for quick compositional EOS calculations."""

    __array_priority__ = 1000

    def __init__(self, val, jac):
        self.val = np.asarray(val, dtype=float).reshape(-1)
        self.jac = np.asarray(jac, dtype=float)
        if self.jac.ndim == 1:
            self.jac = self.jac.reshape(self.val.size, 1)
        if self.jac.shape[0] != self.val.size:
            raise ValueError(f"jacobian row count {self.jac.shape[0]} must match value size {self.val.size}")

    def copy(self) -> "FastAD":
        return FastAD(self.val.copy(), self.jac.copy())

    def value(self) -> np.ndarray:
        return self.val

    def double(self) -> np.ndarray:
        return self.val

    def __array__(self, dtype=None):
        return np.asarray(self.val, dtype=dtype)

    def _coerce(self, other):
        return other if isinstance(other, FastAD) else None

    def __neg__(self) -> "FastAD":
        return FastAD(-self.val, -self.jac)

    def __add__(self, other):
        if isinstance(other, FastAD):
            return FastAD(self.val + other.val, self.jac + other.jac)
        return FastAD(self.val + other, self.jac)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other if isinstance(other, FastAD) else -np.asarray(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, FastAD):
            val = self.val * other.val
            jac = self.val[:, None] * other.jac + other.val[:, None] * self.jac
            return FastAD(val, jac)
        arr = np.asarray(other, dtype=float)
        val = self.val * arr
        jac = self.jac * np.asarray(val / self.val if False else arr).reshape(-1, 1) if arr.ndim else self.jac * float(arr)
        if arr.ndim:
            jac = self.jac * arr.reshape(-1, 1)
        return FastAD(val, jac)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, FastAD) else np.asarray(other, dtype=float) ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __pow__(self, other):
        if isinstance(other, FastAD):
            val = self.val ** other.val
            jac = val[:, None] * ((other.val / self.val)[:, None] * self.jac + np.log(self.val)[:, None] * other.jac)
            return FastAD(val, jac)
        exponent = float(other)
        val = self.val ** exponent
        jac = (exponent * self.val ** (exponent - 1.0))[:, None] * self.jac
        return FastAD(val, jac)

    def __rpow__(self, other):
        base = float(other)
        val = base ** self.val
        jac = (val * np.log(base))[:, None] * self.jac
        return FastAD(val, jac)

    def __lt__(self, other):
        return self.val < value(other)

    def __le__(self, other):
        return self.val <= value(other)

    def __gt__(self, other):
        return self.val > value(other)

    def __ge__(self, other):
        return self.val >= value(other)

    def sum(self):
        return FastAD(np.array([self.val.sum()]), self.jac.sum(axis=0, keepdims=True))

    def max(self, other=None):
        if other is None:
            idx = int(np.argmax(self.val))
            return FastAD(np.array([self.val[idx]]), self.jac[idx:idx + 1, :])
        if isinstance(other, FastAD):
            choose_self = self.val > other.val
            val = np.where(choose_self, self.val, other.val)
            jac = np.where(choose_self[:, None], self.jac, other.jac)
            return FastAD(val, jac)
        arr = np.asarray(other, dtype=float)
        choose_self = self.val > arr
        val = np.where(choose_self, self.val, arr)
        jac = np.where(choose_self[:, None], self.jac, 0.0)
        return FastAD(val, jac)

    def log(self):
        return FastAD(np.log(self.val), (1.0 / self.val)[:, None] * self.jac)

    def exp(self):
        val = np.exp(self.val)
        return FastAD(val, val[:, None] * self.jac)


def log(u):
    return u.log() if isinstance(u, FastAD) else np.log(u)


def exp(u):
    return u.exp() if isinstance(u, FastAD) else np.exp(u)


def fast_sum(u):
    return u.sum() if isinstance(u, FastAD) else np.sum(u)


def fast_max(u, v=None):
    if isinstance(u, FastAD):
        return u.max(v)
    return np.max(u) if v is None else np.maximum(u, v)


__all__ = ["FastAD", "value", "log", "exp", "fast_sum", "fast_max"]
