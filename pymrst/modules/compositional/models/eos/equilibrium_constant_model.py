"""MRST ``models/eos/EquilibriumConstantModel.m`` compatibility class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from pymrst.modules.compositional.eos import R_GAS, Mixture
from pymrst.modules.compositional.models.eos.equation_of_state_model import (
    EquationOfStateModel,
)


@dataclass
class EquilibriumConstantFlashResult:
    """K-value flash output."""

    V: np.ndarray
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    Z_L: np.ndarray
    Z_V: np.ndarray


class EquilibriumConstantModel(EquationOfStateModel):
    """Equation-of-state model driven by user supplied equilibrium constants.

    MRST's ``EquilibriumConstantModel`` subclasses ``EquationOfStateModel`` but
    replaces cubic fugacity with prescribed K-value functions.  The functions
    are evaluated as ``K_i(P, T, z)`` and interpreted as ``K_i = y_i/x_i``.
    """

    def __init__(
        self,
        G: Any = None,
        fluid: Mixture | None = None,
        k_values: Iterable[Callable[..., Any]] | Callable[..., Any] | None = None,
        **kwargs,
    ):
        if k_values is None:
            k_values = kwargs.pop("equilibrium_constant_functions", None)
        super().__init__(G, fluid, **kwargs)
        if k_values is None:
            raise ValueError("k_values must be supplied")
        self.equilibriumConstantFunctions = self._normalize_k_functions(k_values)

    @staticmethod
    def _normalize_k_functions(k_values):
        if callable(k_values):
            return [k_values]
        functions = list(k_values)
        if not functions or not all(callable(fn) for fn in functions):
            raise TypeError("k_values must be a callable or a non-empty iterable of callables")
        return functions

    @staticmethod
    def _call_k_function(fn, P, T, z, component_index: int | None = None):
        for args in ((P, T, z), (P, T), (P, z), (P,)):
            try:
                out = fn(*args)
                break
            except TypeError:
                continue
        else:
            if component_index is None:
                out = fn(P, T, z)
            else:
                out = fn(P, T, z[:, component_index])
        return np.asarray(out, dtype=float)

    def evaluate_equilibrium_constants(self, P, T, z) -> np.ndarray:
        """Evaluate K-values in MRST component order."""
        return super().evaluateEquilibriumConstants(P, T, z)

    def evaluateEquilibriumConstants(self, P, T, z) -> np.ndarray:
        """MRST camelCase alias for :meth:`evaluate_equilibrium_constants`."""
        return self.evaluate_equilibrium_constants(P, T, z)

    def get_compressibility_and_fugacity(
        self,
        P,
        T,
        x,
        y,
        z=None,
        Z_L=None,
        Z_V=None,
        *args,
        **kwargs,
    ):
        """Return ``Z_L, Z_V`` and optionally K-value fugacity surrogates."""
        del Z_L, Z_V, args, kwargs
        P = np.asarray(P, dtype=float).reshape(-1)
        T = self._broadcast_temperature(P, T)
        if np.any(P <= 0.0):
            raise AssertionError("Pressures must be positive!")

        rho_l = np.asarray(
            self.PropertyModel.computeMolarDensity(self, P, x, np.nan, T, True),
            dtype=float,
        ).reshape(-1)
        rho_v = np.asarray(
            self.PropertyModel.computeMolarDensity(self, P, y, np.nan, T, False),
            dtype=float,
        ).reshape(-1)
        z_l = P / (rho_l * R_GAS * T)
        z_v = P / (rho_v * R_GAS * T)
        if z is None:
            return z_l, z_v

        K = self.evaluate_equilibrium_constants(P, T, z)
        if isinstance(x, (list, tuple)):
            f_l = [np.asarray(xi, dtype=float) * K[:, i] for i, xi in enumerate(x)]
            f_v = [np.asarray(yi, dtype=float) for yi in y]
        else:
            f_l = self._as_2d(x, "x") * K
            f_v = self._as_2d(y, "y")
        return z_l, z_v, f_l, f_v

    def getCompressibilityAndFugacity(
        self,
        P,
        T,
        x,
        y,
        z=None,
        Z_L=None,
        Z_V=None,
        *args,
        **kwargs,
    ):
        """MRST camelCase alias for :meth:`get_compressibility_and_fugacity`."""
        return self.get_compressibility_and_fugacity(
            P, T, x, y, z, Z_L, Z_V, *args, **kwargs
        )

    def flash(self, P, T, z) -> EquilibriumConstantFlashResult:
        """Flash overall composition using the prescribed K-values."""
        P = np.asarray(P, dtype=float).reshape(-1)
        T = self._broadcast_temperature(P, T)
        z = _normalize_rows(self._as_2d(z, "z"))
        K = self.evaluate_equilibrium_constants(P, T, z)
        V = _rachford_rice(z, K)
        x = _normalize_rows(z / (1.0 + V[:, None] * (K - 1.0)))
        y = _normalize_rows(K * x)
        Z_L, Z_V = self.get_compressibility_and_fugacity(P, T, x, y)
        return EquilibriumConstantFlashResult(V=V, x=x, y=y, K=K, Z_L=Z_L, Z_V=Z_V)

    def update_after_convergence(self, state, report=None, *args, **kwargs):
        """MRST-style state update for K-value flash output.

        This intentionally follows MRST's field names: ``components`` for the
        overall composition, ``x``/``y`` for phase compositions and ``Z_L`` /
        ``Z_V`` for compressibility factors.
        """
        del args, kwargs
        pressure = _state_get(state, "pressure")
        temperature = _state_get(state, "T", _state_get(state, "temperature", None))
        if temperature is None:
            temperature = getattr(self.CompositionalMixture, "T", 300.0)
        components = _state_get(state, "components", _state_get(state, "z", None))
        if components is None:
            raise ValueError("state must contain components or z")
        flashed = self.flash(pressure, temperature, components)
        _state_set(state, "x", flashed.x)
        _state_set(state, "y", flashed.y)
        _state_set(state, "Z_L", flashed.Z_L)
        _state_set(state, "Z_V", flashed.Z_V)
        _state_set(state, "L", 1.0 - flashed.V)
        _state_set(state, "V", flashed.V)
        return state, report

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None, *args, **kwargs):
        """MRST camelCase alias for :meth:`update_after_convergence`."""
        del state0, dt, drivingForces
        return self.update_after_convergence(state, *args, **kwargs)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), 1.0e-300)
    return x / x.sum(axis=1, keepdims=True)


def _rachford_rice(z: np.ndarray, K: np.ndarray, tol: float = 1.0e-12, max_it: int = 100):
    out = np.zeros(z.shape[0])
    for c in range(z.shape[0]):
        k = K[c]
        zz = z[c]
        if np.all(k <= 1.0):
            out[c] = 0.0
            continue
        if np.all(k >= 1.0):
            out[c] = 1.0
            continue
        lo, hi = 0.0, 1.0
        flo = np.sum(zz * (k - 1.0) / (1.0 + lo * (k - 1.0)))
        for _ in range(max_it):
            mid = 0.5 * (lo + hi)
            fmid = np.sum(zz * (k - 1.0) / (1.0 + mid * (k - 1.0)))
            if abs(fmid) < tol or hi - lo < tol:
                lo = hi = mid
                break
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo = mid
                flo = fmid
        out[c] = 0.5 * (lo + hi)
    return out


def _state_get(state, name: str, default=None):
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _state_set(state, name: str, value) -> None:
    if isinstance(state, dict):
        state[name] = value
    else:
        setattr(state, name, value)


__all__ = ["EquilibriumConstantModel", "EquilibriumConstantFlashResult"]
