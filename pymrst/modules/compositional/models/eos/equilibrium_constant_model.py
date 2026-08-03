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
        # Bracket check. Mixed K-values do not guarantee a root in [0, 1]: a
        # feed below its bubble point has f(0) <= 0 and f(1) <= 0, and the
        # bisection below then walks `lo` to 1.0 and reports an all-vapour
        # mixture that is entirely liquid. The same defect was fixed in
        # `fluids/blackoil/kvalue_model.py`.
        f0 = np.sum(zz * (k - 1.0))
        if f0 <= 0.0:
            out[c] = 0.0          # bubble point not reached
            continue
        absent = k <= 0.0
        if np.any(absent & (zz > 0.0)):
            f1 = -np.inf
        else:
            safe = np.where(absent, 1.0, k)
            f1 = np.sum(np.where(absent, 0.0, zz * (k - 1.0) / safe))
        if f1 >= 0.0:
            out[c] = 1.0          # past the dew point
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


if __name__ == "__main__":

    def _residual(z, K, V):
        """The Rachford-Rice function itself, which must vanish at the root."""
        return float(np.sum(z * (K - 1.0) / (1.0 + V * (K - 1.0))))

    def test_a_binary_split_matches_the_closed_form():
        """Two components have an analytic solution, so this is exact.

        For a binary, `z1 a/(1+Va) + z2 b/(1+Vb) = 0` with `a = K1-1`,
        `b = K2-1` rearranges to `V = -(z1 a + z2 b)/(a b)`. Bisection has no
        business disagreeing with that, and comparing against it catches a
        bracket that converges to the wrong root.
        """
        z = np.array([[0.6, 0.4]])
        K = np.array([[3.0, 0.2]])
        V = _rachford_rice(z, K)[0]
        a, b = K[0, 0] - 1.0, K[0, 1] - 1.0
        analytic = -(z[0, 0] * a + z[0, 1] * b) / (a * b)
        assert abs(V - analytic) < 1e-9, (V, analytic)
        print(f"ok test_a_binary_split_matches_the_closed_form "
              f"(V = {V:.10f} against {analytic:.10f})")

    def test_the_rachford_rice_function_vanishes_at_the_root():
        """The defining property, checked on several multicomponent mixtures.

        A solver that returns a plausible vapour fraction without satisfying
        the material balance gives compositions that do not sum to the feed,
        and nothing downstream raises.
        """
        cases = [
            (np.array([0.5, 0.3, 0.2]), np.array([4.0, 0.9, 0.05])),
            (np.array([0.1, 0.2, 0.3, 0.4]), np.array([8.0, 2.0, 0.4, 0.02])),
            (np.array([0.25, 0.25, 0.25, 0.25]), np.array([1.8, 1.2, 0.7, 0.3])),
        ]
        worst = 0.0
        for z, K in cases:
            V = _rachford_rice(z[None, :], K[None, :])[0]
            assert 0.0 <= V <= 1.0, V
            worst = max(worst, abs(_residual(z, K, V)))
        assert worst < 1e-9, worst
        print(f"ok test_the_rachford_rice_function_vanishes_at_the_root "
              f"({len(cases)} mixtures, worst residual {worst:.2e})")

    def test_an_all_liquid_mixture_returns_no_vapour():
        """Every K below one means nothing boils: V must be exactly 0.

        Running the bisection here would converge to a bracket endpoint and
        return a small positive fraction of a phase that does not exist.
        """
        z = np.array([[0.4, 0.6]])
        K = np.array([[0.4, 0.05]])
        assert _rachford_rice(z, K)[0] == 0.0
        print("ok test_an_all_liquid_mixture_returns_no_vapour")

    def test_an_all_vapour_mixture_returns_no_liquid():
        z = np.array([[0.4, 0.6]])
        K = np.array([[2.5, 8.0]])
        assert _rachford_rice(z, K)[0] == 1.0
        print("ok test_an_all_vapour_mixture_returns_no_liquid")

    def test_more_volatile_feed_gives_more_vapour():
        """V rises with the light component's K, which a sign error reverses.

        The first version of this test compared K = [6.0, 0.2] against
        K = [1.6, 0.2]; the latter is below its bubble point, not two-phase,
        which is how the bracket bug below was found.
        """
        z = np.array([[0.5, 0.5]])
        # Only K1 varies. Moving both at once is ill-posed: raising the light
        # K and lowering the heavy one push V in opposite directions, and the
        # heavy component wins - K = [3.0, 0.4] gives more vapour than
        # K = [6.0, 0.2], which looks like a violation and is not.
        fractions = [_rachford_rice(z, np.array([[k1, 0.2]]))[0]
                     for k1 in (2.0, 3.0, 4.0, 6.0, 8.0)]
        assert all(0.0 < v < 1.0 for v in fractions), fractions
        assert all(a < b for a, b in zip(fractions, fractions[1:])), fractions
        print(f"ok test_more_volatile_feed_gives_more_vapour "
              f"({fractions[0]:.4f} to {fractions[-1]:.4f} as K1 goes 2 to 8)")

    def test_a_feed_below_its_bubble_point_is_all_liquid():
        """Mixed K-values do not guarantee a two-phase split.

        With z = [0.5, 0.5] and K = [1.6, 0.2] the Rachford-Rice function is
        negative at both ends - f(0) = -0.1, f(1) = -1.81 - so no root exists
        and the feed is entirely liquid. Without a bracket check the bisection
        walks its lower bound to 1.0 and reports an all-vapour mixture. The
        identical defect was fixed in `fluids/blackoil/kvalue_model.py`; this
        test is what found the second copy.
        """
        z = np.array([[0.5, 0.5]])
        K = np.array([[1.6, 0.2]])
        assert _residual(z[0], K[0], 0.0) < 0.0, "the case is not single-phase"
        assert _rachford_rice(z, K)[0] == 0.0
        print("ok test_a_feed_below_its_bubble_point_is_all_liquid")

    def test_each_cell_is_solved_independently():
        """Rows must not bleed into one another."""
        z = np.array([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        K = np.array([[3.0, 0.2], [0.4, 0.05], [2.5, 8.0]])
        V = _rachford_rice(z, K)
        assert V.shape == (3,)
        assert abs(V[0] - _rachford_rice(z[:1], K[:1])[0]) < 1e-12
        assert V[1] == 0.0 and V[2] == 1.0, V
        print(f"ok test_each_cell_is_solved_independently ({np.round(V, 6).tolist()})")

    def test_k_values_are_required():
        """A K-value model without K-values would flash everything to liquid."""
        try:
            EquilibriumConstantModel(None, None, None)
        except ValueError as error:
            assert "k_values" in str(error), error
        else:
            raise AssertionError("expected a missing k_values to be rejected")
        print("ok test_k_values_are_required")

    test_a_binary_split_matches_the_closed_form()
    test_the_rachford_rice_function_vanishes_at_the_root()
    test_an_all_liquid_mixture_returns_no_vapour()
    test_an_all_vapour_mixture_returns_no_liquid()
    test_more_volatile_feed_gives_more_vapour()
    test_a_feed_below_its_bubble_point_is_all_liquid()
    test_each_cell_is_solved_independently()
    test_k_values_are_required()
    print("\nAll tests passed!")
