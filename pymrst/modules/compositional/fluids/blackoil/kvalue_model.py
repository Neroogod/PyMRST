"""K-value compositional helpers used by SPE1 black-oil conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.fluids.blackoil.convert_rs_rv_to_k_value import (
    RSRVKValueProperties,
)

R_GAS = 8.3144598


def _regular_interp(axis0, axis1, values):
    return RegularGridInterpolator(
        (np.asarray(axis0, dtype=float), np.asarray(axis1, dtype=float)),
        np.asarray(values, dtype=float),
        bounds_error=False,
        fill_value=None,
    )


class BlackOilPropertyModel:
    """Port of MRST ``BlackOilPropertyModel``.

    Two construction modes are supported:
    - ``BlackOilPropertyModel(props)`` for PyMRST's tabulated RSRV/K-value
      conversion path.
    - ``BlackOilPropertyModel(rho, mu, satfn=None)`` matching MRST's function
      backed constructor, where ``rho`` and ``mu`` are two callables/lists for
      liquid and vapour.
    """

    def __init__(self, props_or_rho, mu=None, satfn=None, **kwargs):
        del kwargs
        self.props = None
        self.densityFunctions = None
        self.viscosityFunctions = None
        self.checkStabilityFunction = satfn
        if isinstance(props_or_rho, RSRVKValueProperties):
            self.props = props_or_rho
            self._init_table_interpolants()
        else:
            if mu is None:
                raise TypeError("function-backed BlackOilPropertyModel requires rho and mu")
            self.densityFunctions = list(props_or_rho)
            self.viscosityFunctions = list(mu)
            if len(self.densityFunctions) != 2 or len(self.viscosityFunctions) != 2:
                raise ValueError("rho and mu must each contain liquid and vapour functions")

    def _init_table_interpolants(self):
        pr = self.props.properties
        p_axis = pr.pressure[0, :]
        x_o_axis = pr.x_o[:, 0]
        y_g_axis = pr.y_g[:, 0]
        self._rho_o = _regular_interp(x_o_axis, p_axis, pr.rhoO)
        self._mu_o_inv = _regular_interp(x_o_axis, p_axis, 1.0 / pr.muO)
        self._rho_g = _regular_interp(y_g_axis, p_axis, pr.rhoG)
        self._mu_g_inv = _regular_interp(y_g_axis, p_axis, 1.0 / pr.muG)

    @staticmethod
    def _as_component_columns(comp):
        if isinstance(comp, (list, tuple)):
            return [np.asarray(c, dtype=float).reshape(-1) for c in comp]
        arr = np.asarray(comp, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        return [arr[:, i] for i in range(arr.shape[1])]

    @staticmethod
    def _call_property_function(fn, p, T, comp):
        cols = BlackOilPropertyModel._as_component_columns(comp)
        try:
            return fn(p, T, cols)
        except TypeError:
            try:
                return fn(p, T, np.column_stack(cols))
            except TypeError:
                return fn(p, np.column_stack(cols))

    def compute_density(self, p, comp, is_liquid: bool, T=None):
        p = np.asarray(p, dtype=float).ravel()
        if self.props is None:
            ix = 0 if is_liquid else 1
            return np.asarray(self._call_property_function(self.densityFunctions[ix], p, T, comp), dtype=float).reshape(-1)
        comp = np.asarray(comp, dtype=float)
        axis = comp[:, 0] if is_liquid else comp[:, 1]
        interp = self._rho_o if is_liquid else self._rho_g
        return interp(np.column_stack([axis, p]))

    def compute_molar_density(self, p, comp, is_liquid: bool, T=None):
        return self.compute_density(p, comp, is_liquid, T=T)

    def compute_viscosity(self, p, comp, is_liquid: bool, T=None):
        p = np.asarray(p, dtype=float).ravel()
        if self.props is None:
            ix = 0 if is_liquid else 1
            return np.asarray(self._call_property_function(self.viscosityFunctions[ix], p, T, comp), dtype=float).reshape(-1)
        comp = np.asarray(comp, dtype=float)
        axis = comp[:, 0] if is_liquid else comp[:, 1]
        interp = self._mu_o_inv if is_liquid else self._mu_g_inv
        return 1.0 / interp(np.column_stack([axis, p]))

    # MRST camelCase method aliases. ``eos`` and ``Z`` are accepted for
    # signature compatibility; black-oil functions use pressure, temperature
    # and composition directly.
    def computeDensity(self, eos, p, x, Z, T, isLiquid):
        del eos, Z
        return self.compute_density(p, x, bool(isLiquid), T=T)

    def computeMolarDensity(self, eos, p, x, Z, T, isLiquid):
        del eos, Z
        return self.compute_molar_density(p, x, bool(isLiquid), T=T)

    def computeViscosity(self, eos, p, x, Z, T, isLiquid):
        del eos, Z
        return self.compute_viscosity(p, x, bool(isLiquid), T=T)

    def checkStability(self, p, T, z):
        if self.checkStabilityFunction is None:
            return np.zeros(np.asarray(p, dtype=float).reshape(-1).shape, dtype=bool)
        return np.asarray(self.checkStabilityFunction(p, T, z), dtype=bool)


@dataclass
class EquilibriumConstantModel:
    """Port of MRST ``EquilibriumConstantModel`` for two pseudo components."""

    mixture: Mixture
    props: RSRVKValueProperties
    property_model: BlackOilPropertyModel

    def __post_init__(self):
        kv = self.props.k_values
        p_axis = kv.pressure[0, :]
        z_g_axis = kv.z_g[:, 0]
        z_o_axis = kv.z_o[::-1, 0]
        self._k_g = _regular_interp(z_g_axis, p_axis, kv.K_g)
        self._k_o = _regular_interp(z_o_axis, p_axis, kv.K_o[::-1, :])
        self._free_gas = _regular_interp(z_g_axis, p_axis, kv.freeGas)

    def evaluate_equilibrium_constants(self, p, T, z):
        del T
        p = np.asarray(p, dtype=float).ravel()
        z = np.asarray(z, dtype=float)
        ko = self._k_o(np.column_stack([z[:, 0], p]))
        kg = self._k_g(np.column_stack([z[:, 1], p]))
        return np.column_stack([ko, kg])

    def is_stable(self, p, T, z):
        del T
        p = np.asarray(p, dtype=float).ravel()
        z = np.asarray(z, dtype=float)
        return self._free_gas(np.column_stack([z[:, 1], p])) == 0.0

    def flash(self, p, T, z):
        """Flash using tabulated equilibrium constants.

        This is the K-value analogue of MRST's equilibrium-constant EOS:
        ``K_i = y_i/x_i`` and Rachford-Rice gives the phase split.
        """
        K = self.evaluate_equilibrium_constants(p, T, z)
        V = _rachford_rice_k(np.asarray(z, dtype=float), K)
        x = z / (1.0 + V[:, None] * (K - 1.0))
        y = K * x
        x = _normalize(x)
        y = _normalize(y)
        rho_l = self.property_model.compute_molar_density(p, x, True)
        rho_v = self.property_model.compute_molar_density(p, y, False)
        p = np.asarray(p, dtype=float).ravel()
        T = np.broadcast_to(np.asarray(T, dtype=float).ravel(), p.shape)
        Z_L = p / (rho_l * R_GAS * T)
        Z_V = p / (rho_v * R_GAS * T)
        return KValueFlashResult(V=V, x=x, y=y, K=K, Z_L=Z_L, Z_V=Z_V)


@dataclass
class KValueFlashResult:
    V: np.ndarray
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    Z_L: np.ndarray
    Z_V: np.ndarray


def _normalize(x):
    x = np.maximum(np.asarray(x, dtype=float), 1.0e-300)
    return x / x.sum(axis=1, keepdims=True)


def _rachford_rice_k(z, K, tol: float = 1.0e-12, max_it: int = 100):
    """Vapour fraction `V` from `z` and `K`, by bisection on Rachford-Rice.

        f(V) = sum_i z_i (K_i - 1) / (1 + V (K_i - 1))

    `f` is strictly decreasing in `V`, so the mixture is two-phase exactly when
    `f(0) > 0 > f(1)`; those two endpoint signs are what decide the phase state.
    Checking instead whether every `K_i` sits on one side of 1 is necessary but
    not sufficient: a mixture with `K_oil < 1 < K_gas` can still be single
    phase, which is the ordinary situation for undersaturated oil carrying a
    little dissolved gas. Bisecting `[0, 1]` in that case searches a bracket
    holding no root - `f` keeps one sign throughout, every step moves the same
    end, the interval collapses on `V = 1`, and an all-liquid cell is reported
    as all vapour. On a sweep of the physical `K_oil < 1 < K_gas` regime that
    mistake hits about 4% of states, each with the maximum possible error.
    """
    z = np.asarray(z, dtype=float)
    K = np.asarray(K, dtype=float)
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
        if np.sum(zz * (k - 1.0)) <= 0.0:
            out[c] = 0.0          # f(0) <= 0: bubble point not reached
            continue
        # f(1) = sum z_i (K_i - 1)/K_i. A component with K_i = 0 cannot enter
        # the vapour at all, which drives f(1) to -inf - two-phase, and worth
        # spelling out rather than dividing by zero for the same answer.
        absent = k <= 0.0
        if np.any(absent & (zz > 0.0)):
            f1 = -np.inf
        else:
            safe = np.where(absent, 1.0, k)
            f1 = np.sum(np.where(absent, 0.0, zz * (k - 1.0) / safe))
        if f1 >= 0.0:
            out[c] = 1.0          # f(1) >= 0: past the dew point
            continue
        lo, hi = 0.0, 1.0
        for _ in range(max_it):
            mid = 0.5 * (lo + hi)
            fm = np.sum(zz * (k - 1.0) / (1.0 + mid * (k - 1.0)))
            if abs(fm) < tol or hi - lo < tol:
                lo = hi = mid
                break
            # f decreases in V, so a positive residual puts the root above mid.
            if fm > 0.0:
                lo = mid
            else:
                hi = mid
        out[c] = 0.5 * (lo + hi)
    return out


if __name__ == "__main__":
    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import setup_spe1_tutorial
    from pymrst.modules.compositional.fluids.blackoil.convert_rs_rv_to_k_value import convert_rs_rv_to_k_value
    from pymrst.modules.compositional.examples.run_compositional_spe1 import pseudo_blackoil_mixture

    def test_kvalue_model_evaluates_spe1_tables():
        _, model, state = setup_spe1_tutorial()
        props = convert_rs_rv_to_k_value(model, np_=12, nz=16)
        prop_model = BlackOilPropertyModel(props)
        eos = EquilibriumConstantModel(pseudo_blackoil_mixture(), props, prop_model)
        p = np.asarray(state["pressure"], dtype=float)[:3]
        z = np.tile(np.array([0.8, 0.2]), (3, 1))
        T = np.full(3, 273.15)
        K = eos.evaluate_equilibrium_constants(p, T, z)
        fr = eos.flash(p, T, z)
        assert K.shape == (3, 2)
        assert fr.x.shape == (3, 2)
        assert np.allclose(fr.x.sum(axis=1), 1.0)
        assert np.allclose(fr.y.sum(axis=1), 1.0)
        print("ok test_kvalue_model_evaluates_spe1_tables")

    def test_single_phase_liquid_with_mixed_k_values():
        """Undersaturated oil must flash to V = 0, not V = 1.

        `K_oil < 1 < K_gas` puts the mixture past neither the `all K <= 1` nor
        the `all K >= 1` shortcut, yet with little enough gas it is still single
        phase liquid - `f(0) <= 0`. Bisecting `[0, 1]` there searches a bracket
        with no root and lands on V = 1, turning an all-liquid cell into an
        all-vapour one.
        """
        z = np.array([[0.9957, 0.0043], [0.9101, 0.0899], [0.9976, 0.0024]])
        K = np.array([[0.0448, 2.141], [0.0652, 7.401], [0.7590, 15.885]])
        assert np.all(np.sum(z * (K - 1.0), axis=1) <= 0.0), "these must be liquid"
        V = _rachford_rice_k(z, K)
        assert np.allclose(V, 0.0), f"expected all-liquid, got V = {V}"
        print("ok test_single_phase_liquid_with_mixed_k_values")

    def test_rachford_rice_endpoints_and_root():
        # past the dew point: f(1) >= 0 => all vapour
        z = np.array([[0.02, 0.98]])
        K = np.array([[0.9, 40.0]])
        assert np.sum(z[0] * (K[0] - 1.0) / K[0]) >= 0.0
        assert np.allclose(_rachford_rice_k(z, K), 1.0)
        # a genuine two-phase state: the returned V must zero the residual
        z2 = np.array([[0.6, 0.4]])
        K2 = np.array([[0.2, 5.0]])
        V = _rachford_rice_k(z2, K2)
        assert 0.0 < V[0] < 1.0, V
        f = np.sum(z2[0] * (K2[0] - 1.0) / (1.0 + V[0] * (K2[0] - 1.0)))
        assert abs(f) < 1e-9, f
        # the one-sided shortcuts still hold
        assert np.allclose(_rachford_rice_k(np.array([[0.5, 0.5]]),
                                            np.array([[0.3, 0.8]])), 0.0)
        assert np.allclose(_rachford_rice_k(np.array([[0.5, 0.5]]),
                                            np.array([[1.2, 3.0]])), 1.0)
        print("ok test_rachford_rice_endpoints_and_root")

    def test_zero_k_component_is_two_phase_not_a_divide_by_zero():
        """K_i = 0 means component i cannot enter the vapour: f(1) = -inf."""
        z = np.array([[0.5, 0.5]])
        K = np.array([[0.0, 4.0]])
        with np.errstate(divide="raise", invalid="raise"):
            V = _rachford_rice_k(z, K)
        assert 0.0 < V[0] < 1.0, V
        print("ok test_zero_k_component_is_two_phase_not_a_divide_by_zero")

    test_kvalue_model_evaluates_spe1_tables()
    test_single_phase_liquid_with_mixed_k_values()
    test_rachford_rice_endpoints_and_root()
    test_zero_k_component_is_two_phase_not_a_divide_by_zero()
    print("\nAll tests passed!")
