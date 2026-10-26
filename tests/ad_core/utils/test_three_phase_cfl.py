"""The three-phase saturation wave speed (MRST `getFractionalFlowMagnitude`).

With `n` phases there are `n-1` independent saturations, so fractional flow is
a map from `R^(n-1)` to itself and its derivative is an `(n-1)x(n-1)` matrix
per cell. The wave speed is the **largest absolute eigenvalue** of that matrix.

That is the whole content, and it is not a two-phase formula applied three
times. The characteristic speeds of a coupled system are its eigenvalues; a
single row of the Jacobian describes no wave. Taking a row or diagonal maximum
instead under-estimates the fastest wave wherever the phases are strongly
coupled - which is where the front is - and an under-estimated CFL is a
timestep chosen too long.

Before this, `estimate_flow_cfl` returned **zeros** for any model with more
than two phases. That is worse than an approximation: the saturation
controller was silently neutral on exactly the models whose fronts move
fastest, and nothing said so.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.utils import cfl as C
from pymrst.modules.ad_core.utils.cfl import (
    fractional_flow_derivative, fractional_flow_magnitude,
)
from pymrst.utils.adi import init_variables_adi


class _Fluid2:
    mu = [1e-3, 5e-3]
    kr_w = staticmethod(lambda s: s**2)
    kr_o = staticmethod(lambda s: s**2)


class _Fluid3:
    mu = [1e-3, 5e-3, 2e-5]
    kr_w = staticmethod(lambda s: s**2)
    kr_o = staticmethod(lambda s: s**2)
    kr_g = staticmethod(lambda s: s**2)


def _model(fluid, phases):
    return SimpleNamespace(fluid=fluid, raw_fluid=fluid, n_phases=len(phases),
                           get_phase_names=lambda: list(phases))


# -- the two-phase reduction -------------------------------------------------

def test_two_phases_reduce_to_the_buckley_leverett_slope():
    """The 1x1 case must be exactly `|df_w/ds_w|`, or this is a second
    implementation rather than a generalisation."""
    model = _model(_Fluid2(), ["W", "O"])
    sw = np.array([0.2, 0.5, 0.8])

    got = fractional_flow_magnitude(model, [sw, 1.0 - sw])

    h = 1e-7
    def fw(s):
        return (s**2/1e-3)/((s**2/1e-3) + ((1 - s)**2/5e-3))
    expected = np.abs((fw(sw + h) - fw(sw - h))/(2*h))
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_the_two_phase_entry_point_is_unchanged():
    """`fractional_flow_derivative` reaches the fluid's curves directly, so it
    works on models with no property graph. It must keep doing so."""
    model = _model(_Fluid2(), ["W", "O"])
    sw = np.array([0.3, 0.6])
    np.testing.assert_allclose(
        fractional_flow_derivative(model, sw),
        fractional_flow_magnitude(model, [sw, 1.0 - sw]), rtol=1e-10)


# -- three phases ------------------------------------------------------------

def _three_phase_case():
    model = _model(_Fluid3(), ["W", "O", "G"])
    sw = np.array([0.3, 0.2, 0.1])
    so = np.array([0.5, 0.5, 0.45])
    return model, [sw, so, 1.0 - sw - so]


def test_the_three_phase_speed_is_finite_and_positive():
    model, sat = _three_phase_case()
    got = fractional_flow_magnitude(model, sat)
    assert got.shape == (3,)
    assert np.all(np.isfinite(got)) and np.all(got > 0)


def test_the_speed_is_an_eigenvalue_not_a_diagonal_entry():
    """The assertion the whole port rests on.

    If the eigenvalue step were dropped and the largest diagonal entry used
    instead, this would still return finite positive numbers - just smaller
    ones. They must differ, or the test cannot tell the two apart.
    """
    model, sat = _three_phase_case()
    got = fractional_flow_magnitude(model, sat)

    jacobian = _jacobian(model, sat)
    diagonal_max = np.max(np.abs(np.diagonal(jacobian, axis1=1, axis2=2)),
                          axis=1)
    assert np.any(np.abs(got - diagonal_max) > 1e-9), (
        "eigenvalue and diagonal maximum agree; this case cannot detect the "
        "difference between them")


def test_the_speed_respects_the_gershgorin_bound():
    """Every eigenvalue lies within the largest absolute row sum. A cheap,
    independent check that what came back is an eigenvalue of *this* matrix."""
    model, sat = _three_phase_case()
    got = fractional_flow_magnitude(model, sat)
    bound = np.max(np.abs(_jacobian(model, sat)).sum(axis=2), axis=1)
    assert np.all(got <= bound + 1e-12)


def test_the_speed_matches_a_direct_eigenvalue_computation():
    model, sat = _three_phase_case()
    expected = np.max(np.abs(np.linalg.eigvals(_jacobian(model, sat))), axis=1)
    np.testing.assert_allclose(fractional_flow_magnitude(model, sat), expected,
                               rtol=1e-12)


def test_the_jacobian_entries_match_finite_differences():
    """Pins the seeding: the last saturation is `1 - sum(others)`, not an
    independent variable. Seeding it too would give a singular system for a
    state the model cannot occupy."""
    model, sat = _three_phase_case()
    jacobian = _jacobian(model, sat)

    h = 1e-7
    for j in range(2):
        up = [s.copy() for s in sat[:2]]
        down = [s.copy() for s in sat[:2]]
        up[j] = up[j] + h
        down[j] = down[j] - h
        f_up = _fractions(model, up)
        f_down = _fractions(model, down)
        for i in range(2):
            np.testing.assert_allclose(
                jacobian[:, i, j], (f_up[i] - f_down[i])/(2*h),
                rtol=1e-4, atol=1e-8)


def test_a_single_phase_has_unit_speed():
    model = _model(_Fluid2(), ["W"])
    np.testing.assert_allclose(
        fractional_flow_magnitude(model, [np.ones(4)]), 1.0)


# -- the relative-permeability rule -----------------------------------------

def test_a_three_phase_fluid_with_two_phase_oil_curves_needs_the_rule():
    """The three-phase oil curve is interpolated from `kr_ow` and `kr_og` and
    has no form of its own. A single `kr_o` would be a smooth, plausible wave
    speed computed from a relative permeability the model never uses."""
    fluid = _Fluid3()
    fluid.kr_ow = staticmethod(lambda s: s**2)
    model = _model(fluid, ["W", "O", "G"])
    sw = np.array([0.3])
    with pytest.raises(NotImplementedError, match="rel_perm_wog"):
        fractional_flow_magnitude(model, [sw, np.array([0.5]),
                                          np.array([0.2])])


def test_the_models_interpolation_rule_is_used_when_present():
    """With a rule attached the oil curve must come from it, not from
    `kr_o` - the two give different numbers, which is the point."""
    model, sat = _three_phase_case()
    seen = {}

    def rule(sw, so, sg, fluid):
        seen["called"] = True
        return [sw*0 + 0.4, so*0 + 0.3, sg*0 + 0.2]

    model.rel_perm_wog = rule
    got = fractional_flow_magnitude(model, sat)
    assert seen.get("called"), "the interpolation rule was bypassed"
    # Constant relperms make every fraction constant, so the derivative is 0.
    np.testing.assert_allclose(got, 0.0, atol=1e-12)


# -- helpers -----------------------------------------------------------------

def _jacobian(model, sat):
    """`d(f_i)/d(s_j)` per cell, built the same way the port does."""
    seeded = list(init_variables_adi(*[s.copy() for s in sat[:-1]]))
    last = 1.0
    for s in seeded:
        last = last - s
    seeded.append(last)

    mobility = C._mobilities(model, seeded)
    total = mobility[0]
    for m in mobility[1:]:
        total = total + m

    n = len(sat) - 1
    out = np.zeros((np.size(sat[0]), n, n))
    for i in range(n):
        frac = mobility[i]/total
        for j in range(n):
            out[:, i, j] = frac.jac[j].diagonal()
    return out


def _fractions(model, independent):
    """`f_i` at plain (non-AD) saturations."""
    sat = list(independent)
    last = 1.0
    for s in sat:
        last = last - s
    sat.append(last)
    mobility = C._mobilities(model, sat)
    total = sum(mobility[1:], mobility[0])
    return [mobility[i]/total for i in range(len(independent))]
