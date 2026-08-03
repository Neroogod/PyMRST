"""Saturation endpoints inferred from relative permeability.

MRST ``getMinMaxPhaseSaturationsFromRelPerm``. On F0.4's risk list with no
focused test. These endpoints set the range equilibrium initialization is
allowed to place saturations in, so an endpoint off by a sample is an
initialization that starts slightly outside the mobile range and a first
timestep that has to correct for it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.initialization.get_min_max_phase_saturations_from_rel_perm import (
    get_min_max_phase_saturations_from_rel_perm,
)


def _corey(residual, opposite_residual, exponent=2.0):
    """A curve that is exactly zero below `residual` and flat above the plateau."""
    span = 1.0 - residual - opposite_residual

    def kr(saturation):
        scaled = (np.asarray(saturation, dtype=float) - residual)/span
        return np.clip(scaled, 0.0, 1.0)**exponent

    return kr


def _model(**fluid):
    phases = {"water": "krW" in fluid, "oil": "krO" in fluid or "krOW" in fluid,
              "gas": "krG" in fluid}
    return SimpleNamespace(fluid=SimpleNamespace(**fluid), **phases)


def test_endpoints_recover_the_residual_and_the_plateau():
    """The two numbers the sampling is for.

    `s_min` is the largest saturation still giving exactly zero mobility and
    `s_max` the smallest one at the plateau. Both are read off a sampled
    curve, so both are exact only to the sample spacing.
    """
    model = _model(krW=_corey(0.2, 0.1), krO=_corey(0.15, 0.25))
    s_min, s_max = get_min_max_phase_saturations_from_rel_perm(model, tol=1e-4)
    np.testing.assert_allclose(s_min, [0.2, 0.15], atol=1e-4)
    np.testing.assert_allclose(s_max, [0.9, 0.75], atol=2e-4)


def test_the_endpoints_bracket_the_mobile_range_in_the_right_order():
    model = _model(krW=_corey(0.2, 0.1), krO=_corey(0.15, 0.25))
    s_min, s_max = get_min_max_phase_saturations_from_rel_perm(model, tol=1e-4)
    assert np.all(s_min < s_max)
    assert np.all(s_min >= 0.0) and np.all(s_max <= 1.0)


def test_a_finer_tolerance_gives_a_closer_endpoint():
    """`tol` is the sample spacing, so the error must shrink with it.

    An implementation that ignored `tol` - or sampled a fixed grid - would
    return the same answer at both resolutions.
    """
    model = _model(krW=_corey(0.2, 0.1))
    coarse, _ = get_min_max_phase_saturations_from_rel_perm(model, tol=1e-1)
    fine, _ = get_min_max_phase_saturations_from_rel_perm(model, tol=1e-4)
    assert abs(fine[0] - 0.2) <= abs(coarse[0] - 0.2)
    np.testing.assert_allclose(fine[0], 0.2, atol=1e-4)


def test_a_mobile_everywhere_curve_has_no_residual():
    """kr > 0 at every sample: there is no zero block to take an endpoint from."""
    model = _model(krW=lambda s: np.full(np.size(s), 0.5))
    s_min, s_max = get_min_max_phase_saturations_from_rel_perm(model, tol=1e-3)
    np.testing.assert_allclose(s_min, [0.0])
    # Constant kr means every sample is the plateau, so the plateau starts at 0.
    np.testing.assert_allclose(s_max, [0.0])


def test_oil_falls_back_to_the_minimum_of_the_two_phase_curves():
    """Three-phase oil has no single `krO`; MRST takes min(krOW, krOG).

    The fallback is only reached when `krO` is absent, and taking either curve
    alone instead of their minimum would report a residual oil saturation
    lower than the model actually enforces.
    """
    model = SimpleNamespace(
        water=True, oil=True, gas=True,
        fluid=SimpleNamespace(
            krW=_corey(0.2, 0.1),
            krOW=_corey(0.1, 0.2),      # zero below 0.10
            krOG=_corey(0.3, 0.2),      # zero below 0.30 - the binding one
            krG=_corey(0.05, 0.1),
        ),
    )
    s_min, _ = get_min_max_phase_saturations_from_rel_perm(model, tol=1e-4)
    np.testing.assert_allclose(s_min[1], 0.3, atol=1e-4)


def test_a_non_positive_or_oversized_tolerance_is_refused():
    model = _model(krW=_corey(0.2, 0.1))
    for bad in (0.0, -1e-3, 1.5):
        with pytest.raises(ValueError, match="tol"):
            get_min_max_phase_saturations_from_rel_perm(model, tol=bad)


def test_only_the_active_phases_are_returned():
    two_phase = _model(krW=_corey(0.2, 0.1), krO=_corey(0.15, 0.25))
    s_min, s_max = get_min_max_phase_saturations_from_rel_perm(two_phase, tol=1e-3)
    assert s_min.shape == s_max.shape == (2,)

    three_phase = SimpleNamespace(
        water=True, oil=True, gas=True,
        fluid=SimpleNamespace(krW=_corey(0.2, 0.1), krO=_corey(0.15, 0.25),
                              krG=_corey(0.05, 0.1)),
    )
    s_min, s_max = get_min_max_phase_saturations_from_rel_perm(three_phase, tol=1e-3)
    assert s_min.shape == s_max.shape == (3,)
