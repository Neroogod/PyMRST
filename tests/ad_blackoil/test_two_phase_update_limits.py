"""`TwoPhaseOilWaterModelAD`'s update limits are MRST's rule (Phase 3).

This model sits outside the AD-OO hierarchy - it has no base class - so its
inline update limiting looked like a local invention, one of the ad-hoc
duplicates the migration was meant to remove.

It is not. `PhysicalModel.limit_update_relative` scales a row by
`min(cap/|dv|, 1)`; for a single-column update that is exactly
`sign(dv)*min(|dv|, cap)`, which is a clip. The two agree to within an ulp, and
the clip is the *more* accurate form because it returns `cap` exactly rather
than `dv*(cap/|dv|)`.

So the code stays and the equivalence is pinned here instead. That is the
useful outcome: what looked like divergence was agreement, and nobody could
tell without checking.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.ad_core.models.physical_model import PhysicalModel


def _clip_relative(dv, value, rel):
    """The model's inline form."""
    cap = rel*np.abs(value)
    return np.clip(dv, -cap, cap)


def test_the_inline_relative_clip_is_mrsts_scaling_rule():
    rng = np.random.default_rng(0)
    dv = rng.standard_normal(2000)*1e6
    value = np.abs(rng.standard_normal(2000))*2e7 + 1e6

    clipped = _clip_relative(dv, value, 0.2)
    scaled, _ = PhysicalModel.limit_update_relative(dv[:, None], value[:, None],
                                                     0.2)
    np.testing.assert_allclose(clipped, scaled.ravel(), rtol=1e-14, atol=0)


def test_the_inline_absolute_clip_is_mrsts_scaling_rule():
    rng = np.random.default_rng(1)
    dv = rng.standard_normal(2000)
    scaled, _ = PhysicalModel.limit_update_absolute(dv[:, None], 0.2)
    np.testing.assert_allclose(np.clip(dv, -0.2, 0.2), scaled.ravel(),
                               rtol=1e-14, atol=0)


def test_they_are_equal_but_not_bit_identical():
    """The reason the code was left alone.

    Swapping the clip for the shared helper would move the last bit of every
    limited update - enough to fail a bit-identical characterization gate, for
    no mathematical gain. Asserting the difference is *nonzero* keeps that
    reasoning honest: if the two ever become bit-identical, this note is stale
    and the code should simply use the helper.
    """
    rng = np.random.default_rng(0)
    dv = rng.standard_normal(2000)*1e6
    value = np.abs(rng.standard_normal(2000))*2e7 + 1e6

    clipped = _clip_relative(dv, value, 0.2)
    scaled, _ = PhysicalModel.limit_update_relative(dv[:, None], value[:, None],
                                                     0.2)
    assert not np.array_equal(clipped, scaled.ravel())
    difference = np.abs(clipped - scaled.ravel())
    moved = difference > 0
    assert moved.any(), "some entries must actually be limited"
    assert np.all(difference[moved]/np.abs(clipped[moved]) < 1e-14), (
        "the two forms must agree to rounding, not merely approximately")


def test_the_clip_is_the_more_accurate_form():
    """`clip` returns the cap exactly; scaling divides then multiplies."""
    dv = np.array([1.0e9])
    value = np.array([1.0])
    cap = 0.2*np.abs(value)

    clipped = np.clip(dv, -cap, cap)
    scaled, _ = PhysicalModel.limit_update_relative(dv[:, None], value[:, None],
                                                     0.2)
    assert clipped[0] == cap[0], "clip lands on the cap exactly"
    assert abs(clipped[0] - cap[0]) <= abs(scaled.ravel()[0] - cap[0])


def test_a_saturation_update_stays_within_the_declared_step():
    """The property the limit exists for, independent of which form computes it."""
    rng = np.random.default_rng(3)
    ds = rng.standard_normal(500)*5.0
    limited, _ = PhysicalModel.limit_update_absolute(ds[:, None], 0.2)
    assert np.abs(limited).max() <= 0.2 + 1e-15
    np.testing.assert_allclose(np.clip(ds, -0.2, 0.2), limited.ravel(),
                               rtol=1e-14, atol=0)
