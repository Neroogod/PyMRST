"""Relative-permeability endpoint scaling (ad_core port).

A relperm table is measured on one rock. Endpoint scaling stretches it onto
each cell's own saturation limits, so a cell with higher connate water does not
start mobilising water at the table's connate value.

Two schemes, and the difference is what the middle of the curve does:

* **two-point** fixes one linear map from the critical saturation to the upper
  point. The ends land correctly and nothing between them is controlled;
* **three-point** pins the *displacing* phase's residual as well, giving two
  segments. That middle point is exactly what two-point cannot represent, and
  it is the reason three-point exists.

The load-bearing detail is quieter than either: a deck supplies endpoints for
some cells and not others, and `get_pair` falls back to the table wherever the
scaled value is NaN. Without that, an unspecified cell gets a NaN that
propagates into every permeability it touches.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.statefunctions.flowprops.virtual.saturation_property import (
    SaturationProperty,
)

N = 3


def _fluid(**points):
    """`krPts[phase] = [connate, critical, upper, max_kr]` per region."""
    default = {"w": [[0.1, 0.2, 0.8, 1.0]],
               "ow": [[0.0, 0.15, 0.9, 1.0]],
               "g": [[0.0, 0.05, 0.85, 1.0]],
               "og": [[0.0, 0.1, 0.9, 1.0]]}
    default.update(points)
    return SimpleNamespace(krPts=SimpleNamespace(**{
        k: np.array(v, dtype=float) for k, v in default.items()}))


def _scaled(**points):
    """Per-cell endpoint overrides; NaN means "use the table"."""
    return SimpleNamespace(**{k: np.array(v, dtype=float)
                              for k, v in points.items()})


REG = np.zeros(N, dtype=int)
CELLS = np.arange(N)


# -- the fallback ----------------------------------------------------------

def test_a_cell_without_an_override_inherits_the_table():
    """The quiet one. A NaN here propagates into every permeability the cell
    touches, and nothing upstream would say why."""
    table, scaled = SaturationProperty.get_pair(
        "w", SaturationProperty.CR,
        lambda p, i: np.array([0.2]),
        lambda p, i: np.array([np.nan, 0.35, np.nan]))

    np.testing.assert_allclose(scaled, [0.2, 0.35, 0.2])
    assert not np.any(np.isnan(scaled))


def test_an_override_wins_where_it_is_given():
    _, scaled = SaturationProperty.get_pair(
        "w", SaturationProperty.CR,
        lambda p, i: np.array([0.2]),
        lambda p, i: np.array([0.4, 0.4, 0.4]))
    np.testing.assert_allclose(scaled, 0.4)


def test_the_point_indices_are_mrsts_one_based_ones():
    """Written as named constants because an off-by-one here silently reads
    the critical saturation where the connate was meant."""
    assert (SaturationProperty.L, SaturationProperty.CR,
            SaturationProperty.U, SaturationProperty.KM) == (1, 2, 3, 4)


# -- two-point scaling -----------------------------------------------------

def test_two_point_scaling_maps_both_endpoints_exactly():
    """`s = m*S + c` must send the cell's critical point to the table's, and
    the cell's upper point to the table's. Those two conditions define it."""
    fluid = _fluid()
    scaled = _scaled(w=np.tile([[0.15, 0.3, 0.75, 1.0]], (N, 1)))

    m, c, points, _ = SaturationProperty.get_two_point_scalers(
        scaled, "w", REG, fluid, CELLS)

    SCR, SU = points
    np.testing.assert_allclose(m*np.asarray(SCR) + c, 0.2, rtol=1e-12)
    np.testing.assert_allclose(m*np.asarray(SU) + c, 0.8, rtol=1e-12)


def test_an_unscaled_cell_gets_the_identity_map():
    """If the overrides equal the table, scaling must do nothing - otherwise
    turning scaling on would change a model that has no endpoint data."""
    fluid = _fluid()
    scaled = _scaled(w=np.tile([[0.1, 0.2, 0.8, 1.0]], (N, 1)))

    m, c, _, _ = SaturationProperty.get_two_point_scalers(
        scaled, "w", REG, fluid, CELLS)
    np.testing.assert_allclose(m, 1.0)
    np.testing.assert_allclose(c, 0.0, atol=1e-15)


def test_the_oil_upper_point_is_one_minus_the_other_connates():
    """Oil has no upper endpoint of its own - it occupies whatever the other
    phases leave."""
    fluid = _fluid()
    scaled = _scaled(w=np.tile([[0.15, 0.3, 0.75, 1.0]], (N, 1)),
                     g=np.tile([[0.05, 0.1, 0.8, 1.0]], (N, 1)),
                     ow=np.tile([[0.0, 0.2, 0.9, 1.0]], (N, 1)))

    _, _, points, _ = SaturationProperty.get_two_point_scalers(
        scaled, "ow", REG, fluid, CELLS)
    np.testing.assert_allclose(points[1], 1.0 - 0.15 - 0.05)


def test_an_unknown_phase_is_refused():
    with pytest.raises(ValueError, match="two-point scalers"):
        SaturationProperty.get_two_point_scalers(
            _scaled(), "xx", REG, _fluid(), CELLS)


# -- three-point scaling ---------------------------------------------------

def test_three_point_scaling_pins_the_residual_as_well_as_the_ends():
    """The whole reason it exists: two-point controls the ends only, so the
    displacing phase's residual - where the front actually sits - is free."""
    fluid = _fluid()
    # The oil-water critical saturation has to be high enough that the water
    # residual `SR = 1 - SOWCR - SGL` falls *below* the water upper point.
    # Chosen deliberately: an earlier version put SU under SR, which triggers
    # the degenerate guard and zeroes the second segment, so the segments
    # could not meet and the test failed against correct code.
    scaled = _scaled(w=np.tile([[0.15, 0.3, 0.75, 1.0]], (N, 1)),
                     ow=np.tile([[0.0, 0.35, 0.9, 1.0]], (N, 1)),
                     g=np.tile([[0.0, 0.05, 0.85, 1.0]], (N, 1)))

    m, c, points, _ = SaturationProperty.get_three_point_scalers(
        scaled, "w", REG, fluid, CELLS)

    assert np.all(np.asarray(points[2]) > np.asarray(points[1])), (
        "this case must be non-degenerate, or the second segment is zeroed")
    assert len(points) == 3, "critical, residual, upper"
    SCR, SR, SU = (np.asarray(p) for p in points)
    # first segment carries the critical point onto the table's
    np.testing.assert_allclose(m[0]*SCR + c[0], 0.2, rtol=1e-12)
    # and the two segments meet at the residual
    np.testing.assert_allclose(m[0]*SR + c[0], m[1]*SR + c[1], rtol=1e-9)


def test_a_degenerate_upper_interval_is_zeroed_not_divided_by():
    """No room between the residual and the maximum - scaling across it would
    divide by zero and put inf into the saturation map."""
    fluid = _fluid()
    # force SU <= SR by giving water an upper point below its residual
    scaled = _scaled(w=np.tile([[0.1, 0.2, 0.05, 1.0]], (N, 1)),
                     ow=np.tile([[0.0, 0.15, 0.9, 1.0]], (N, 1)),
                     g=np.tile([[0.0, 0.05, 0.85, 1.0]], (N, 1)))

    m, c, _, _ = SaturationProperty.get_three_point_scalers(
        scaled, "w", REG, fluid, CELLS)

    assert np.all(np.isfinite(m[1])) and np.all(np.isfinite(c[1]))
    np.testing.assert_allclose(m[1], 0.0)
    np.testing.assert_allclose(c[1], 0.0)


@pytest.mark.parametrize("phase", ["w", "g", "ow", "og"])
def test_every_phase_has_a_three_point_rule(phase):
    fluid = _fluid()
    scaled = _scaled(**{p: np.tile([[0.05, 0.15, 0.85, 1.0]], (N, 1))
                        for p in ("w", "g", "ow", "og")})
    m, c, points, k = SaturationProperty.get_three_point_scalers(
        scaled, phase, REG, fluid, CELLS)
    assert len(m) == 2 and len(c) == 2 and len(points) == 3
    assert len(k) == 2, "the maximum relperm pair comes back too"


def test_an_unknown_phase_is_refused_for_three_point():
    with pytest.raises(ValueError, match="three-point scalers"):
        SaturationProperty.get_three_point_scalers(
            _scaled(), "zz", REG, _fluid(), CELLS)
