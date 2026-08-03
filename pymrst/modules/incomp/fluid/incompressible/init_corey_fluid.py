"""Corey incompressible fluid (MRST ``initCoreyFluid.m``).

Corey relative permeabilities with residual saturations and end-point values:

    s1 = clip((s_w - sr_w) / (1 - sr_w - sr_o), 0, 1)
    s2 = clip((1 - s_w - sr_o) / (1 - sr_w - sr_o), 0, 1)

    kr_w = kwm_w * s1^n_w
    kr_o = kwm_o * s2^n_o

The two extra parameters matter. `sr` is where each phase stops flowing, so
without it a fluid stays mobile down to zero saturation, which no rock is.
`kwm` is the relative permeability at the other end point, rarely 1 in a real
description. The Egg model's sequential example specifies both, and dropping
them changes the flood.

With `sr = (0, 0)` and `kwm = (1, 1)` this reduces to the plain Corey form in
`init_simple_fluid`, which is what the defaults give, so existing callers are
unaffected.
"""

from typing import Sequence

import numpy as np

from pymrst.modules.incomp.fluid import Fluid
from pymrst.modules.incomp.fluid.incompressible.init_simple_fluid import (
    init_simple_fluid,
)


def init_corey_fluid(
    mu: Sequence[float],
    rho: Sequence[float],
    n: Sequence[float] = (1.0, 1.0),
    sr: Sequence[float] = (0.0, 0.0),
    kwm: Sequence[float] = (1.0, 1.0),
) -> Fluid:
    """Construct MRST's Corey fluid.

    ``sr`` holds the residual saturations of the two phases and ``kwm`` their
    end-point relative permeabilities.
    """
    sr_arr = np.asarray(sr, dtype=float).ravel()
    kwm_arr = np.asarray(kwm, dtype=float).ravel()
    n_arr = np.asarray(n, dtype=float).ravel()
    if sr_arr.size != 2 or kwm_arr.size != 2 or n_arr.size != 2:
        raise ValueError("mu, rho, n, sr and kwm each describe two phases")
    if sr_arr.sum() >= 1.0:
        raise ValueError(
            f"residual saturations must leave a mobile range; got "
            f"sr={sr_arr.tolist()}")

    base = init_simple_fluid(mu=mu, rho=rho, n=n)
    if np.allclose(sr_arr, 0.0) and np.allclose(kwm_arr, 1.0):
        return base

    den = 1.0 - sr_arr.sum()

    def relperm(s, *args, **kwargs):
        sw = np.asarray(s, dtype=float)
        if sw.ndim > 1:
            sw = sw[:, 0]
        s1 = np.clip((sw - sr_arr[0]) / den, 0.0, 1.0)
        s2 = np.clip((1.0 - sw - sr_arr[1]) / den, 0.0, 1.0)
        return np.squeeze(np.stack([kwm_arr[0] * s1 ** n_arr[0],
                                    kwm_arr[1] * s2 ** n_arr[1]], axis=-1))

    fluid = _clone_with_relperm(base, relperm)
    return fluid


def _clone_with_relperm(base: Fluid, relperm) -> Fluid:
    """A copy of `base` with its relperm replaced.

    `Fluid` may be a frozen dataclass or a plain object depending on the
    construction path, so the assignment is attempted directly and falls back
    to `object.__setattr__` rather than assuming either.
    """
    import copy
    fluid = copy.copy(base)
    try:
        fluid.relperm = relperm
    except Exception:
        object.__setattr__(fluid, "relperm", relperm)
    return fluid


__all__ = ["Fluid", "init_corey_fluid"]


if __name__ == "__main__":
    from pymrst.utils.units import CENTI, POISE

    def _egg_fluid():
        """The fluid the Egg sequential example specifies."""
        return init_corey_fluid(
            mu=[1.0 * CENTI * POISE, 5.0 * CENTI * POISE],
            rho=[1000.0, 900.0], n=[3.0, 4.0],
            sr=[0.2, 0.1], kwm=[0.749, 0.8])

    def test_defaults_reduce_to_the_plain_corey_model():
        """`sr = 0`, `kwm = 1` must reproduce `init_simple_fluid` exactly.

        Existing callers pass neither, so the defaults must leave their fluid
        unchanged.
        """
        a = init_corey_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 900.0], n=[2.0, 2.0])
        b = init_simple_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 900.0], n=[2.0, 2.0])
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert np.allclose(np.asarray(a.relperm(s), dtype=float).ravel(),
                               np.asarray(b.relperm(s), dtype=float).ravel()), s
        print("ok test_defaults_reduce_to_the_plain_corey_model")

    def test_relperm_vanishes_at_residual_saturation():
        """Below its residual saturation a phase does not flow."""
        fluid = _egg_fluid()
        assert np.isclose(np.asarray(fluid.relperm(0.2), float).ravel()[0], 0.0,
                          atol=1e-14)
        assert np.isclose(np.asarray(fluid.relperm(0.1), float).ravel()[0], 0.0,
                          atol=1e-14), "water mobile below its residual"
        assert np.isclose(np.asarray(fluid.relperm(0.9), float).ravel()[1], 0.0,
                          atol=1e-14), "oil mobile below its residual"
        print("ok test_relperm_vanishes_at_residual_saturation")

    def test_end_points_reach_kwm():
        fluid = _egg_fluid()
        assert np.isclose(np.asarray(fluid.relperm(0.9), float).ravel()[0], 0.749,
                          rtol=1e-12)
        assert np.isclose(np.asarray(fluid.relperm(0.2), float).ravel()[1], 0.8,
                          rtol=1e-12)
        print("ok test_end_points_reach_kwm")

    def test_matches_the_mrst_formula():
        """Checked against MRST's `modified_saturations` and relperm lines."""
        fluid = _egg_fluid()
        sr, kwm = np.array([0.2, 0.1]), np.array([0.749, 0.8])
        n = np.array([3.0, 4.0])
        den = 1.0 - sr.sum()
        for sw in (0.25, 0.4, 0.55, 0.7, 0.85):
            s1 = min(max((sw - sr[0]) / den, 0.0), 1.0)
            s2 = min(max((1.0 - sw - sr[1]) / den, 0.0), 1.0)
            want = [kwm[0] * s1 ** n[0], kwm[1] * s2 ** n[1]]
            got = np.asarray(fluid.relperm(sw), dtype=float).ravel()
            assert np.allclose(got, want, rtol=1e-12), (sw, got, want)
        print("ok test_matches_the_mrst_formula")

    def test_relperm_is_monotone_between_the_end_points():
        fluid = _egg_fluid()
        s = np.linspace(0.2, 0.9, 40)
        kr = np.asarray([np.asarray(fluid.relperm(v), float).ravel() for v in s])
        assert np.all(np.diff(kr[:, 0]) >= -1e-15), "water relperm not increasing"
        assert np.all(np.diff(kr[:, 1]) <= 1e-15), "oil relperm not decreasing"
        print("ok test_relperm_is_monotone_between_the_end_points")

    def test_bad_inputs_are_rejected():
        try:
            init_corey_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 900.0], sr=[0.7, 0.4])
        except ValueError as exc:
            assert "mobile range" in str(exc), exc
        else:
            raise AssertionError("expected residual saturations summing past 1 to fail")
        try:
            init_corey_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 900.0], sr=[0.1])
        except ValueError:
            pass
        else:
            raise AssertionError("expected a one-element sr to fail")
        print("ok test_bad_inputs_are_rejected")

    test_defaults_reduce_to_the_plain_corey_model()
    test_relperm_vanishes_at_residual_saturation()
    test_end_points_reach_kwm()
    test_matches_the_mrst_formula()
    test_relperm_is_monotone_between_the_end_points()
    test_bad_inputs_are_rejected()
    print("\nAll tests passed!")
