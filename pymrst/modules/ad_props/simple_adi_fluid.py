"""Simple AD-ready fluid, ported from MRST initSimpleADIFluid (phases 'WO').

Shrinkage factors b(p) = exp(c (p - p_ref)) give constant-compressibility
phases; relative permeabilities are power laws of each phase's own
saturation. All property functions accept NumPy arrays or ADI variables.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from pymrst.utils.adi import ad_exp


@dataclass
class ADFluid:
    """Two-phase (water, oil) fluid for the AD-OO models."""

    rhoS: np.ndarray          # (2,) surface densities [water, oil]
    mu: np.ndarray            # (2,) constant viscosities
    b_w: Callable             # shrinkage factor of pressure
    b_o: Callable
    kr_w: Callable            # relperm of own saturation
    kr_o: Callable
    # pcOW(sW) = p_oil - p_water. None means the phases share one pressure,
    # which is what every PyMRST model assumed before capillarity existed.
    pc_ow: Optional[Callable] = None


def init_simple_adi_fluid(
    mu: Sequence[float],
    rho: Sequence[float],
    n: Sequence[float] = (1.0, 1.0),
    c: Sequence[float] = (0.0, 0.0),
    p_ref: float = 0.0,
    pc_ow: Optional[Callable] = None,
) -> ADFluid:
    """Water-oil fluid with b = exp(c (p - p_ref)) and kr = s^n.

    `pc_ow(sW)` is MRST's `fluid.pcOW`: the capillary pressure `p_o - p_w`,
    a decreasing function of water saturation. Omit it and the phases share
    a single pressure, as before.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    n = np.asarray(n, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    assert mu.size == rho.size == n.size == c.size == 2
    assert np.all(c >= 0), "compressibilities must be non-negative"

    def make_b(ci: float) -> Callable:
        if ci == 0.0:
            return lambda p: p * 0.0 + 1.0
        return lambda p: ad_exp(ci * (p - p_ref))

    return ADFluid(
        rhoS=rho,
        mu=mu,
        b_w=make_b(c[0]),
        b_o=make_b(c[1]),
        kr_w=lambda s: s ** n[0],
        kr_o=lambda s: s ** n[1],
        pc_ow=pc_ow,
    )


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi, value
    from pymrst.utils.units import BARSA

    def test_b_factors():
        """b(p_ref) = 1; slope equals compressibility."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA],
                                  p_ref=200 * BARSA)
        (p,) = init_variables_adi(np.array([200 * BARSA, 210 * BARSA]))
        bw = f.b_w(p)
        assert np.isclose(value(bw)[0], 1.0)
        assert np.isclose(bw.jac[0].toarray()[0, 0], 1e-6 / BARSA)
        assert np.isclose(value(f.b_o(p))[1],
                          np.exp(1e-4 / BARSA * 10 * BARSA))
        print("ok test_b_factors")

    def test_incompressible_b_is_one():
        """c = 0 gives b identically 1 (also through ADI)."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0])
        (p,) = init_variables_adi(np.array([1e7, 2e7]))
        b = f.b_w(p)
        assert np.allclose(value(b), 1.0)
        assert b.jac[0].count_nonzero() == 0
        print("ok test_incompressible_b_is_one")

    def test_relperms():
        """Power-law relperms of own saturation."""
        f = init_simple_adi_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                                  n=[2, 3])
        assert np.isclose(f.kr_w(np.array([0.5]))[0], 0.25)
        assert np.isclose(f.kr_o(np.array([0.5]))[0], 0.125)
        print("ok test_relperms")

    test_b_factors()
    test_incompressible_b_is_one()
    test_relperms()
    print("\nAll tests passed!")
