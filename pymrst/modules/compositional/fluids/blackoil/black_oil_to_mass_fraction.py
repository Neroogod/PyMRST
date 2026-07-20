"""Port of MRST ``blackOilToMassFraction.m``.

The SPE1 compositional example first rewrites a dissolved-gas black-oil state as
two pseudo-components: ``PseudoOil`` and ``PseudoGas``.  These are mass
fractions, not mole fractions.  MRST then uses a K-value compositional model
built from the same black-oil tables.  This module ports the conversion step
directly so PyMRST can construct the same compositional state from a deck.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BlackOilMassFractions:
    """Pseudo-component fractions and phase properties from a black-oil state."""

    xo: np.ndarray
    xg: np.ndarray
    yo: np.ndarray
    yg: np.ndarray
    zo: np.ndarray
    zg: np.ndarray
    rhoO: np.ndarray
    rhoG: np.ndarray
    muO: np.ndarray
    muG: np.ndarray
    freeGas: np.ndarray


def black_oil_to_mass_fraction(fluid, p, sO, sG, rs, rv=0.0, *, disgas: bool = True) -> BlackOilMassFractions:
    """Evaluate MRST's black-oil pseudo-component mass fractions.

    Parameters follow MRST ``blackOilToMassFraction(model, p, sO, sG, rs, rv)``.
    ``rv`` is accepted for API parity; SPE1 is a dissolved-gas case and does not
    use vaporised oil.
    """
    del rv
    p = np.asarray(p, dtype=float).ravel()
    s_o = np.asarray(sO, dtype=float).ravel()
    s_g = np.asarray(sG, dtype=float).ravel()
    rs = np.broadcast_to(np.asarray(rs, dtype=float).ravel(), p.shape)
    rho_s = np.asarray(fluid.rhoS if hasattr(fluid, "rhoS") else fluid.rho_s, dtype=float)
    rho_os = float(rho_s[1])
    rho_gs = float(rho_s[2])

    if disgas:
        free_gas = rs >= _rs_sat(fluid, p)
        b_o = _b_o(fluid, p, rs, free_gas)
        mu_o = _mu_o(fluid, p, rs, free_gas)
        rho_o = b_o * (rs * rho_gs + rho_os)
    else:
        free_gas = s_g > 0.0
        b_o = _b_o(fluid, p, np.zeros_like(p), free_gas)
        mu_o = _mu_o(fluid, p, np.zeros_like(p), free_gas)
        rho_o = b_o * rho_os

    b_g = _b_g(fluid, p)
    mu_g = _mu_g(fluid, p)
    rho_g = b_g * rho_gs

    denom = rho_o * s_o + rho_g * s_g
    with np.errstate(divide="ignore", invalid="ignore"):
        zg = (rho_g * s_g + float(disgas) * rho_gs * rs * b_o * s_o) / denom
    zg = np.where(denom > 0.0, zg, 0.0)
    zo = 1.0 - zg

    xo = b_o * rho_os / rho_o
    xg = float(disgas) * b_o * rho_gs * rs / rho_o
    yo = np.zeros_like(p)
    yg = np.ones_like(p)

    return BlackOilMassFractions(
        xo=xo,
        xg=xg,
        yo=yo,
        yg=yg,
        zo=zo,
        zg=zg,
        rhoO=rho_o,
        rhoG=rho_g,
        muO=mu_o,
        muG=mu_g,
        freeGas=free_gas.astype(bool),
    )


def _rs_sat(fluid, p):
    return fluid.rs_sat(p)


def _b_o(fluid, p, rs, free_gas):
    if hasattr(fluid, "b_o"):
        return fluid.b_o(p, rs, free_gas)
    return fluid.b(p, rs, free_gas, None, None)[1]


def _mu_o(fluid, p, rs, free_gas):
    if hasattr(fluid, "mu_o"):
        return fluid.mu_o(p, rs, free_gas)
    return fluid.mu(p, rs, free_gas, None, None)[1]


def _b_g(fluid, p):
    if hasattr(fluid, "b_g"):
        return fluid.b_g(p)
    zeros = np.zeros_like(np.asarray(p, dtype=float))
    flags = np.zeros_like(zeros, dtype=bool)
    return fluid.b(p, zeros, flags, None, None)[2]


def _mu_g(fluid, p):
    if hasattr(fluid, "mu_g"):
        return fluid.mu_g(p)
    zeros = np.zeros_like(np.asarray(p, dtype=float))
    flags = np.zeros_like(zeros, dtype=bool)
    return fluid.mu(p, zeros, flags, None, None)[2]


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.modules.ad_props import init_deck_blackoil_fluid
    from pymrst.modules.deckformat import read_spe1_deck

    deck_path = Path(__file__).resolve().parents[6] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
    deck = read_spe1_deck(deck_path)
    fluid = init_deck_blackoil_fluid(deck)

    def test_spe1_initial_mass_fractions():
        s_o = 1.0 - deck["swat"] - deck["sgas"]
        out = black_oil_to_mass_fraction(fluid, deck["pressure"], s_o, deck["sgas"], deck["rs"])
        assert np.allclose(out.xo + out.xg, 1.0)
        assert np.allclose(out.yo + out.yg, 1.0)
        assert np.allclose(out.zo + out.zg, 1.0)
        assert np.all(out.rhoO > out.rhoG)
        assert out.freeGas.dtype == bool
        print("ok test_spe1_initial_mass_fractions")

    test_spe1_initial_mass_fractions()
    print("\nAll tests passed!")
