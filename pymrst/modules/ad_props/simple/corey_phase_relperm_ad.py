"""MRST ``coreyPhaseRelpermAD.m`` Corey relative permeability."""


def corey_phase_relperm_ad(s, exponent=1.0):
    """Corey relative permeability `kr = s**exponent`.

    Works for NumPy arrays and PyMRST ADI values through Python's power
    operator, matching the simple AD relperm path used by PyMRST fluids.
    """

    return s ** exponent


__all__ = ["corey_phase_relperm_ad"]
