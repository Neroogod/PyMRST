"""Reservoir phase densities to inverse formation-volume factors (MRST
`phaseDensitiesTobfactor`).

A black-oil model stores each phase by its *b-factor* `b = 1/B`, the ratio of
reservoir volume to surface volume, rather than by density. Given the reservoir
densities `rho` and the surface densities `rhoS`, the b-factor of a phase is

    b_i = rho_i / (rhoS_i + sum_j rhoS_j * r_ji)

where `r_ji` is the amount of surface phase `j` dissolved per surface volume of
phase `i` (the black-oil `rs`/`rv`). With no dissolution the denominator is just
`rhoS_i`, so `b = rho/rhoS` - the definition of the shrinkage factor. This is the
inverse of the density state function and is used wherever a model needs to go
from a computed reservoir density back to a conserved surface volume.
"""

from typing import List, Optional, Sequence

import numpy as np


def phase_densities_to_bfactor(
    rho: Sequence[np.ndarray],
    rho_s: Sequence[float],
    dissolved: Optional[Sequence[Sequence[Optional[np.ndarray]]]] = None,
) -> List[np.ndarray]:
    """`b_i = rho_i / (rhoS_i + sum_j rhoS_j * dissolved[j][i])` per phase.

    `dissolved[j][i]` is the surface volume of phase `j` dissolved per surface
    volume of phase `i` (e.g. `dissolved[gas][oil] = rs`), or `None` where there
    is none. Mirrors MRST's cell-array branch: `rho` and the result are lists,
    one array per phase.
    """
    n = len(rho_s)
    b: List[np.ndarray] = [None] * n            # type: ignore[list-item]
    for i in range(n):
        factor = rho_s[i]
        if dissolved is not None:
            for j in range(n):
                r_ph = dissolved[j][i]
                if r_ph is not None:
                    factor = factor + rho_s[j] * r_ph
        b[i] = rho[i] / factor
    return b


if __name__ == "__main__":
    def test_no_dissolution_is_rho_over_rhos():
        """Dead oil/water: b is just rho/rhoS phase by phase."""
        rho = [np.array([1000.0, 1005.0]), np.array([800.0, 790.0])]
        rho_s = [1000.0, 850.0]
        b = phase_densities_to_bfactor(rho, rho_s)
        assert np.allclose(b[0], rho[0] / 1000.0)
        assert np.allclose(b[1], rho[1] / 850.0)
        print("ok test_no_dissolution_is_rho_over_rhos")

    def test_live_oil_denominator_carries_dissolved_gas():
        """With rs gas in oil, the oil denominator is rhoS_o + rhoS_g * rs."""
        rho = [np.array([1000.0]), np.array([850.0]), np.array([1.2])]
        rho_s = [1000.0, 800.0, 1.0]        # water, oil, gas
        rs = np.array([120.0])
        # dissolved[gas][oil] = rs, everything else None
        dissolved = [[None, None, None],
                     [None, None, None],
                     [None, rs, None]]
        b = phase_densities_to_bfactor(rho, rho_s, dissolved)
        assert np.allclose(b[1], 850.0 / (800.0 + 1.0 * 120.0))
        assert np.allclose(b[0], 1000.0 / 1000.0)   # water unaffected
        assert np.allclose(b[2], 1.2 / 1.0)         # gas unaffected
        print("ok test_live_oil_denominator_carries_dissolved_gas")

    test_no_dissolution_is_rho_over_rhos()
    test_live_oil_denominator_carries_dissolved_gas()
    print("\nAll tests passed!")
