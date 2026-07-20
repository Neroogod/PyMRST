"""Incompressible well pressure drop, ported from MRST helper logic."""

import numpy as np

from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.params.wells_and_bc.types import Well


def incomp_well_pressure_drop(
    w: Well,
    mob: np.ndarray,
    rho: np.ndarray,
    norm_g: float = NORM_GRAVITY,
) -> np.ndarray:
    """Return hydrostatic connection pressure drop for a well."""
    n_phases = rho.size
    injector = w.sign > 0 or (w.type == "rate" and w.val > 0)
    if injector:
        n_comp = min(n_phases, w.compi.size)
        rho_mix = float(rho[:n_comp] @ w.compi[:n_comp])
    else:
        mob_well = mob[w.cells]
        frac = mob_well / mob_well.sum(axis=1, keepdims=True)
        n_comp = min(n_phases, mob.shape[1])
        rho_mix = float((w.WI * (frac[:, :n_comp] @ rho[:n_comp])).sum() / w.WI.sum())
    return norm_g * w.dZ * rho_mix


if __name__ == "__main__":

    def test_injector_pressure_drop() -> None:
        well = Well(
            cells=np.array([0, 1]),
            type="rate",
            val=1.0,
            r=np.full(2, 0.1),
            dir="z",
            WI=np.ones(2),
            dZ=np.array([10.0, 20.0]),
            name="W1",
            compi=np.array([1.0]),
            refDepth=0.0,
            sign=1,
        )
        dp = incomp_well_pressure_drop(well, np.ones((2, 1)), np.array([1000.0]))
        assert np.allclose(dp, NORM_GRAVITY * 1000.0 * np.array([10.0, 20.0]))
        print("ok test_injector_pressure_drop")

    test_injector_pressure_drop()
    print("\nAll tests passed!")

