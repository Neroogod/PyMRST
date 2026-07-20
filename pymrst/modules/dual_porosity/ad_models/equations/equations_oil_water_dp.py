"""Oil-water dual-porosity residual (MRST ``equationsOilWaterDP.m``)."""

from pymrst.modules.dual_porosity.dp_model import (
    DPOilWaterState,
    TwoPhaseOilWaterModelDP,
)


def equations_oil_water_dp(
    model: TwoPhaseOilWaterModelDP,
    state0: DPOilWaterState,
    p,
    sw,
    pm,
    swm,
    dt: float,
):
    """Return fracture/matrix water and oil equations for a two-phase DP model."""
    return model.assemble(state0, p, sw, pm, swm, dt)


__all__ = ["DPOilWaterState", "TwoPhaseOilWaterModelDP", "equations_oil_water_dp"]
