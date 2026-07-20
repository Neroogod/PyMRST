"""Single-phase dual-porosity residual (MRST ``equationsWaterDP.m``)."""

from pymrst.modules.dual_porosity.dp_model import DPWaterState, WaterModelDP


def equations_water_dp(model: WaterModelDP, state0: DPWaterState, p, pm, dt: float):
    """Return fracture and matrix water equations for a DP water model."""
    return model.assemble(state0, p, pm, dt)


__all__ = ["DPWaterState", "WaterModelDP", "equations_water_dp"]
