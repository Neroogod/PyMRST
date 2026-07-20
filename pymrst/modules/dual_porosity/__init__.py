"""Dual-porosity flow in naturally fractured rock (MRST dual-porosity/)."""

from pymrst.modules.dual_porosity.dp_model import (
    DPOilWaterState,
    DPWaterState,
    TwoPhaseOilWaterModelDP,
    WaterModelDP,
)
from pymrst.modules.dual_porosity.transfer_models.shape_factor_models import (
    CoatsShapeFactor,
    ConstantShapeFactor,
    KazemiShapeFactor,
    LimAzizShapeFactor,
    ShapeFactor,
    WarrenRootShapeFactor,
)
from pymrst.modules.dual_porosity.transfer_models import (
    KazemiSinglePhaseTransfer,
    KazemiTwoPhaseTransfer,
)

__all__ = [
    "ShapeFactor", "ConstantShapeFactor", "KazemiShapeFactor",
    "WarrenRootShapeFactor", "CoatsShapeFactor", "LimAzizShapeFactor",
    "KazemiSinglePhaseTransfer", "KazemiTwoPhaseTransfer",
    "WaterModelDP", "TwoPhaseOilWaterModelDP",
    "DPWaterState", "DPOilWaterState",
]
