"""Compatibility landing file for MRST `PressureOverallCompositionModel.m`.

PyMRST exposes the pressure-only solve as
`OverallCompositionModel.pressure_step`.
"""

from pymrst.modules.compositional.model import (
    CompositionalState,
    OverallCompositionModel,
)

__all__ = ["OverallCompositionModel", "CompositionalState"]
