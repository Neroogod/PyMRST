"""Compatibility imports for the relocated poromechanics model kernels.

Operational implementations live in :mod:`pymrst.modules.ad_mechanics.models`
so the package root no longer owns numerical code.  Existing user imports are
kept working for backward compatibility.
"""

from pymrst.modules.ad_mechanics.models.poromechanics import (
    FixedStressBlackOilWellModel,
    FixedStressOilWaterWellModel,
    FixedStressPoreVolume,
    FixedStressSplitModel,
    FixedStressWaterWellModel,
    MechWaterModel,
    MechWaterWellModel,
    PoroMechanicalPoreVolume,
)

__all__ = [
    "PoroMechanicalPoreVolume",
    "FixedStressPoreVolume",
    "MechWaterModel",
    "MechWaterWellModel",
    "FixedStressWaterWellModel",
    "FixedStressOilWaterWellModel",
    "FixedStressBlackOilWellModel",
    "FixedStressSplitModel",
]
