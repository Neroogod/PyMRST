"""Backward-compatible import path for the AD-core component classes.

The implementation lives in `ad_core.statefunctions.components.components`,
mirroring MRST's `ad-core/statefunctions/components/` (`ImmiscibleComponent.m`,
`GenericComponent.m`, `ConcentrationComponent.m`).
"""

from pymrst.modules.ad_core.statefunctions.components.components import (
    Component,
    ConcentrationComponent,
    GasComponent,
    ImmiscibleComponent,
    OilComponent,
    blackoil_components,
)

__all__ = [
    "Component",
    "ConcentrationComponent",
    "GasComponent",
    "ImmiscibleComponent",
    "OilComponent",
    "blackoil_components",
]
