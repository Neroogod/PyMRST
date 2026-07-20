"""MRST ad-core equilibrium initialisation (`ad-core/initialization/`).

The implementation lives in `equilibrium.py`; this package surface mirrors
MRST's `initialization/` directory (`initializeEquilibriumPressures.m`,
`initializeEquilibriumSaturations.m`, `initStateBlackOilAD.m`).
"""

from pymrst.modules.ad_core.initialization.equilibrium import (
    EquilibriumRegion,
    init_state_equil,
    initialize_equilibrium_pressures,
    initialize_equilibrium_saturations,
)

__all__ = [
    "EquilibriumRegion",
    "init_state_equil",
    "initialize_equilibrium_pressures",
    "initialize_equilibrium_saturations",
]
