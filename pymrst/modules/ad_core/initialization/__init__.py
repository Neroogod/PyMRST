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
from pymrst.modules.ad_core.initialization.regions import (
    get_initialization_regions_base,
    get_min_max_phase_saturations,
    init_state_equil_regions,
)
from pymrst.modules.ad_core.initialization.get_min_max_phase_saturations_from_rel_perm import (
    get_min_max_phase_saturations_from_rel_perm,
    getMinMaxPhaseSaturationsFromRelPerm,
)

__all__ = [
    "EquilibriumRegion",
    "get_initialization_regions_base",
    "get_min_max_phase_saturations",
    "get_min_max_phase_saturations_from_rel_perm",
    "getMinMaxPhaseSaturationsFromRelPerm",
    "init_state_equil",
    "init_state_equil_regions",
    "initialize_equilibrium_pressures",
    "initialize_equilibrium_saturations",
]
