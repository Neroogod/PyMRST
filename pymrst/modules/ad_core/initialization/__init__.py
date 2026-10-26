"""MRST ad-core equilibrium initialisation (`ad-core/initialization/`).

Each MRST file has its port in the module of the same name
(`initializeEquilibriumPressures.m` -> `initialize_equilibrium_pressures.py`,
...). PyMRST's earlier versions keep an `_ad` suffix; most live in
`equilibrium.py` and `regions.py`.
"""

from pymrst.modules.ad_core.initialization.get_initialization_regions_base import (
    get_initialization_regions_base,
    getInitializationRegionsBase,
)
from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
    get_initialization_regions_black_oil,
    get_initialization_regions_black_oil_ad,
    getInitializationRegionsBlackOil,
)
from pymrst.modules.ad_core.initialization.get_min_max_phase_saturations import (
    get_min_max_phase_saturations,
    getMinMaxPhaseSaturations,
)
from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
    init_state_black_oil_ad,
    init_state_black_oil_ad_ad,
    initStateBlackOilAD,
)
from pymrst.modules.ad_core.initialization.initialize_equilibrium_pressures import (
    initialize_equilibrium_pressures,
    initializeEquilibriumPressures,
)
from pymrst.modules.ad_core.initialization.initialize_equilibrium_saturations import (
    initialize_equilibrium_saturations,
    initializeEquilibriumSaturations,
)
from pymrst.modules.ad_core.initialization.equilibrium import (
    EquilibriumRegion,
    init_state_equil,
    initialize_equilibrium_pressures_ad,
    initialize_equilibrium_saturations_ad,
)
from pymrst.modules.ad_core.initialization.regions import (
    get_initialization_regions_base_ad,
    get_min_max_phase_saturations_ad,
    init_state_equil_regions,
)
from pymrst.modules.ad_core.initialization.get_min_max_phase_saturations_from_rel_perm import (
    get_min_max_phase_saturations_from_rel_perm,
    getMinMaxPhaseSaturationsFromRelPerm,
)
from pymrst.modules.ad_core.initialization.get_initialization_regions_deck import (
    get_initialization_regions_deck,
    get_initialization_regions_deck_ad,
    getInitializationRegionsDeck,
)
from pymrst.modules.ad_core.initialization.get_initialization_regions_compositional import (
    get_initialization_regions_compositional,
    getInitializationRegionsCompositional,
)
from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
    init_eclipse_problem_ad,
    init_eclipse_problem_ad_ad,
    initEclipseProblemAD,
)

__all__ = [
    "EquilibriumRegion",
    "get_initialization_regions_base",
    "getInitializationRegionsBase",
    "get_initialization_regions_black_oil",
    "get_initialization_regions_black_oil_ad",
    "getInitializationRegionsBlackOil",
    "get_initialization_regions_deck",
    "get_min_max_phase_saturations",
    "getMinMaxPhaseSaturations",
    "init_eclipse_problem_ad",
    "init_state_black_oil_ad",
    "init_state_black_oil_ad_ad",
    "initStateBlackOilAD",
    "initialize_equilibrium_pressures",
    "initializeEquilibriumPressures",
    "initialize_equilibrium_saturations",
    "initializeEquilibriumSaturations",
    "get_initialization_regions_base_ad",
    "get_min_max_phase_saturations_ad",
    "get_min_max_phase_saturations_from_rel_perm",
    "getMinMaxPhaseSaturationsFromRelPerm",
    "get_initialization_regions_deck_ad",
    "getInitializationRegionsDeck",
    "get_initialization_regions_compositional",
    "getInitializationRegionsCompositional",
    "init_eclipse_problem_ad_ad",
    "initEclipseProblemAD",
    "init_state_equil",
    "init_state_equil_regions",
    "initialize_equilibrium_pressures_ad",
    "initialize_equilibrium_saturations_ad",
]
