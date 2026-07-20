"""CO2 (and H2) storage: sampled fluid properties and the brine-gas model.

Ports of MRST's co2lab-common property tables and co2lab-ve's
TwoPhaseWaterGasModel.
"""

from pymrst.modules.co2lab_common.co2_brine import (
    CO2BrineModel,
    WaterGasFluid,
    WGState,
    co2_brine_fluid,
)
from pymrst.modules.co2lab_common.co2props import CO2Props, SampledProp2D, co2_props
from pymrst.modules.co2lab_common.h2props import h2_brine_fluid, h2_props
from pymrst.modules.co2lab_spillpoint.trapping import (
    TrapAnalysis,
    downstream_traps,
    trap_analysis,
    volumes_of_traps,
)
from pymrst.modules.co2lab_ve.ve import (
    TopSurfaceGrid,
    VEFluid,
    VEModel,
    VEState,
    average_rock,
    top_surface_grid,
)

__all__ = [
    "SampledProp2D", "CO2Props", "co2_props",
    "WaterGasFluid", "co2_brine_fluid", "CO2BrineModel", "WGState",
    "h2_props", "h2_brine_fluid",
    "TopSurfaceGrid", "top_surface_grid", "average_rock",
    "VEFluid", "VEModel", "VEState",
    "TrapAnalysis", "trap_analysis", "volumes_of_traps", "downstream_traps",
]

from pymrst.modules.co2lab_mit.inventory import (
    CATEGORIES,
    Inventory,
    inventory_history,
    mass_trapping_distribution,
    trap_depth,
)
from pymrst.modules.co2lab_spillpoint.trapping import (
    TrapTree,
    downstream_closure,
    find_optimal_injection_point,
    maximize_trapping,
)

__all__ = list(globals().get("__all__", [])) + [
    "mass_trapping_distribution", "inventory_history", "Inventory",
    "CATEGORIES", "trap_depth",
    "maximize_trapping", "find_optimal_injection_point", "downstream_closure",
    "TrapTree",
]

from pymrst.modules.co2lab_mit.co2_brine_props import (
    brine_co2_viscosity,
    brine_density_rc1970,
    co2_activity_ds2003,
    co2_density,
    co2_density_rk1949,
    co2_viscosity,
    co2_viscosity_f1998,
    gas_mixture_viscosity,
)

__all__ = list(globals().get("__all__", [])) + [
    "co2_density_rk1949", "co2_viscosity_f1998", "brine_density_rc1970",
    "brine_co2_viscosity", "co2_activity_ds2003", "gas_mixture_viscosity",
    "co2_density", "co2_viscosity",
]

from pymrst.modules.co2lab_ve.ve_dissolution import (
    DissolutionFluid,
    VEDissolutionModel,
    VEDisState,
)

__all__ = list(globals().get("__all__", [])) + [
    "VEDissolutionModel", "DissolutionFluid", "VEDisState",
]

from pymrst.modules.co2lab_mit.atlas import (
    atlas_grid,
    available_formations,
    convert_atlas_to_3d,
    read_aai_grid,
    read_atlas_dataset,
)

__all__ = list(globals().get("__all__", [])) + [
    "atlas_grid", "read_aai_grid", "read_atlas_dataset",
    "convert_atlas_to_3d", "available_formations",
]

from pymrst.modules.co2lab_ve.ve_fluids import (
    VECapillaryFringeFluid,
    VECapLinearFluid,
    VETable,
    free_sg,
    make_ve_table_h,
    make_ve_table_p,
    upper_cap_pressure,
)

__all__ = list(globals().get("__all__", [])) + [
    "VECapillaryFringeFluid", "VECapLinearFluid", "VETable", "free_sg",
    "make_ve_table_p", "make_ve_table_h", "upper_cap_pressure",
]

from pymrst.modules.co2lab_spillpoint.node_trapping import (
    NodeTrapAnalysis,
    node_trap_analysis,
    nodes_to_cells,
)

__all__ = list(globals().get("__all__", [])) + [
    "node_trap_analysis", "nodes_to_cells", "NodeTrapAnalysis",
]
