"""MRST ad-core facility state functions (`.../statefunctions/facility/`).

The well-flux half of the state-function graph: given the reservoir state and
the active wells, compute the flux between wellbore and reservoir per
perforation. Hosted on a light `FacilityContext` (the graph counterpart of
MRST's facility model) so the well physics composes as state functions rather
than one procedural method.

Includes the complete core and component/crossflow chain, from well mapping,
well index, pressure gradient, mobility, and phase flux through perforation
component density, injection density, component phase flux, and all three
component-total mixing formulations.
"""

from pymrst.modules.ad_core.statefunctions.facility.facility_flow_discretization import (  # noqa: E501
    FacilityContext,
    FacilityFlowDiscretization,
    WellMapping,
    build_well_mapping,
)
from pymrst.modules.ad_core.statefunctions.facility.facility_well_mapping import (
    FacilityWellMapping,
)
from pymrst.modules.ad_core.statefunctions.facility.well_index import WellIndex
from pymrst.modules.ad_core.statefunctions.facility.perforation_pressure_gradient import (  # noqa: E501
    PerforationPressureGradient,
)
from pymrst.modules.ad_core.statefunctions.facility.perforation_mobility import (
    PerforationMobility,
)
from pymrst.modules.ad_core.statefunctions.facility.well_phase_flux import (
    WellPhaseFlux,
    calculate_phase_rate,
)
from pymrst.modules.ad_core.statefunctions.facility.injection_surface_density import (
    InjectionSurfaceDensity,
)
from pymrst.modules.ad_core.statefunctions.facility.perforation_component_phase_density import (
    PerforationComponentPhaseDensity,
)
from pymrst.modules.ad_core.statefunctions.facility.well_component_phase_flux import (
    WellComponentPhaseFlux,
)
from pymrst.modules.ad_core.statefunctions.facility.well_component_total_flux import (
    WellComponentTotalFlux,
)
from pymrst.modules.ad_core.statefunctions.facility.well_component_total_flux_density_mix import (
    WellComponentTotalFluxDensityMix,
)
from pymrst.modules.ad_core.statefunctions.facility.well_component_total_volume_balance_crossflow import (
    WellComponentTotalVolumeBalanceCrossflow,
)

__all__ = [
    "FacilityContext",
    "FacilityFlowDiscretization",
    "FacilityWellMapping",
    "InjectionSurfaceDensity",
    "PerforationComponentPhaseDensity",
    "PerforationMobility",
    "PerforationPressureGradient",
    "WellIndex",
    "WellMapping",
    "WellPhaseFlux",
    "WellComponentPhaseFlux",
    "WellComponentTotalFlux",
    "WellComponentTotalFluxDensityMix",
    "WellComponentTotalVolumeBalanceCrossflow",
    "build_well_mapping",
    "calculate_phase_rate",
]
