"""MRST ad-core facility state functions (`.../statefunctions/facility/`).

The well-flux half of the state-function graph: given the reservoir state and
the active wells, compute the flux between wellbore and reservoir per
perforation. Hosted on a light `FacilityContext` (the graph counterpart of
MRST's facility model) so the well physics composes as state functions rather
than one procedural method.

Core landed: `FacilityWellMapping`, `WellIndex`, `PerforationPressureGradient`,
`PerforationMobility`, `WellPhaseFlux` (validated against PyMRST's procedural
`_single_well`). The component-well fluxes (`WellComponentPhaseFlux`,
`WellComponentTotalFlux`, `WellComponentTotalFluxDensityMix`,
`WellComponentTotalVolumeBalanceCrossflow`), `PerforationComponentPhaseDensity`
and `InjectionSurfaceDensity` - the crossflow-mixture layer - are a follow-on.
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

__all__ = [
    "FacilityContext",
    "FacilityFlowDiscretization",
    "FacilityWellMapping",
    "PerforationMobility",
    "PerforationPressureGradient",
    "WellIndex",
    "WellMapping",
    "WellPhaseFlux",
    "build_well_mapping",
    "calculate_phase_rate",
]
