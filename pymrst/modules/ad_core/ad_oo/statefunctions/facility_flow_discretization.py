"""MRST ``ad-core/statefunctions/FacilityFlowDiscretization.m``.

The facility's grouping: perforation pressure differences, well indices and
mobilities, the phase and component fluxes between wells and cells, and the
injection surface densities. The generic facility model assembles its well
equations from it; the standard `FacilityModel` builds it but does not
evaluate it.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.ad_oo.statefunctions.facility.facility_well_mapping import (
    FacilityWellMapping,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.injection_surface_density import (
    InjectionSurfaceDensity,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.perforation_component_phase_density import (
    PerforationComponentPhaseDensity,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.perforation_mobility import (
    PerforationMobility,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.perforation_pressure_gradient import (
    PerforationPressureGradient,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.well_component_phase_flux import (
    WellComponentPhaseFlux,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.well_index import WellIndex
from pymrst.modules.ad_core.ad_oo.statefunctions.facility.well_phase_flux import (
    WellPhaseFlux,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.component_total_flux import (
    ComponentTotalFlux,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    StateFunctionGrouping,
)


class FacilityFlowDiscretization(StateFunctionGrouping):
    """MRST's `FacilityFluxProps` container."""

    def __init__(self, model: Any):
        super().__init__("FacilityFluxProps")
        self.set_state_function("PhaseFlux", WellPhaseFlux(model))
        ctf = ComponentTotalFlux(model)
        ctf.label = "Q_i"
        self.set_state_function("ComponentTotalFlux", ctf)
        self.set_state_function("ComponentPhaseFlux", WellComponentPhaseFlux(model))
        self.set_state_function("PressureGradient", PerforationPressureGradient(model))
        self.set_state_function("WellIndex", WellIndex(model))
        self.set_state_function("FacilityWellMapping", FacilityWellMapping(model))
        self.set_state_function("InjectionSurfaceDensity",
                                InjectionSurfaceDensity(model))
        self.set_state_function("Mobility", PerforationMobility(model))
        self.set_state_function("ComponentPhaseDensity",
                                PerforationComponentPhaseDensity(model))


__all__ = ["FacilityFlowDiscretization"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_the_grouping_carries_mrsts_functions():
        group = FacilityFlowDiscretization(SimpleNamespace(AutoDiffBackend=None))
        assert group.struct_name == "FacilityFluxProps"
        assert set(group.get_names()) == {
            "PhaseFlux", "ComponentTotalFlux", "ComponentPhaseFlux",
            "PressureGradient", "WellIndex", "FacilityWellMapping",
            "InjectionSurfaceDensity", "Mobility", "ComponentPhaseDensity"}
        print("ok test_the_grouping_carries_mrsts_functions")

    test_the_grouping_carries_mrsts_functions()
    print("\nAll tests passed!")
