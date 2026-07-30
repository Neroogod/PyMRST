"""Volume-balanced component flux for cross-flowing wells."""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.facility.well_component_total_flux import (
    apply_total_crossflow_mixing,
    sum_component_phase_flux,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class WellComponentTotalVolumeBalanceCrossflow(StateFunction):
    """Conservative surface-volume component mixing across perforations.

    MRST's mass-basis implementation converts incoming phase mass to volume,
    mixes it, then converts back. PyMRST's component equations already use
    surface volume, so those conversions cancel and the conservative mixture
    can be applied directly while retaining local AD derivatives.
    """

    only_local_derivatives = True

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="Q_i^{V-xflow}")
        self.depends_on([
            "FacilityWellMapping", "ComponentPhaseFlux",
            "InjectionSurfaceDensity",
        ])

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[Any]:
        mapping = self.get_evaluated(state, "FacilityWellMapping")
        phase_flux = self.get_evaluated(state, "ComponentPhaseFlux")
        self.get_evaluated(state, "InjectionSurfaceDensity")
        return apply_total_crossflow_mixing(
            facility, mapping, sum_component_phase_flux(phase_flux)
        )


__all__ = ["WellComponentTotalVolumeBalanceCrossflow"]
