"""Density-mixture variant of total component well flux."""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.facility.well_component_total_flux import (
    apply_total_crossflow_mixing,
    sum_component_phase_flux,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class WellComponentTotalFluxDensityMix(StateFunction):
    """Crossflow total using per-well injection-density dependencies.

    PyMRST conserves components on a surface-volume basis. On this basis the
    density-mix construction reduces to the same conservative wellbore
    component mixture as ``WellComponentTotalFlux``; the surface-density
    dependency is still evaluated so compositional/custom-well overrides are
    represented in the graph.
    """

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="Q_i^{rho-mix}")
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


__all__ = ["WellComponentTotalFluxDensityMix"]
