"""Total component well flux with MRST crossflow mixing."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.facility.component_mixing import (
    crossflow_composition,
    well_component_composition,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value


def sum_component_phase_flux(phase_flux: Any) -> List[Any]:
    """Sum nonempty phase contributions for each component."""
    output: List[Any] = []
    for component_flux in phase_flux:
        total = None
        for flux in component_flux:
            if flux is not None:
                total = flux if total is None else total + flux
        output.append(total)
    return output


def apply_total_crossflow_mixing(
    facility: Any, mapping: Any, component_flux: List[Any]
) -> List[Any]:
    """Replace injecting connections by their wellbore component mixture."""
    present = [flux for flux in component_flux if flux is not None]
    if not present:
        return component_flux
    nperf = np.asarray(value(present[0])).size
    numeric = np.zeros((nperf, len(component_flux)))
    total = None
    for component, flux in enumerate(component_flux):
        if flux is None:
            continue
        numeric[:, component] = np.asarray(value(flux), dtype=float)
        total = flux if total is None else total + flux
    injecting = np.asarray(value(total), dtype=float) > 0.0
    if not np.any(injecting):
        return component_flux

    topside = well_component_composition(
        facility.reservoir_model, mapping.wells
    )
    mixture = crossflow_composition(
        numeric, topside, mapping, conserve_component_amount=True
    )
    perf_mixture = mixture[mapping.perf2well]
    for component, flux in enumerate(component_flux):
        if flux is not None:
            flux[injecting] = (
                total[injecting] * perf_mixture[injecting, component]
            )
    return component_flux


class WellComponentTotalFlux(StateFunction):
    """Sum phase fluxes and enforce component conservation through crossflow."""

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="Q_i")
        self.depends_on(["FacilityWellMapping", "ComponentPhaseFlux"])

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[Any]:
        mapping = self.get_evaluated(state, "FacilityWellMapping")
        phase_flux = self.get_evaluated(state, "ComponentPhaseFlux")
        total = sum_component_phase_flux(phase_flux)
        return apply_total_crossflow_mixing(facility, mapping, total)


__all__ = [
    "WellComponentTotalFlux",
    "apply_total_crossflow_mixing",
    "sum_component_phase_flux",
]
