"""Perforation component flux split by reservoir phase."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from pymrst.modules.ad_core.statefunctions.facility.component_mixing import (
    crossflow_composition,
    well_component_composition,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value


class WellComponentPhaseFlux(StateFunction):
    """Compute ``q_alpha * rho*x`` for every component and phase.

    ``WellPhaseFlux`` already selects reservoir mobility for producing
    connections and total-mobility/well composition for injecting ones.
    Multiplication by the sampled component density therefore yields the
    component flow on PyMRST's configured component basis and retains the full
    AD Jacobian.
    """

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="q_{i,alpha}^W")
        self.depends_on(
            ["FacilityWellMapping", "WellPhaseFlux", "ComponentPhaseDensity"]
        )

    def evaluate_on_domain(
        self, facility: Any, state: Any
    ) -> List[List[Optional[Any]]]:
        mapping = self.get_evaluated(state, "FacilityWellMapping")
        phase_flux = self.get_evaluated(state, "WellPhaseFlux")
        density = self.get_evaluated(state, "ComponentPhaseDensity")
        result = [
            [None if component_density is None
             else component_density * phase_flux[phase]
             for phase, component_density in enumerate(component)]
            for component in density
        ]
        if not mapping.wells:
            return result

        # Re-injection at a cross-flowing connection must carry the mixture
        # entering that wellbore, rather than blindly using the declared
        # topside composition. Fractions are frozen from values, while the
        # total component amount retains its AD derivatives.
        injecting = np.sum(
            np.column_stack([value(flux) for flux in phase_flux]), axis=1
        ) > 0.0
        if not np.any(injecting):
            return result
        reservoir = facility.reservoir_model
        for phase, q_phase in enumerate(phase_flux):
            numeric = np.zeros((injecting.size, len(result)))
            phase_total = None
            for component in range(len(result)):
                flux = result[component][phase]
                if flux is None:
                    continue
                numeric[:, component] = np.asarray(value(flux), dtype=float)
                phase_total = flux if phase_total is None else phase_total + flux
            if phase_total is None:
                continue
            topside = well_component_composition(
                reservoir, mapping.wells, phase=phase
            )
            mixture = crossflow_composition(numeric, topside, mapping)
            perf_mixture = mixture[mapping.perf2well]
            for component in range(len(result)):
                flux = result[component][phase]
                if flux is not None:
                    flux[injecting] = (
                        phase_total[injecting]
                        * perf_mixture[injecting, component]
                    )
        return result


__all__ = ["WellComponentPhaseFlux"]
