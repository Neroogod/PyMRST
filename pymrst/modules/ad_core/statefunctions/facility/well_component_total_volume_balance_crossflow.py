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

    #: MRST switches the mixing rule on this. Only the surface-volume rule is
    #: expressible in PyMRST's basis; see `compute_cross_flux`.
    mass_type = "surfacevolume"

    def compute_cross_flux(self, facility: Any, mapping: Any, phase_flux: Any,
                           mass_type: str = None) -> List[Any]:
        """Mix the perforation fluxes that flow back into the wellbore.

        Crossflow is one perforation producing into the wellbore while another
        injects out of it. What comes back out carries the *mixture* in the
        wellbore, not the composition of the cell it re-enters, and computing
        it any other way lets a component be created at one perforation and
        destroyed at another - conservative globally, wrong per component.

        MRST selects the mixing rule by mass type. PyMRST's component
        equations are written in **surface volume**, where MRST's conversions
        to and from mass cancel exactly, so the surface-volume rule is the one
        this basis can express. The mass-fraction rules are refused rather
        than approximated: they mix by a quantity this port does not carry,
        and silently substituting the volume rule would give a different
        composition under the same name.
        """
        requested = (mass_type or self.mass_type).lower()
        if requested != "surfacevolume":
            raise NotImplementedError(
                f"crossflow mass type {requested!r} mixes by mass fraction; "
                f"PyMRST's component equations are in surface volume, where "
                f"MRST's mass conversions cancel. Only 'surfacevolume' is "
                f"expressible without changing the equations' basis.")
        return apply_total_crossflow_mixing(
            facility, mapping, sum_component_phase_flux(phase_flux))

    computeCrossFlux = compute_cross_flux

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[Any]:
        mapping = self.get_evaluated(state, "FacilityWellMapping")
        phase_flux = self.get_evaluated(state, "ComponentPhaseFlux")
        self.get_evaluated(state, "InjectionSurfaceDensity")
        return self.compute_cross_flux(facility, mapping, phase_flux)


__all__ = ["WellComponentTotalVolumeBalanceCrossflow"]
