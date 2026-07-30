"""Component phase densities sampled at active well perforations."""

from __future__ import annotations

from typing import Any, List, Optional

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PerforationComponentPhaseDensity(StateFunction):
    """Return ``rho*x`` for every component/phase at perforated cells.

    The reservoir components are the source of truth for component density.
    Indexing their cell arrays with the facility mapping preserves AD
    Jacobians while reducing the values to active perforations.
    """

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="rho*x|_W")
        self.depends_on("FacilityWellMapping")

    def evaluate_on_domain(
        self, facility: Any, state: Any
    ) -> List[List[Optional[Any]]]:
        mapping = self.get_evaluated(state, "FacilityWellMapping")
        reservoir = facility.reservoir_model
        reservoir_state = state["reservoir_state"]
        output: List[List[Optional[Any]]] = []
        for component in reservoir.components:
            cell_density = component.get_component_density(
                reservoir, reservoir_state
            )
            output.append(
                [None if density is None else density[mapping.cells]
                 for density in cell_density]
            )
        return output


__all__ = ["PerforationComponentPhaseDensity"]
