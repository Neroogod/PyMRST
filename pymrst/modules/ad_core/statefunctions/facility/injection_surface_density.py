"""Per-well surface densities used for injection streams."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


def _well_surface_density(well: Any) -> Any:
    """Read either spelling used by MRST/PyMRST well structures."""
    density = getattr(well, "rhoS", None)
    if density is None:
        density = getattr(well, "rho_s", None)
    return density


class InjectionSurfaceDensity(StateFunction):
    """Surface phase density for each active well.

    Reservoir surface density is sampled at the first completion. Injector
    wells may override it with ``well.rhoS``/``well.rho_s``; producer
    overrides are intentionally ignored, as in MRST.
    """

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="rho_S^W")
        self.depends_on("FacilityWellMapping")

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[Any]:
        mapping = self.get_evaluated(state, "FacilityWellMapping")
        try:
            surface = facility.reservoir_prop(state, "SurfaceDensity")
        except KeyError:
            # Older PyMRST reservoir graphs expose the same black-oil
            # constants directly on the fluid instead of registering the
            # SurfaceDensity state function.
            reservoir = facility.reservoir_model
            rho_s_value = getattr(reservoir.fluid, "rho_s", None)
            if rho_s_value is None:
                rho_s_value = getattr(reservoir.fluid, "rhoS")
            rho_s = np.asarray(rho_s_value, dtype=float)
            nc = reservoir.G.cells.num
            surface = [np.full(nc, rho_s[phase])
                       for phase in range(facility.n_phases)]
        first_cells = np.asarray(
            [np.asarray(w.cells, dtype=int).ravel()[0] for w in mapping.wells],
            dtype=int,
        )
        result = [density[first_cells].copy() for density in surface]

        for well_index, well in enumerate(mapping.wells):
            override = _well_surface_density(well)
            if not mapping.is_injector[well_index] or override is None:
                continue
            override = np.asarray(override, dtype=float).ravel()
            if override.size < len(result):
                raise ValueError(
                    f"well {getattr(well, 'name', well_index)!r} supplies "
                    f"{override.size} surface densities for {len(result)} phases"
                )
            for phase in range(len(result)):
                result[phase][well_index] = override[phase]
        return result


__all__ = ["InjectionSurfaceDensity"]
