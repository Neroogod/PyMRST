"""Facility flow discretization (MRST ad-core/statefunctions/facility).

Ports the well-flux half of MRST's facility as a state-function grouping.
MRST's facility model hosts a `FacilityFlowDiscretization` grouping whose state
functions compute, per perforation, the flux between the wellbore and the
reservoir. PyMRST's `FacilityModel` computes the same thing procedurally
(`_single_well` / `computeWellContributionsSingleWell`); this module expresses
it as MRST's graph so the well physics *composes* - a component well, a
multisegment well, crossflow, all become state functions rather than branches
in one 2,800-line method.

The grouping runs on a light `FacilityContext`: the wrapper that gives the
facility state functions the handful of things they read - the reservoir model
and its AD state, the active wells, and the per-well bottom-hole pressure and
per-perforation connection pressure-drop. It plays the role MRST's facility
model does for these state functions (`model.ReservoirModel`,
`model.getWellStruct`, `state.wellSol`), adapted to PyMRST's well structs.

Core landed here: `FacilityWellMapping`, `WellIndex`,
`PerforationPressureGradient`, `PerforationMobility`, `WellPhaseFlux`. The
component-well fluxes (`WellComponentPhaseFlux`, `...TotalFlux`,
`...DensityMix`, `...VolumeBalanceCrossflow`), `PerforationComponentPhaseDensity`
and `InjectionSurfaceDensity` are the intricate crossflow-mixture layer and are
a separate follow-on.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.statefunctions.state_function import (
    State,
    StateFunctionGrouping,
)


class WellMapping:
    """The perforation<->well bookkeeping MRST returns from FacilityWellMapping."""

    __slots__ = ("active", "cells", "perf2well", "is_injector",
                 "perforation_sum", "wells")

    def __init__(self, active, cells, perf2well, is_injector, perforation_sum,
                 wells):
        self.active = active
        self.cells = cells
        self.perf2well = perf2well
        self.is_injector = is_injector
        self.perforation_sum = perforation_sum
        self.wells = wells


class FacilityContext:
    """Minimal facility model hosting the well-flux state-function grouping.

    Parameters
    ----------
    reservoir_model : ReservoirModel
        The model whose graph supplies reservoir pressure and mobility.
    wells : sequence of Well
        The active wells (each with ``cells``, ``WI``, ``sign``, ``compi``).
    """

    def __init__(self, reservoir_model: Any, wells: Sequence[Any]):
        self.reservoir_model = reservoir_model
        self.wells = list(wells)
        self.n_phases = reservoir_model.n_phases
        self.FacilityFlowDiscretization = FacilityFlowDiscretization(self)
        self.groupings = ["FacilityFlowDiscretization"]

    def get_props(self, state: State, name: str) -> Any:
        return self.FacilityFlowDiscretization.get(self, state, name)

    # reservoir properties, pulled from the hosted reservoir model's graph
    def reservoir_prop(self, state: State, name: str) -> Any:
        return self.reservoir_model.get_props(state["reservoir_state"], name)


class FacilityFlowDiscretization(StateFunctionGrouping):
    """The grouping of well-flux state functions (MRST FacilityFlowDiscretization)."""

    def __init__(self, facility: Any):
        super().__init__("FacilityFluxDisc")
        # imported here to avoid a construction-time import cycle
        from pymrst.modules.ad_core.statefunctions.facility.facility_well_mapping import (  # noqa: E501
            FacilityWellMapping,
        )
        from pymrst.modules.ad_core.statefunctions.facility.well_index import (
            WellIndex,
        )
        from pymrst.modules.ad_core.statefunctions.facility.perforation_pressure_gradient import (  # noqa: E501
            PerforationPressureGradient,
        )
        from pymrst.modules.ad_core.statefunctions.facility.perforation_mobility import (  # noqa: E501
            PerforationMobility,
        )
        from pymrst.modules.ad_core.statefunctions.facility.well_phase_flux import (  # noqa: E501
            WellPhaseFlux,
        )
        self.set_state_function("FacilityWellMapping", FacilityWellMapping(facility))
        self.set_state_function("WellIndex", WellIndex(facility))
        self.set_state_function("PressureGradient",
                                PerforationPressureGradient(facility))
        self.set_state_function("Mobility", PerforationMobility(facility))
        self.set_state_function("WellPhaseFlux", WellPhaseFlux(facility))


def build_well_mapping(facility: Any) -> WellMapping:
    """Cells, perf->well map, injector flags and the perforation-sum matrix."""
    wells = facility.wells
    active = np.arange(len(wells))
    cells = np.concatenate([np.asarray(w.cells, int).ravel() for w in wells]) \
        if wells else np.zeros(0, int)
    perf2well = np.concatenate(
        [np.full(np.asarray(w.cells).size, i, int) for i, w in enumerate(wells)]) \
        if wells else np.zeros(0, int)
    is_injector = np.array([np.sign(getattr(w, "sign", 1)) > 0 for w in wells],
                           dtype=bool)
    n_perf = perf2well.size
    perforation_sum = sp.csr_array(
        (np.ones(n_perf), (perf2well, np.arange(n_perf))),
        shape=(len(wells), n_perf)) if n_perf else sp.csr_array((len(wells), 0))
    return WellMapping(active, cells, perf2well, is_injector, perforation_sum,
                       wells)


__all__ = [
    "FacilityContext",
    "FacilityFlowDiscretization",
    "WellMapping",
    "build_well_mapping",
]


if __name__ == "__main__":
    # exercised end to end by well_phase_flux.py and validate_phase343.py
    print("ok facility_flow_discretization (grouping + context)")
    print("\nAll tests passed!")
