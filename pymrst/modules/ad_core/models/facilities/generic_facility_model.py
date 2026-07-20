"""MRST ``ad-core/models/facilities/GenericFacilityModel.m`` surface."""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.utils.units import ATM


class GenericFacilityModel(FacilityModel):
    """Generic facility model with MRST's generic-facility defaults."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = 288.15
        self.pressure = ATM
        self.SeparatorGroup = None
        self.separator_group = None
        self.outputFluxes = True
        self.output_fluxes = True
        self.doPostUpdate = None
        self.do_post_update = None
        self.setWellTargets = False

    def getNumberOfComponents(self) -> int:
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if reservoir_model is None:
            return self._number_of_components()
        for method in ("getNumberOfComponents", "get_number_of_components"):
            if hasattr(reservoir_model, method):
                return int(getattr(reservoir_model, method)())
        return self._number_of_components()

    get_number_of_components = getNumberOfComponents

    def getNumberOfPhases(self) -> int:
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if reservoir_model is not None:
            for method in ("getNumberOfPhases", "get_number_of_phases"):
                if hasattr(reservoir_model, method):
                    return int(getattr(reservoir_model, method)())
        return len(self._phase_names())

    get_number_of_phases = getNumberOfPhases

    def getSurfaceConditions(self) -> tuple[float, float]:
        """Return separator/surface pressure and temperature."""
        return float(self.pressure), float(self.T)

    get_surface_conditions = getSurfaceConditions

    def getComponentSources(self, state: Any):
        """Return component source data from the facility discretization."""
        fp = self.FacilityFlowDiscretization
        if fp is None or not hasattr(fp, "get"):
            return {"value": [], "cells": []}
        mapping = fp.get(self, state, "FacilityWellMapping")
        wells = getattr(mapping, "W", None)
        cells = getattr(mapping, "cells", [])
        if wells is None or len(wells) == 0:
            values = []
        else:
            values = fp.get(self, state, "ComponentTotalFlux")
        return {"value": values, "cells": cells}

    get_component_sources = getComponentSources


__all__ = ["GenericFacilityModel"]
