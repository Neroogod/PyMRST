"""MRST `ad-core/models/ReservoirModel.m` reservoir base model."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from pymrst.modules.ad_core.statefunctions.components.components import Component, blackoil_components
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.statefunctions.flow_discretization import (
    FlowDiscretization,
)
from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
    FlowPropertyFunctions,
)
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    PVTPropertyFunctions,
    PhaseFluid,
)
from pymrst.modules.ad_core.statefunctions.state_function import State


class ReservoirModel(PhysicalModel):
    """A flow model on a grid: rock, fluid, operators, and state functions."""

    def __init__(self, g, rock, fluid: PhaseFluid, operators, pv_ref,
                 gdz: np.ndarray, gravity: float = 0.0,
                 components: Optional[List[Component]] = None,
                 mass_basis: bool = False):
        super().__init__()
        self.G = g
        self.rock = rock
        self.fluid = fluid
        self.operators = operators
        self.pv_ref = pv_ref
        self.gdz = gdz
        self.mass_basis = bool(mass_basis)
        self.gravity_magnitude = float(gravity)
        self.components = (components if components is not None
                           else blackoil_components(fluid.n_phases,
                                                    fluid.disgas,
                                                    fluid.vapoil))
        self.PVTPropertyFunctions = PVTPropertyFunctions(self)
        self.FlowPropertyFunctions = FlowPropertyFunctions(self)
        self.FlowDiscretization = FlowDiscretization(self)
        self.groupings = ["PVTPropertyFunctions", "FlowPropertyFunctions",
                          "FlowDiscretization"]

    @property
    def n_phases(self) -> int:
        return self.fluid.n_phases

    @property
    def n_components(self) -> int:
        return len(self.components)

    def component_names(self) -> List[str]:
        return [c.name for c in self.components]

    def get_component_names(self) -> List[str]:
        """Return component names using MRST's ``getComponentNames`` surface."""
        return self.component_names()

    getComponentNames = get_component_names

    def get_number_of_components(self) -> int:
        return self.n_components

    getNumberOfComponents = get_number_of_components

    @staticmethod
    def _mrst_phase_name(name: str) -> str:
        lookup = {"water": "W", "oil": "O", "gas": "G"}
        key = str(name).lower()
        return lookup.get(key, str(name).upper())

    def get_phase_names(self) -> List[str]:
        """Return active phase names in MRST's short-name convention."""
        return [self._mrst_phase_name(name) for name in self.fluid.phases]

    getPhaseNames = get_phase_names

    def get_number_of_phases(self) -> int:
        return self.n_phases

    getNumberOfPhases = get_number_of_phases

    def get_active_phases(self) -> np.ndarray:
        """Return a boolean mask for active phases, matching MRST helper API."""
        return np.ones(self.n_phases, dtype=bool)

    getActivePhases = get_active_phases

    def get_phase_index(self, name: str) -> int:
        """Return the zero-based phase index for an MRST phase name."""
        key = str(name).lower()
        short = [str(item).lower() for item in self.get_phase_names()]
        long = [str(item).lower() for item in self.fluid.phases]
        if key in short:
            return short.index(key)
        if key in long:
            return long.index(key)
        raise ValueError(f"unknown phase name {name!r}")

    getPhaseIndex = get_phase_index

    @property
    def component_rho_s(self) -> np.ndarray:
        rho_s = np.ones(self.n_components)
        n = min(self.n_components, self.fluid.n_phases)
        rho_s[:n] = self.fluid.rho_s[:n]
        return rho_s

    def _basis_scale(self) -> np.ndarray:
        if not self.mass_basis:
            return np.ones(self.n_components)
        return self.component_rho_s

    def component_state_keys(self) -> List[str]:
        keys = ["rs"] if self.fluid.disgas else []
        keys += ["rv"] if self.fluid.vapoil else []
        keys += [c.state_key for c in self.components
                 if hasattr(c, "state_key")]
        return keys

    def conservation_equations(self, state0: State, state: State, dt: float):
        return self.FlowDiscretization.component_conservation_equations(
            self, state, state0, dt
        )


__all__ = ["ReservoirModel"]
