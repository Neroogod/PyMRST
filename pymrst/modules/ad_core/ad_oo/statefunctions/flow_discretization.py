"""MRST ``ad-core/statefunctions/FlowDiscretization.m``.

The grouping of face quantities - transmissibility, pressure and gravity
potential differences, upwind flags, face mobilities and the phase and
component fluxes - plus the flow-state builder that decides which state the
fluxes are evaluated in, and MRST's generic component conservation
equations (accumulation over `dt` and the total component flux).
"""

from __future__ import annotations

from typing import Any, Dict, List

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.component_phase_flux import (
    ComponentPhaseFlux,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.component_total_flux import (
    ComponentTotalFlux,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.face_component_mobility import (
    FaceComponentMobility,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.face_mobility import FaceMobility
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.gravity_potential_difference import (
    GravityPotentialDifference,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.implicit_flow_state_builder import (
    ImplicitFlowStateBuilder,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.permeability_potential_gradient import (
    PermeabilityPotentialGradient,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.phase_flux import PhaseFlux
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.phase_potential_difference import (
    PhasePotentialDifference,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.phase_upwind_flag import (
    PhaseUpwindFlag,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.pressure_gradient import (
    PressureGradient,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.transmissibility import (
    Transmissibility,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.two_point_flux_approximation import (
    TwoPointFluxApproximation,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    StateFunctionGrouping,
)


class FlowDiscretization(StateFunctionGrouping):
    """MRST's `FluxDisc` container."""

    def __init__(self, model: Any):
        tpfa = TwoPointFluxApproximation(model)
        super().__init__("FluxDisc")
        self.function_dependencies: Dict[str, List[str]] = {}
        self.set_state_function("Transmissibility", Transmissibility(model))
        self.set_state_function("PermeabilityPotentialGradient",
                                PermeabilityPotentialGradient(model, tpfa))
        self.set_state_function("PressureGradient", PressureGradient(model))
        self.set_state_function("GravityPotentialDifference",
                                GravityPotentialDifference(model))
        solution = _field(model.inputdata, "SOLUTION")
        if model.inputdata is not None and _field(solution, "THPRES") is not None:
            from pymrst.modules.ad_core.ad_oo.statefunctions.flux.phase_potential_difference_thresholded import (
                PhasePotentialDifferenceThresholded,
            )
            ppd = PhasePotentialDifferenceThresholded(model)
        else:
            ppd = PhasePotentialDifference(model)
        self.set_state_function("PhasePotentialDifference", ppd)
        self.set_state_function("PhaseUpwindFlag", PhaseUpwindFlag(model))
        self.set_state_function("FaceComponentMobility", FaceComponentMobility(model))
        self.set_state_function("FaceMobility", FaceMobility(model))
        self.set_state_function("ComponentPhaseFlux", ComponentPhaseFlux(model))
        self.set_state_function("ComponentTotalFlux", ComponentTotalFlux(model))
        self.set_state_function("PhaseFlux", PhaseFlux(model))
        self.FlowStateBuilder = ImplicitFlowStateBuilder()
        self.function_depends_on("componentConservationEquations",
                                 ["ComponentTotalMass", "ComponentTotalFlux"])

    def function_depends_on(self, function_name: str,
                            names: List[str]) -> "FlowDiscretization":
        """MRST `functionDependsOn`: record what a member function reads."""
        self.function_dependencies[function_name] = list(names)
        return self

    def set_flow_state_builder(self, fb: Any) -> "FlowDiscretization":
        self.FlowStateBuilder = fb
        return self

    def get_flow_state_builder(self) -> Any:
        return self.FlowStateBuilder

    def component_conservation_equations(self, model: Any, state: Any,
                                         state0: Any, dt: float):
        """`(acc, v, names, types)`: `(M_i - M_i^0)/dt` and the flux `V_i`."""
        ncomp = model.get_number_of_components()
        names = model.get_component_names()
        types = ["cell"]*ncomp
        mass = model.get_props(state, "ComponentTotalMass")
        mass0 = model.get_props(state0, "ComponentTotalMass")
        flow_state = self.build_flow_state(model, state, state0, dt)
        v = model.get_props(flow_state, "ComponentTotalFlux")
        if not isinstance(mass0, list):
            mass0 = [mass0]
        acc = [(mass[c] - mass0[c])/dt for c in range(ncomp)]
        return acc, v, names, types

    def build_flow_state(self, model: Any, state: Any, state0: Any,
                         dt: float) -> Any:
        return self.FlowStateBuilder.build(self, model, state, state0, dt)

    def get_maximum_timestep(self, model: Any, state: Any, state0: Any,
                             dt: float, forces: Any) -> float:
        return self.FlowStateBuilder.get_maximum_timestep(self, model, state,
                                                          state0, dt, forces)

    def prepare_timestep(self, model: Any, state: Any, state0: Any, dt: float,
                         driving_forces: Any):
        self.FlowStateBuilder, state = self.FlowStateBuilder.prepare_timestep(
            self, model, state, state0, dt, driving_forces)
        return self, state

    functionDependsOn = function_depends_on
    setFlowStateBuilder = set_flow_state_builder
    getFlowStateBuilder = get_flow_state_builder
    componentConservationEquations = component_conservation_equations
    buildFlowState = build_flow_state
    getMaximumTimestep = get_maximum_timestep
    prepareTimestep = prepare_timestep


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


__all__ = ["FlowDiscretization"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model

    def test_the_grouping_carries_mrsts_functions():
        model = toy_model("WO", nx=2)
        fd = model.FlowDiscretization
        assert fd.struct_name == "FluxDisc"
        assert {"Transmissibility", "PhaseFlux", "PhaseUpwindFlag",
                "FaceMobility", "ComponentTotalFlux"} <= set(fd.get_names())
        assert isinstance(fd.get_flow_state_builder(), ImplicitFlowStateBuilder)
        print("ok test_the_grouping_carries_mrsts_functions")

    def test_the_implicit_builder_sets_no_timestep_limit():
        model = toy_model("WO", nx=2)
        dt = model.FlowDiscretization.get_maximum_timestep(model, None, None,
                                                           1.0, None)
        assert dt == float("inf")
        print("ok test_the_implicit_builder_sets_no_timestep_limit")

    test_the_grouping_carries_mrsts_functions()
    test_the_implicit_builder_sets_no_timestep_limit()
    print("\nAll tests passed!")
