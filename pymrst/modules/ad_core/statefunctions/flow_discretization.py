"""MRST `FlowDiscretization` flux state functions."""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
    _sum_over_phases,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    State,
    StateFunction,
    StateFunctionGrouping,
)
from pymrst.utils.adi import value


class Transmissibility(StateFunction):
    """Face transmissibility (TPFA)."""

    def __init__(self, model=None):
        super().__init__(model, label="T")

    def evaluate_on_domain(self, model, state):
        return model.operators.T


class PressureGradient(StateFunction):
    """Phase pressure gradient on interior faces."""

    def __init__(self, model=None):
        super().__init__(model, label="\\nabla p_\\alpha")
        self.Grad = model.operators.grad if model is not None else None
        self.depends_on("pressure", STATE)
        self.has_pc = model is not None and model.fluid.has_pc
        if self.has_pc:
            self.depends_on("PhasePressures", "PVTPropertyFunctions")

    def evaluate_on_domain(self, model, state):
        grad = self.Grad if self.Grad is not None else model.operators.grad
        if not self.has_pc:
            return [grad(state["pressure"])] * model.n_phases
        p_phase = self.get_evaluated_external(
            model, state, "PVTPropertyFunctions", "PhasePressures")
        return [grad(p) for p in p_phase]


class GravityPotentialDifference(StateFunction):
    """rho g dz on interior faces."""

    def __init__(self, model=None):
        super().__init__(model, label="\\rho g \\Delta z")
        self.weight = None
        self.depends_on("Density", "PVTPropertyFunctions")

    def evaluate_on_domain(self, model, state):
        rho = self.get_evaluated_external(model, state,
                                          "PVTPropertyFunctions", "Density")
        out = [model.operators.avg(r) * model.gdz for r in rho]
        if self.weight is not None:
            from pymrst.utils.adi import ad_matmul

            out = [ad_matmul(self.weight, value) for value in out]
        return out


class PhasePotentialDifference(StateFunction):
    """Phase pressure potential difference."""

    def __init__(self, model=None):
        super().__init__(model, label="\\Phi_\\alpha")
        self.depends_on(["PressureGradient", "GravityPotentialDifference"])

    def evaluate_on_domain(self, model, state):
        dp = self.get_evaluated(state, "PressureGradient")
        gdz = self.get_evaluated(state, "GravityPotentialDifference")
        return [dp[i] - gdz[i] for i in range(len(gdz))]


class PhaseUpwindFlag(StateFunction):
    """Upwind side of each face for each phase."""

    def __init__(self, model=None):
        super().__init__(model, label="\\theta_\\alpha")
        self.depends_on("PhasePotentialDifference")

    def evaluate_on_domain(self, model, state):
        pot = self.get_evaluated(state, "PhasePotentialDifference")
        return [value(pt) <= 0 for pt in pot]


class FaceMobility(StateFunction):
    """Upwinded phase mobility on each face."""

    def __init__(self, model=None, upstream=None):
        super().__init__(model, label="\\lambda_\\alpha^f")
        self.upstream = upstream
        self.depends_on("PhaseUpwindFlag")
        self.depends_on("Mobility", "FlowPropertyFunctions")

    def evaluate_on_domain(self, model, state):
        flag = self.get_evaluated(state, "PhaseUpwindFlag")
        mob = self.get_evaluated_external(model, state,
                                          "FlowPropertyFunctions", "Mobility")
        if self.upstream is None:
            upw = model.operators.upw
            return [upw(flag[i], mob[i]) for i in range(len(mob))]
        return [
            self.upstream.face_upstream(model, state, flag[i], mob[i])
            for i in range(len(mob))
        ]


class PhaseFlux(StateFunction):
    """Darcy phase flux."""

    def __init__(self, model=None):
        super().__init__(model, label="v_\\alpha")
        self.depends_on(["FaceMobility", "PermeabilityPotentialGradient"])

    def evaluate_on_domain(self, model, state):
        mobf = self.get_evaluated(state, "FaceMobility")
        kgrad = self.get_evaluated(state, "PermeabilityPotentialGradient")
        return [-mobf[i] * kgrad[i] for i in range(len(mobf))]


class ComponentPhaseFlux(StateFunction):
    """Flux of each component carried by each phase."""

    def __init__(self, model=None):
        super().__init__(model, label="v_{i\\alpha}")
        self.depends_on(["PermeabilityPotentialGradient", "FaceComponentMobility"])

    def evaluate_on_domain(self, model, state):
        kgrad = self.get_evaluated(state, "PermeabilityPotentialGradient")
        component_mobility = self.get_evaluated(state, "FaceComponentMobility")
        scale = model._basis_scale()
        return [[None if mobility is None else -scale[c] * mobility * kgrad[ph]
                 for ph, mobility in enumerate(row)]
                for c, row in enumerate(component_mobility)]


class ComponentTotalFlux(StateFunction):
    """Total flux of each component."""

    def __init__(self, model=None):
        super().__init__(model, label="V_i")
        self.depends_on("ComponentPhaseFlux")

    def evaluate_on_domain(self, model, state):
        per_phase = self.get_evaluated(state, "ComponentPhaseFlux")
        return [_sum_over_phases(f) for f in per_phase]


class FlowDiscretization(StateFunctionGrouping):
    """How mass moves between cells (MRST FlowDiscretization)."""

    def __init__(self, model):
        # Local import avoids the flux package's compatibility exports
        # circularly importing this grouping during module initialization.
        from pymrst.modules.ad_core.statefunctions.flux.flow_state.flow_state_builder import (  # noqa: E501
            ImplicitFlowStateBuilder,
        )
        from pymrst.modules.ad_core.statefunctions.flux.face_component_mobility import (
            FaceComponentMobility,
        )
        from pymrst.modules.ad_core.statefunctions.flux.permeability_potential_gradient import (
            PermeabilityPotentialGradient,
        )

        super().__init__("FluxDisc")
        self.FlowStateBuilder = ImplicitFlowStateBuilder()
        self.set_state_function("Transmissibility", Transmissibility(model))
        self.set_state_function("PressureGradient", PressureGradient(model))
        self.set_state_function("GravityPotentialDifference",
                                GravityPotentialDifference(model))
        self.set_state_function("PhasePotentialDifference",
                                PhasePotentialDifference(model))
        self.set_state_function("PermeabilityPotentialGradient",
                                PermeabilityPotentialGradient(model))
        self.set_state_function("PhaseUpwindFlag", PhaseUpwindFlag(model))
        self.set_state_function("FaceMobility", FaceMobility(model))
        self.set_state_function("FaceComponentMobility",
                                FaceComponentMobility(model))
        self.set_state_function("PhaseFlux", PhaseFlux(model))
        self.set_state_function("ComponentPhaseFlux", ComponentPhaseFlux(model))
        self.set_state_function("ComponentTotalFlux", ComponentTotalFlux(model))

    def build_flow_state(self, model, state, state0, dt):
        """The state the flux terms are evaluated on (MRST `buildFlowState`).

        One line, and the seam the whole explicit/adaptive-implicit machinery
        hangs from: an implicit builder returns `state` unchanged, an explicit
        one returns a state whose mobilities come from `state0`, and the flux
        code above never has to know which. Calling the properties directly
        instead - the obvious shortcut - hard-wires the implicit choice.
        """
        return self.FlowStateBuilder.build(model, state, state0, dt)

    buildFlowState = build_flow_state

    def set_flow_state_builder(self, builder):
        """Attach the FIM/AIM/IMPES flow-state policy."""
        self.FlowStateBuilder = builder
        return self

    setFlowStateBuilder = set_flow_state_builder

    def get_flow_state_builder(self):
        return self.FlowStateBuilder

    getFlowStateBuilder = get_flow_state_builder

    def get_maximum_timestep(self, model, state, state0, dt, forces=None):
        return self.FlowStateBuilder.get_maximum_timestep(
            model, state, state0, dt, forces
        )

    getMaximumTimestep = get_maximum_timestep

    def prepare_timestep(self, model, state, state0, dt, forces=None):
        builder, state = self.FlowStateBuilder.prepare_timestep(
            model, state, state0, dt, forces
        )
        self.FlowStateBuilder = builder
        return self, state

    prepareTimestep = prepare_timestep

    def component_conservation_equations(self, model, state: State,
                                         state0: State, dt: float) -> List[Any]:
        """(m_i - m_i^0)/dt + div(V_i) for every component."""
        mass = model.get_props(state, "ComponentTotalMass")
        mass0 = model.get_props(state0, "ComponentTotalMass")
        flow_state = self.FlowStateBuilder.build(
            model, state, state0, dt, fd=self
        )
        flux = model.get_props(flow_state, "ComponentTotalFlux")
        div = model.operators.div
        return [(mass[i] - mass0[i]) / dt + div(flux[i])
                for i in range(len(mass))]


__all__ = [
    "ComponentPhaseFlux",
    "ComponentTotalFlux",
    "FaceMobility",
    "FlowDiscretization",
    "GravityPotentialDifference",
    "PhaseFlux",
    "PhasePotentialDifference",
    "PhaseUpwindFlag",
    "PressureGradient",
    "Transmissibility",
]
