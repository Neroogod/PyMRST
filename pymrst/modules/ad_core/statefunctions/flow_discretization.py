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

    def __init__(self, model=None):
        super().__init__(model, label="\\lambda_\\alpha^f")
        self.depends_on("PhaseUpwindFlag")
        self.depends_on("Mobility", "FlowPropertyFunctions")

    def evaluate_on_domain(self, model, state):
        flag = self.get_evaluated(state, "PhaseUpwindFlag")
        mob = self.get_evaluated_external(model, state,
                                          "FlowPropertyFunctions", "Mobility")
        upw = model.operators.upw
        return [upw(flag[i], mob[i]) for i in range(len(mob))]


class PhaseFlux(StateFunction):
    """Darcy phase flux."""

    def __init__(self, model=None):
        super().__init__(model, label="v_\\alpha")
        self.depends_on(["Transmissibility", "FaceMobility",
                         "PhasePotentialDifference"])

    def evaluate_on_domain(self, model, state):
        t = self.get_evaluated(state, "Transmissibility")
        mobf = self.get_evaluated(state, "FaceMobility")
        pot = self.get_evaluated(state, "PhasePotentialDifference")
        return [-t * mobf[i] * pot[i] for i in range(len(mobf))]


class ComponentPhaseFlux(StateFunction):
    """Flux of each component carried by each phase."""

    def __init__(self, model=None):
        super().__init__(model, label="v_{i\\alpha}")
        self.depends_on(["PhaseFlux", "PhaseUpwindFlag"])
        self.depends_on("ShrinkageFactors", "PVTPropertyFunctions")
        if model is not None:
            for key in model.component_state_keys():
                self.depends_on(key, STATE)

    def evaluate_on_domain(self, model, state):
        v = self.get_evaluated(state, "PhaseFlux")
        flag = self.get_evaluated(state, "PhaseUpwindFlag")
        b = self.get_evaluated_external(model, state, "PVTPropertyFunctions",
                                        "ShrinkageFactors")
        rs = state["rs"] if model.fluid.disgas else None
        rv = state["rv"] if model.fluid.vapoil else None
        upw = model.operators.upw
        scale = model._basis_scale()

        out = []
        for c, comp in enumerate(model.components):
            rho = comp.component_density(model, state, b, rs, rv)
            out.append([
                None if rho[ph] is None else scale[c] * upw(flag[ph], rho[ph]) * v[ph]
                for ph in range(len(v))
            ])
        return out


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

        super().__init__("FluxDisc")
        self.FlowStateBuilder = ImplicitFlowStateBuilder()
        self.set_state_function("Transmissibility", Transmissibility(model))
        self.set_state_function("PressureGradient", PressureGradient(model))
        self.set_state_function("GravityPotentialDifference",
                                GravityPotentialDifference(model))
        self.set_state_function("PhasePotentialDifference",
                                PhasePotentialDifference(model))
        self.set_state_function("PhaseUpwindFlag", PhaseUpwindFlag(model))
        self.set_state_function("FaceMobility", FaceMobility(model))
        self.set_state_function("PhaseFlux", PhaseFlux(model))
        self.set_state_function("ComponentPhaseFlux", ComponentPhaseFlux(model))
        self.set_state_function("ComponentTotalFlux", ComponentTotalFlux(model))

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
