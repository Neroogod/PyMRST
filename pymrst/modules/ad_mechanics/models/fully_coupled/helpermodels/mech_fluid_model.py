"""Composite base for fully coupled mechanics/fluid models."""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_mechanics.models.poromechanics import PoroMechanicalPoreVolume


def _call(owner: Any, snake: str, camel: str, *args: Any, **kwargs: Any):
    method = getattr(owner, snake, None) or getattr(owner, camel, None)
    if method is None:
        raise AttributeError(f"{type(owner).__name__} has neither {snake} nor {camel}")
    return method(*args, **kwargs)


def _names(owner: Any) -> list[str]:
    for spelling in ("get_all_vars_names", "getAllVarsNames"):
        method = getattr(owner, spelling, None)
        if method is not None:
            return list(method())
    return []


class MechFluidModel:
    """MRST-style owner coordinating a fluid model and mechanics helper.

    Concrete water, oil-water and black-oil models own equation assembly. This
    base owns cross-model variable routing, updates, control propagation and
    state/model validation.
    """

    def __init__(
        self,
        grid: Any,
        rock: Any,
        fluid: Any,
        mech_model: Any,
        fluid_model: Any | None = None,
        *,
        facility_model: Any | None = None,
    ):
        self.G = grid
        self.rock = rock
        self.fluid = fluid
        self.mechModel = mech_model
        self.fluidModel = fluid_model if fluid_model is not None else self.setup_fluid_model()
        self.FacilityModel = facility_model
        self.fluidfds = _names(self.fluidModel)
        self.mechfds = _names(self.mechModel)

    def setup_fluid_model(self):
        raise RuntimeError("MechFluidModel is abstract: concrete class must set up fluid model")

    setupFluidModel = setup_fluid_model

    def get_equations(self, *args: Any, **kwargs: Any):
        raise RuntimeError("MechFluidModel is abstract: concrete class must assemble equations")

    getEquations = get_equations

    def compute_coupling_terms(self, *args: Any, **kwargs: Any):
        raise RuntimeError("MechFluidModel is abstract: concrete class must compute coupling terms")

    computeCouplingTerms = compute_coupling_terms

    def get_variable_field(self, name: str, *args: Any, **kwargs: Any):
        if name in self.fluidfds:
            return _call(self.fluidModel, "get_variable_field", "getVariableField", name, *args, **kwargs)
        if name in self.mechfds:
            return _call(self.mechModel, "get_variable_field", "getVariableField", name, *args, **kwargs)
        raise KeyError(f"Unknown coupled variable {name!r}")

    getVariableField = get_variable_field

    def update_state(self, state: Any, problem: Any, increment: Any, driving_forces: Any = None):
        fluid_result = _call(
            self.fluidModel, "update_state", "updateState",
            state, problem, increment, driving_forces,
        )
        fluid_state, fluid_report = fluid_result if isinstance(fluid_result, tuple) else (fluid_result, {})
        mech_result = _call(
            self.mechModel, "update_state", "updateState",
            fluid_state, problem, increment, None,
        )
        final_state, mech_report = mech_result if isinstance(mech_result, tuple) else (mech_result, {})
        return final_state, {"fluid": fluid_report, "mechanics": mech_report}

    updateState = update_state

    def validate_driving_forces(self, control: Any, index: int | None = None):
        return _call(
            self.fluidModel, "validate_driving_forces", "validateDrivingForces",
            control, index,
        )

    validateDrivingForces = validate_driving_forces

    def validate_model(self, *args: Any, **kwargs: Any):
        if self.FacilityModel is not None and hasattr(self.fluidModel, "FacilityModel"):
            self.fluidModel.FacilityModel = self.FacilityModel
        method = getattr(self.fluidModel, "validate_model", None) or getattr(self.fluidModel, "validateModel", None)
        if method is not None:
            result = method(*args, **kwargs)
            if result is not None:
                self.fluidModel = result
        return self

    validateModel = validate_model

    def update_for_changed_controls(self, state: Any, forces: Any):
        result = _call(
            self.fluidModel, "update_for_changed_controls", "updateForChangedControls",
            state, forces,
        )
        if isinstance(result, tuple) and len(result) == 2:
            self.fluidModel, state = result
        else:
            state = result
        self.fluidfds = _names(self.fluidModel)
        return self, state

    updateForChangedControls = update_for_changed_controls

    def validate_state(self, state: Any):
        state = _call(self.fluidModel, "validate_state", "validateState", state)
        return _call(self.mechModel, "validate_state", "validateState", state)

    validateState = validate_state


__all__ = ["MechFluidModel", "PoroMechanicalPoreVolume"]
