"""MRST ``ad-core/models/WrapperModel.m``.

A model that wraps a `parentModel` and forwards to it: primary variables,
AD initialisation, equations, state and schedule validation, updates,
convergence, control changes, timestep hooks, driving forces and state
function groupings. Validating it validates the parent and then sets up the
wrapper's own state function groupings (with defaults only if the parent had
none). Subclasses (the sequential pressure and transport models) override
what differs.

`WrapperModel` in `ad_core/models/wrapper_model.py` is PyMRST's earlier
version on the legacy models.
"""

from __future__ import annotations

from copy import copy
from typing import Any

from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel, merge_options


class WrapperModel(PhysicalModel):
    """MRST `WrapperModel`."""

    def __init__(self, parentModel: Any, **options: Any):
        super().__init__(parentModel.G)
        self.parentModel = None
        merge_options(self, options)
        self.parentModel = parentModel

    def get_primary_variables(self, state: Any):
        return self.parentModel.get_primary_variables(state)

    def init_state_ad(self, state: Any, variables: Any, names: Any, origin: Any) -> Any:
        return self.parentModel.init_state_ad(state, variables, names, origin)

    def get_equations(self, state0: Any, state: Any, dt: float, forces: Any, **options: Any):
        return self.parentModel.get_equations(state0, state, dt, forces, **options)

    def get_valid_driving_forces(self):
        return self.parentModel.get_valid_driving_forces()

    def validate_state(self, state: Any) -> Any:
        return self.parentModel.validate_state(state)

    def validate_schedule(self, schedule: Any) -> Any:
        return self.parentModel.validate_schedule(schedule)

    def update_state(self, *args: Any):
        return self.parentModel.update_state(*args)

    def check_convergence(self, *args: Any):
        return self.parentModel.check_convergence(*args)

    def update_for_changed_controls(self, state: Any, forces: Any):
        model = copy(self)
        model.parentModel, state = self.parentModel.update_for_changed_controls(state, forces)
        return model, state

    def validate_model(self, *args: Any) -> "WrapperModel":
        set_defaults = len(self.parentModel.get_state_function_groupings()[0]) == 0
        model = copy(self)
        model.parentModel = value_copy_model(self.parentModel).validate_model(*args)
        return model.setup_state_function_groupings(set_defaults)

    def get_variable_field(self, name: str, *args: Any, **kwargs: Any):
        return self.parentModel.get_variable_field(name, *args, **kwargs)

    def prepare_timestep(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        model = copy(self)
        model.parentModel, state = self.parentModel.prepare_timestep(state, state0, dt,
                                                                     driving_forces)
        return model, state

    def prepare_report_step(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        model = copy(self)
        model.parentModel, state = self.parentModel.prepare_report_step(state, state0, dt,
                                                                        driving_forces)
        return model, state

    def get_driving_forces(self, control: Any):
        return self.parentModel.get_driving_forces(control)

    def update_after_convergence(self, *args: Any):
        return self.parentModel.update_after_convergence(*args)

    def get_maximum_timestep(self, state: Any, state0: Any, dt: float, driving_forces: Any):
        return self.parentModel.get_maximum_timestep(state, state0, dt, driving_forces)

    def get_scaling_factors_cpr(self, problem: Any, names: Any, solver: Any):
        return self.parentModel.get_scaling_factors_cpr(problem, names, solver)

    def check_state_function_dependencies(self) -> None:
        self.parentModel.check_state_function_dependencies()

    def validate_driving_forces(self, forces: Any, *args: Any):
        return self.parentModel.validate_driving_forces(forces, *args)

    def get_state_function_groupings(self):
        return self.parentModel.get_state_function_groupings()

    def get_convergence_values(self, *args: Any):
        return self.parentModel.get_convergence_values(*args)

    def get_reservoir_model(self) -> Any:
        """The `ReservoirModel` under any number of wrappers."""
        from pymrst.modules.ad_core.ad_oo.reservoir_model import ReservoirModel
        rmodel = self.parentModel
        while not isinstance(rmodel, ReservoirModel):
            rmodel = rmodel.parentModel
        return rmodel

    getPrimaryVariables = get_primary_variables
    initStateAD = init_state_ad
    getEquations = get_equations
    getValidDrivingForces = get_valid_driving_forces
    validateState = validate_state
    validateSchedule = validate_schedule
    updateState = update_state
    checkConvergence = check_convergence
    updateForChangedControls = update_for_changed_controls
    validateModel = validate_model
    getVariableField = get_variable_field
    prepareTimestep = prepare_timestep
    prepareReportstep = prepare_report_step
    getDrivingForces = get_driving_forces
    updateAfterConvergence = update_after_convergence
    getMaximumTimestep = get_maximum_timestep
    getScalingFactorsCPR = get_scaling_factors_cpr
    checkStateFunctionDependencies = check_state_function_dependencies
    validateDrivingForces = validate_driving_forces
    getStateFunctionGroupings = get_state_function_groupings
    getConvergenceValues = get_convergence_values
    getReservoirModel = get_reservoir_model


def value_copy_model(model: Any) -> Any:
    """A copy of a model that MATLAB's value semantics would give: the model
    and its facility model are new objects, so setting up one's state
    function groupings in place does not set up the other's."""
    out = copy(model)
    if getattr(out, "FacilityModel", None) is not None:
        out.FacilityModel = copy(out.FacilityModel)
    return out


__all__ = ["WrapperModel", "value_copy_model"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock

    G = compute_geometry(cart_grid([3], [3.0]))
    PARENT = GenericBlackOilModel(G, make_rock(G, 1e-12, 0.3),
                                  init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3],
                                                        rho=[1000, 800]), gas=False)

    def test_it_forwards_to_the_parent():
        wrapper = WrapperModel(PARENT)
        assert wrapper.G is G and wrapper.parentModel is PARENT
        assert wrapper.get_variable_field("sw") == PARENT.get_variable_field("sw")
        assert wrapper.get_valid_driving_forces().keys() == PARENT.get_valid_driving_forces().keys()
        print("ok test_it_forwards_to_the_parent")

    def test_validation_validates_the_parent_on_a_copy():
        wrapper = WrapperModel(PARENT)
        validated = wrapper.validate_model()
        assert validated is not wrapper and wrapper.parentModel is PARENT
        assert validated.parentModel.FlowDiscretization is not None
        assert validated.get_reservoir_model() is validated.parentModel
        print("ok test_validation_validates_the_parent_on_a_copy")

    def test_unknown_options_are_refused():
        try:
            WrapperModel(PARENT, noSuchOption=1)
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError")
        print("ok test_unknown_options_are_refused")

    test_it_forwards_to_the_parent()
    test_validation_validates_the_parent_on_a_copy()
    test_unknown_options_are_refused()
    print("\nAll tests passed!")
