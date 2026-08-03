import pytest

from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.mech_fluid_model import (
    MechFluidModel,
)


class Child:
    def __init__(self, prefix, names):
        self.prefix = prefix
        self.names = names

    def get_all_vars_names(self):
        return self.names

    def get_variable_field(self, name):
        return f"{self.prefix}_{name}", None

    def update_state(self, state, problem, increment, forces):
        out = dict(state)
        out[self.prefix] = out.get(self.prefix, 0) + 1
        return out, {"updated": self.prefix}

    def validate_state(self, state):
        out = dict(state)
        out[f"valid_{self.prefix}"] = True
        return out


def build():
    return MechFluidModel(None, None, None, Child("mech", ["xd", "u"]), Child("fluid", ["pressure", "sW"]))


def test_composite_routes_variable_ownership():
    model = build()
    assert model.get_variable_field("pressure") == ("fluid_pressure", None)
    assert model.get_variable_field("u") == ("mech_u", None)
    with pytest.raises(KeyError):
        model.get_variable_field("temperature")


def test_composite_updates_both_children_and_keeps_reports():
    state, report = build().update_state({}, object(), object())
    assert state == {"fluid": 1, "mech": 1}
    assert report["fluid"]["updated"] == "fluid"
    assert report["mechanics"]["updated"] == "mech"


def test_composite_validates_state_through_both_children():
    state = build().validate_state({})
    assert state == {"valid_fluid": True, "valid_mech": True}


def test_base_refuses_physics_owned_by_concrete_models():
    model = build()
    with pytest.raises(RuntimeError, match="abstract"):
        model.get_equations()
    with pytest.raises(RuntimeError, match="abstract"):
        model.compute_coupling_terms()
