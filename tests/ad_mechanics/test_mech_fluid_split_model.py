import pytest

from pymrst.modules.ad_mechanics.models.splitting.helpermodels.mech_fluid_split_model import (
    MechFluidSplitModel,
)


class Child:
    def __init__(self, names):
        self.names = names

    def get_all_vars_names(self):
        return self.names

    def get_variable_field(self, name):
        return f"fluid_{name}", None

    def validate_state(self, state):
        out = dict(state)
        out[f"validated_{self.names[0]}"] = True
        return out


def build(kind="oil water"):
    return MechFluidSplitModel(None, None, None, Child(["xd", "u"]), Child(["pressure", "sW"]), fluid_model_type=kind)


def test_phase_flags_and_variable_routing():
    model = build()
    assert (model.water, model.oil, model.gas) == (True, True, False)
    assert model.get_variable_field("u") == ("u", slice(None))
    assert model.get_variable_field("pressure") == ("fluid_pressure", None)


def test_all_four_state_synchronization_directions():
    model = build()
    state = {"pressure": 10, "sW": 0.2, "xd": 3, "u": 4, "extra": 9}
    assert model.sync_w_state_from_state(state) == {"pressure": 10, "sW": 0.2}
    assert model.sync_m_state_from_state(state) == {"xd": 3, "u": 4}
    assert model.sync_state_from_w_state({"xd": 3}, {"pressure": 12, "sW": 0.3}) == {"xd": 3, "pressure": 12, "sW": 0.3}
    assert model.sync_state_from_m_state({"pressure": 12}, {"xd": 5, "u": 6}) == {"pressure": 12, "xd": 5, "u": 6}


def test_state_validation_visits_fluid_and_mechanics():
    state = build().validate_state({})
    assert state["validated_pressure"]
    assert state["validated_xd"]


def test_base_requires_concrete_split_physics():
    model = build()
    with pytest.raises(RuntimeError, match="abstract"):
        model.step_function()
    with pytest.raises(RuntimeError, match="abstract"):
        model.compute_mech_term()
