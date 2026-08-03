import numpy as np
import pytest

from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.black_oil_fluid_model import BlackOilFluidModel
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.oil_water_fluid_model import OilWaterFluidModel
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.water_fluid_model import WaterFluidModel


@pytest.mark.parametrize(
    "model, primary, fields",
    [
        (WaterFluidModel(None, None, None), ["pressure"], ["wellSol", "pressure"]),
        (OilWaterFluidModel(None, None, None), ["pressure", "sW"], ["wellSol", "pressure", "s", "sW", "sO", "water", "oil"]),
        (BlackOilFluidModel(None, None, None), ["pressure", "sW", "x"], ["wellSol", "pressure", "s", "rs", "rv", "sW", "sG", "sO", "water", "oil", "gas"]),
    ],
)
def test_fluid_helpers_declare_mrst_contract(model, primary, fields):
    assert model.get_all_primary_variables() == primary
    assert model.get_all_vars_names() == fields
    with pytest.raises(RuntimeError, match="coupled helper"):
        model.get_equations()


def test_oil_water_variable_aliases_and_state_update():
    model = OilWaterFluidModel(None, None, None)
    assert model.get_variable_field("water") == ("s", 0)
    assert model.get_variable_field("oil") == ("s", 1)
    problem = type("Problem", (), {"primaryVariables": ["pressure", "sW", "xd"]})()
    state, report = model.update_state(
        {"pressure": np.array([10.0]), "sW": np.array([0.2]), "xd": np.array([0.0])},
        problem,
        [np.array([2.0]), np.array([0.1]), np.array([5.0])],
    )
    assert state["pressure"] == pytest.approx([12.0])
    assert state["sW"] == pytest.approx([0.3])
    assert state["sO"] == pytest.approx([0.7])
    assert state["xd"] == pytest.approx([0.0])
    assert report == {}


def test_helper_adds_facility_primary_variables():
    class Facility:
        @staticmethod
        def getAllPrimaryVariables(well_sol):
            return [object(), object()], ["qWs", "bhp"], [0, 1]

    model = WaterFluidModel(None, None, None)
    model.FacilityModel = Facility()
    assert model.get_all_primary_variables({"wellSol": []}) == ["pressure", "qWs", "bhp"]


def test_update_rejects_missing_coupled_variable():
    model = BlackOilFluidModel(None, None, None)
    problem = type("Problem", (), {"primary_variables": ["pressure", "sW", "xd"]})()
    with pytest.raises(ValueError, match="missing fluid variables"):
        model.update_state({}, problem, [0.0, 0.0, 0.0])
