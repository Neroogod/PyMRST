import numpy as np
import pytest

from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.helpermodels.black_oil_fixed_stress_fluid_model import BlackOilFixedStressFluidModel
from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.helpermodels.oil_water_fixed_stress_fluid_model import OilWaterFixedStressFluidModel
from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.helpermodels.water_fixed_stress_fluid_model import WaterFixedStressFluidModel


@pytest.mark.parametrize(
    "model, fields",
    [
        (WaterFixedStressFluidModel(), ["wellSol", "pressure"]),
        (OilWaterFixedStressFluidModel(), ["wellSol", "pressure", "s"]),
        (BlackOilFixedStressFluidModel(), ["wellSol", "pressure", "s", "rs", "rv"]),
    ],
)
def test_fixed_stress_helpers_expose_force_and_field_contract(model, fields):
    assert model.get_all_vars_names() == fields
    assert "fixedStressTerms" in model.get_valid_driving_forces()


def test_fixed_stress_mechanical_strain_relation():
    forces = {
        "fixedStressTerms": {
            "old": {"sTerm": np.array([1.0]), "pTerm": np.array([2.0])},
            "new": {"sTerm": np.array([3.0]), "pTerm": np.array([4.0])},
        }
    }
    result = WaterFixedStressFluidModel.mechanical_strain_terms(
        {"pressure": np.array([5.0])}, {"pressure": np.array([7.0])}, forces
    )
    assert result["old"] == pytest.approx([11.0])
    assert result["new"] == pytest.approx([31.0])


def test_fixed_stress_equations_delegate_with_mechanics_term():
    model = OilWaterFixedStressFluidModel()
    captured = {}

    def assemble(state0, state, dt, mech_term, forces, **kwargs):
        captured.update(dt=dt, mech_term=mech_term)
        return "problem"

    model.equation_assembler = assemble
    forces = {
        "fixedStressTerms": {
            "old": {"sTerm": 1.0, "pTerm": 2.0},
            "new": {"sTerm": 3.0, "pTerm": 4.0},
        }
    }
    assert model.get_equations({"pressure": 5.0}, {"pressure": 7.0}, 10.0, forces) == "problem"
    assert captured["dt"] == 10.0
    assert captured["mech_term"] == {"old": 11.0, "new": 31.0}


def test_missing_fixed_stress_force_is_rejected():
    with pytest.raises(ValueError, match="fixedStressTerms"):
        WaterFixedStressFluidModel.mechanical_strain_terms(
            {"pressure": 1.0}, {"pressure": 2.0}, {}
        )
