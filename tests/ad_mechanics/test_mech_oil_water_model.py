import pytest

from pymrst.modules.ad_mechanics.mech_blackoil import MechBlackOilModel
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_oil_water_model import (
    MechOilWaterModel,
)


def test_oil_water_is_a_distinct_model_owner():
    assert MechOilWaterModel is not MechBlackOilModel
    assert issubclass(MechOilWaterModel, MechBlackOilModel)
    assert MechOilWaterModel.phase_names == ("water", "oil")


def test_oil_water_rejects_live_oil_before_model_construction():
    with pytest.raises(ValueError, match="dissolved gas"):
        MechOilWaterModel(disgas=True)


def test_coupling_terms_use_old_and_new_full_displacements():
    class Mechanics:
        @staticmethod
        def volumetric_strain_full(u):
            return 2.0 * u

    model = object.__new__(MechOilWaterModel)
    model.mech = Mechanics()
    terms, pressure = model.compute_coupling_terms(
        {"u": 3.0, "pressure": 10.0}, {"u": 5.0, "pressure": 12.0}
    )
    assert terms == {"old": 6.0, "new": 10.0}
    assert pressure == 12.0


def test_coupling_requires_stored_full_displacement():
    model = object.__new__(MechOilWaterModel)
    with pytest.raises(KeyError, match="displacement"):
        model.compute_coupling_terms({"pressure": 1.0}, {"pressure": 2.0})
