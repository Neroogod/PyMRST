import numpy as np
import pytest

from pymrst.modules.ad_mechanics.mech_blackoil import MechBlackOilModel as BlackOilKernel
from pymrst.modules.ad_mechanics.mechanics import MechanicModel as MechanicsKernel
from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import (
    MechBlackOilModel,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_water_model import (
    MechWaterModel,
)
from pymrst.modules.ad_mechanics.poromech import MechWaterModel as LegacyWaterKernel
from pymrst.modules.ad_mechanics.models.poromechanics import MechWaterModel as WaterKernel


def test_layout_owners_are_distinct_operational_subclasses():
    assert LegacyWaterKernel is WaterKernel
    assert MechanicModel is MechanicsKernel
    assert MechWaterModel is not WaterKernel
    assert issubclass(MechWaterModel, WaterKernel)
    # The old root module is compatibility-only after strict kernel
    # relocation; it must resolve to the MRST-layout owner itself.
    assert MechBlackOilModel is BlackOilKernel


def test_mechanics_model_exposes_mrst_contract():
    model = object.__new__(MechanicModel)
    assert model.get_all_primary_variables() == ["xd"]
    assert model.get_all_vars_names() == ["xd", "uu", "u", "stress", "strain", "vdiv"]
    assert model.get_valid_driving_forces() == {"fluidp": None}
    assert model.get_variable_field("stress") == ("stress", slice(None))
    with pytest.raises(KeyError):
        model.get_variable_field("saturation")


@pytest.mark.parametrize("owner", [MechWaterModel, MechBlackOilModel])
def test_coupled_layout_owners_require_full_displacement(owner):
    model = object.__new__(owner)
    with pytest.raises(KeyError, match="displacement"):
        model.compute_coupling_terms({"pressure": np.array([1.0])}, {"pressure": np.array([2.0])})
