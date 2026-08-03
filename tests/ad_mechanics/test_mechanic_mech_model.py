import numpy as np
import pytest

from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.mechanic_mech_model import (
    MechanicMechModel,
)


def test_helper_declares_mrst_variable_contract():
    model = object.__new__(MechanicMechModel)
    assert model.get_all_primary_variables() == ["xd"]
    assert model.get_all_vars_names() == ["xd", "uu", "u", "stress", "strain", "vdiv"]
    assert model.get_variable_field("uu") == ("u", None)
    assert model.get_variable_field("vdiv") == ("volumetric_strain", None)
    with pytest.raises(KeyError):
        model.get_variable_field("temperature")


def test_helper_refuses_standalone_equation_assembly():
    model = object.__new__(MechanicMechModel)
    with pytest.raises(RuntimeError, match="coupled helper"):
        model.get_equations()


def test_update_state_rejects_wrong_mechanics_block_size():
    model = object.__new__(MechanicMechModel)
    model.operators = type("Operators", (), {"is_dir": np.array([False, True, False])})()
    with pytest.raises(ValueError, match="expected 2"):
        model.update_state({"xd": np.zeros(2)}, np.zeros(3))
