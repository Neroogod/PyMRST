import numpy as np
import pytest

from pymrst.modules.shale import SorbedCompositionalFluid, get_shale_comp_fluid_case
from pymrst.modules.shale.compositional_fns.get_shale_comp_fluid_case import (
    get_shale_comp_fluid_case as get_main_shale_comp_fluid_case,
)


def test_barnett_case_carries_isotherm_diffusion_and_bic():
    fluid, info = get_shale_comp_fluid_case("barnett3comps")
    assert isinstance(fluid, SorbedCompositionalFluid)
    assert fluid.names == ("Methane", "Ethane", "n-Propane")
    assert np.allclose(fluid.isotherm[1], [2.99, 4.86, 9.56])
    assert np.allclose(info.diffusion_coefficients, [2.8e-7, 2.5e-7, 1.9e-7])
    assert np.allclose(fluid.bic, fluid.bic.T)


def test_bakken_case_preserves_eight_component_stream_and_properties():
    fluid, info = get_shale_comp_fluid_case("bakken")
    assert fluid.n == 8
    assert info.initial.sum() == pytest.approx(1.0)
    assert info.injection.sum() == pytest.approx(1.0)
    assert info.parachor.size == 8
    assert fluid.bic[0, 1] == pytest.approx(0.005)


def test_base_case_delegation_and_unknown_case():
    fluid, info = get_shale_comp_fluid_case("simple")
    assert fluid.n == 3
    assert info.initial.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown shale"):
        get_shale_comp_fluid_case("not-a-case")


@pytest.mark.parametrize(
    ("case", "count", "pressure_psia"),
    [("eagleford", 6, 8125), ("bakken_light", 5, 5700),
     ("bakken_light_5comps", 5, 4000)],
)
def test_field_fluid_cases_preserve_dimensions_and_reference_tables(
    case, count, pressure_psia
):
    fluid, info = get_shale_comp_fluid_case(case)
    from pymrst.utils.units import PSIA

    assert fluid.n == count
    assert info.initial.size == count
    assert info.injection.size == count
    assert info.initial.sum() == pytest.approx(1.0)
    assert info.pressure == pytest.approx(pressure_psia * PSIA)
    assert info.parachor.size == count
    assert np.allclose(fluid.bic, fluid.bic.T)


@pytest.mark.parametrize(
    ("case", "names", "pressure", "temperature"),
    [
        ("fourmix", ("Methane", "Nitrogen", "n-Pentane", "n-Decane"), 7.5e6, 323.15),
        ("liquid_initial", ("O2", "H2O"), 7.5e6, 303.15),
        ("vapor_initial", ("O2", "H2O"), 7.5e6, 303.15),
        ("watertracer", ("Tracer1", "Tracer2"), 7.5e6, 303.15),
        ("verysimple", ("CarbonDioxide", "n-Decane"), 7.5e6, 423.15),
        ("onlydecane", ("n-Decane",), 7.5e6, 423.15),
        ("watermethane", ("Methane", "Water"), 7.5e6, 423.15),
    ],
)
def test_main_table_catalog_matches_mrst_metadata(case, names, pressure, temperature):
    fluid, info = get_shale_comp_fluid_case(case)
    assert fluid.names == names
    assert info.pressure == pytest.approx(pressure)
    assert info.T == pytest.approx(temperature)
    assert fluid.n == info.initial.size == info.injection.size


def test_twenty_component_diagnostic_case_preserves_mrst_stream_exactly():
    fluid, info = get_shale_comp_fluid_case("20components")
    assert fluid.n == 20
    assert np.allclose(info.injection, np.arange(1, 21) / 20)
    assert np.allclose(info.initial, np.ones(20) / 20)
    assert info.injection.sum() == pytest.approx(10.5)


@pytest.mark.parametrize(
    ("case", "names", "injection", "initial"),
    [
        ("oil_1", ("Methane", "CarbonDioxide", "n-Decane"),
         [1, 0, 0], [.25, .25, .5]),
        ("oil_1_modified", ("Methane", "CarbonDioxide", "n-Decane", "Nitrogen"),
         [0, 0, 0, 1], [.25, .25, .5, 0]),
        ("oil_1_modified_2", ("Nitrogen", "Methane", "CarbonDioxide", "n-Decane"),
         [1, 0, 0, 0], [0, .25, .25, .5]),
    ],
)
def test_extended_oil_catalog_matches_mrst_component_order(case, names, injection, initial):
    fluid, info = get_shale_comp_fluid_case(case)
    assert fluid.names == names
    assert np.allclose(info.injection, injection)
    assert np.allclose(info.initial, initial)


@pytest.mark.parametrize(
    ("case", "names", "injection", "bic_12"),
    [
        ("barnett3comps_modified_2",
         ("Methane", "Ethane", "n-Propane", "CarbonDioxide"), [1, 0, 0, 0], .01),
        ("barnett3comps_nat_vars",
         ("Methane", "Ethane", "n-Propane", "CarbonDioxide"), [1, 0, 0, 0], .01),
        ("barnett3comps_modified",
         ("CarbonDioxide", "Methane", "Ethane", "n-Propane"), [0, 0, 0, 1], .005),
    ],
)
def test_extended_barnett_catalog_preserves_order_isotherm_and_bic(
    case, names, injection, bic_12
):
    fluid, info = get_shale_comp_fluid_case(case)
    assert isinstance(fluid, SorbedCompositionalFluid)
    assert fluid.names == names
    assert np.allclose(info.injection, injection)
    assert fluid.isotherm.shape == (2, 4)
    assert fluid.bic[1, 2] == pytest.approx(bic_12)
    assert np.allclose(fluid.bic, fluid.bic.T)


def test_mrst_catalog_owners_have_distinct_case_boundaries():
    # The compositionalFns owner has 18 cases; the utils owner extends it by 7.
    get_main_shale_comp_fluid_case("bakken_light")
    with pytest.raises(ValueError, match="unknown shale"):
        get_main_shale_comp_fluid_case("oil_1")
    fluid, _ = get_shale_comp_fluid_case("oil_1")
    assert fluid.names == ("Methane", "CarbonDioxide", "n-Decane")
