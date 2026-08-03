"""Direct MRST 2024a certification for both shale fluid catalogs."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.shale import get_shale_comp_fluid_case


FIXTURE = Path(__file__).parents[2] / "tests" / "fixtures" / "shale" / "fluid_catalog.mat"


def test_complete_shale_fluid_catalog_matches_direct_mrst_fixture():
    ref = loadmat(FIXTURE, simplify_cells=True)["utility"]
    for row, case in enumerate(ref["case_names"]):
        fluid, info = get_shale_comp_fluid_case(str(case))
        n = fluid.n
        assert fluid.names == tuple(str(ref["component_names"][row]).split("|"))
        assert n == int(ref["count"][row])
        assert np.allclose(info.initial, ref["initial"][row, :n], rtol=2e-12)
        assert np.isclose(info.pressure, ref["pressure"][row], rtol=2e-12)
        assert np.isclose(info.T, ref["temperature"][row], rtol=2e-12)
        assert np.allclose(fluid.Tcrit, ref["Tcrit"][row, :n], rtol=2e-12)
        assert np.allclose(fluid.Pcrit, ref["Pcrit"][row, :n], rtol=2e-12)
        assert np.allclose(fluid.Vcrit, ref["Vcrit"][row, :n], rtol=2e-12)
        assert np.allclose(fluid.acentric, ref["acentric"][row, :n], rtol=2e-12)
        assert np.allclose(fluid.molarMass, ref["molar_mass"][row, :n], rtol=2e-12)
        assert np.allclose(fluid.get_binary_interaction(), ref["bic"][row, :n, :n], rtol=2e-12)

        if case == "eagleford":
            # The MRST source has eight zero entries for a six-component
            # mixture. PyMRST intentionally trims this malformed stream.
            assert int(ref["injection_count"][row]) == 8
            assert info.injection.size == n == 6
        else:
            assert info.injection.size == int(ref["injection_count"][row])
        assert np.allclose(info.injection, ref["injection"][row, :info.injection.size])

        isotherm = getattr(fluid, "isotherm", None)
        if isotherm is not None:
            assert np.allclose(isotherm, ref["isotherm"][row, :, :n], rtol=2e-12)


def test_main_and_utility_mrst_catalogs_share_identical_common_cases():
    fixture = loadmat(FIXTURE, simplify_cells=True)
    main = fixture["main"]
    utility = fixture["utility"]
    assert tuple(main["case_names"]) == tuple(utility["case_names"][:18])
    for field in ("count", "injection_count", "injection", "initial", "pressure",
                  "temperature", "Tcrit", "Pcrit", "Vcrit", "acentric",
                  "molar_mass", "bic", "isotherm"):
        left = np.asarray(main[field])
        right = np.asarray(utility[field])[:18]
        assert np.allclose(left, right, equal_nan=True)
