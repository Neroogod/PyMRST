from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.utils import (
    compute_init_aquifer,
    get_flux_and_props_gas_bo,
    get_flux_and_props_oil_bo,
    get_flux_and_props_water_bo,
)
from pymrst.modules.ad_core.models.aquifers.aquifer import (
    AquiferConnections,
    equilibrium_aquifer_pressure,
)
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params.rock import make_rock
from pymrst.utils.adi import init_variables_adi, value


def _model():
    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    rock = make_rock(grid, 1.0, 0.2)
    fluid = SimpleNamespace(
        rhoS=np.array([1000.0, 700.0, 2.0]),
        pc_ow=lambda sw: 2.0*sw,
        pc_og=lambda sg: 3.0*sg,
        b_w=lambda p: 1.0 + 0.01*p,
        mu_w=lambda p: 2.0 + 0.001*p,
        b_o=lambda p, rs, sat: 1.2 + 0.02*p + 0.03*rs + 0.04*sat,
        mu_o=lambda p, rs, sat: 3.0 + 0.002*p + 0.01*rs,
        b_g=lambda p, rv, sat: 4.0 + 0.04*p + 0.05*rv + 0.06*sat,
        mu_g=lambda p, rv, sat: 0.5 + 0.003*p + 0.02*rv,
    )
    return SimpleNamespace(
        G=grid, g=grid, rock=rock, raw_fluid=fluid, fluid=fluid,
        operators=setup_operators_tpfa(grid, rock),
        disgas=True, vapoil=True, gravity=0.0,
    )


def test_phase_helpers_return_full_mrst_tuple_and_preserve_ad():
    model = _model()
    p, sw, sg, rs, rv = init_variables_adi(
        np.array([10.0, 12.0, 11.0]),
        np.array([0.2, 0.3, 0.25]),
        np.array([0.1, 0.15, 0.2]),
        np.array([2.0, 2.5, 3.0]),
        np.array([0.01, 0.02, 0.03]),
    )
    krw, kro, krg = sw, 1.0-sw-sg, sg
    common = (model.operators.T, np.array([0.1, -0.2]))
    water = get_flux_and_props_water_bo(model, p, sw, krw, *common)
    oil = get_flux_and_props_oil_bo(
        model, p, kro, kro, *common, rs, np.array([True, False, True])
    )
    gas = get_flux_and_props_gas_bo(
        model, p, sg, krg, *common, rv, np.array([False, True, True])
    )
    for result in (water, oil, gas):
        assert len(result) == 8
        assert np.asarray(value(result[0])).shape == (2,)
        assert np.asarray(value(result[1])).shape == (3,)
        assert len(result[0].jac) == 5
    np.testing.assert_array_equal(water[5], value(water[6]) <= 0)
    np.testing.assert_array_equal(oil[5], value(oil[6]) <= 0)
    np.testing.assert_array_equal(gas[5], value(gas[6]) <= 0)


def test_compute_init_aquifer_uses_zero_flux_equilibrium_and_given_volumes():
    conn = AquiferConnections(
        cells=np.array([0, 1, 2]), aquid=np.array([0, 0, 1]),
        j=np.array([2.0, 1.0, 3.0]), alpha=np.array([0.5, 0.5, 1.0]),
        c=np.full(3, 1e-9), depth_conn=np.array([100.0, 120.0, 90.0]),
        depth_aq=np.array([80.0, 80.0, 70.0]), c_aq=np.full(2, 1e-9),
    )
    p_w = np.array([100.0, 130.0, 90.0])
    b_w = np.array([1.0, 1.1, 0.9])
    aquifer = SimpleNamespace(
        connections=conn, state_arg_keys=("aquifer_p", "aquifer_V"),
        _water_props=lambda model, state: (p_w, b_w),
        pressures=np.zeros(2), volumes=np.zeros(2),
    )
    model = SimpleNamespace(
        aquifer=aquifer, raw_fluid=SimpleNamespace(rhoS=np.array([1000., 700., 2.])),
        fluid=SimpleNamespace(rho_s=np.array([1000., 700., 2.])), gravity=9.81,
    )
    state = {}
    volumes = np.array([1e6, 2e6])
    compute_init_aquifer(model, state, volumes)
    expected = equilibrium_aquifer_pressure(conn, p_w, b_w, 1000.0, 9.81)
    np.testing.assert_allclose(state["aquifer_p"], expected)
    np.testing.assert_allclose(state["aquifer_V"], volumes)
    np.testing.assert_allclose(aquifer.pressures, expected)
