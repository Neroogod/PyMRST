from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_blackoil.utils import calculate_hydrocarbons_from_status_bo
from pymrst.utils.adi import init_variables_adi, value


FLUID = SimpleNamespace(
    rs_sat=lambda p: 10.0 + 0.1*p,
    rv_sat=lambda p: 0.2 + 0.01*p,
)


def test_status_switches_oil_gas_and_two_phase_cells():
    status = np.array([1, 2, 3])
    pressure = np.array([100.0, 200.0, 300.0])
    x = np.array([7.0, 0.9, 0.25])
    result = calculate_hydrocarbons_from_status_bo(
        FLUID, status, np.array([0.8, 0.7, 0.6]), x,
        np.zeros(3), np.zeros(3), pressure, True, True,
    )
    sg, rs, rv, rs_sat, rv_sat = result
    np.testing.assert_allclose(sg, [0.0, 0.7, 0.25])
    np.testing.assert_allclose(rs_sat, [20.0, 30.0, 40.0])
    np.testing.assert_allclose(rs, [7.0, 30.0, 40.0])
    np.testing.assert_allclose(rv_sat, [1.2, 2.2, 3.2])
    np.testing.assert_allclose(rv, [1.2, 0.9, 3.2])


def test_disabled_dissolution_preserves_supplied_ratios():
    rs = np.array([1.0, 2.0, 3.0])
    rv = np.array([0.1, 0.2, 0.3])
    result = calculate_hydrocarbons_from_status_bo(
        FLUID, [np.ones(3), np.zeros(3), np.zeros(3)], np.ones(3),
        np.zeros(3), rs, rv, np.ones(3), False, False,
    )
    assert result[1] is rs and result[3] is rs
    assert result[2] is rv and result[4] is rv


def test_switch_preserves_primary_variable_derivatives():
    p, x = init_variables_adi(np.array([100.0, 200.0, 300.0]),
                              np.array([7.0, 0.9, 0.25]))
    sg, rs, rv, _, _ = calculate_hydrocarbons_from_status_bo(
        FLUID, np.array([1, 2, 3]), np.array([0.8, 0.7, 0.6]), x,
        np.zeros(3), np.zeros(3), p, True, True,
    )
    np.testing.assert_allclose(value(sg), [0.0, 0.7, 0.25])
    np.testing.assert_allclose(rs.jac[1].diagonal(), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rv.jac[1].diagonal(), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(sg.jac[1].diagonal(), [0.0, 0.0, 1.0])
