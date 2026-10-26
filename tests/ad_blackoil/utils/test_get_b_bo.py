"""The legacy `get_b_o_bo_ad`/`get_b_g_bo_ad` (raw_fluid, snake-case names), kept for
ThreePhaseBlackOilModelAD; the MRST ports are checked by validate_ad_blackoil_bo_utils.py."""

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_blackoil.utils import get_b_g_bo_ad, get_b_o_bo_ad
from pymrst.utils.adi import init_variables_adi, value


class RecordingFluid:
    def __init__(self):
        self.calls = []

    def b_o(self, *args):
        self.calls.append(("oil", len(args)))
        if len(args) == 1:
            return 2.0 + 0.1 * args[0]
        pressure, rs, saturated = args
        return 2.0 + 0.1 * pressure + 0.2 * rs + 0.3 * saturated

    def b_g(self, *args):
        self.calls.append(("gas", len(args)))
        if len(args) == 1:
            return 3.0 + 0.4 * args[0]
        pressure, rv, liquid = args
        return 3.0 + 0.4 * pressure + 0.5 * rv + 0.6 * liquid


def test_dead_oil_and_dry_gas_use_pressure_only_dispatch():
    fluid = RecordingFluid()
    model = SimpleNamespace(raw_fluid=fluid, disgas=False, vapoil=False)
    p = np.array([10.0, 20.0])
    np.testing.assert_allclose(get_b_o_bo_ad(model, p, [9, 9], [1, 1]), 2 + 0.1*p)
    np.testing.assert_allclose(get_b_g_bo_ad(model, p, [9, 9], [1, 1]), 3 + 0.4*p)
    assert fluid.calls == [("oil", 1), ("gas", 1)]


def test_live_oil_and_wet_gas_forward_status_arguments():
    fluid = RecordingFluid()
    model = SimpleNamespace(fluid=fluid, disgas=True, vapoil=True)
    p = np.array([10.0, 20.0])
    rs = np.array([1.0, 2.0])
    rv = np.array([0.1, 0.2])
    gas = np.array([True, False])
    oil = np.array([False, True])
    np.testing.assert_allclose(get_b_o_bo_ad(model, p, rs, gas), 2 + .1*p + .2*rs + .3*gas)
    np.testing.assert_allclose(get_b_g_bo_ad(model, p, rv, oil), 3 + .4*p + .5*rv + .6*oil)
    assert fluid.calls == [("oil", 3), ("gas", 3)]


def test_helpers_preserve_adi_derivatives():
    fluid = RecordingFluid()
    model = SimpleNamespace(fluid=fluid, disgas=True, vapoil=True)
    p, rs, rv = init_variables_adi(
        np.array([10.0, 20.0]), np.array([1.0, 2.0]), np.array([0.1, 0.2])
    )
    bo = get_b_o_bo_ad(model, p, rs, np.array([True, False]))
    bg = get_b_g_bo_ad(model, p, rv, np.array([False, True]))
    np.testing.assert_allclose(value(bo), [3.5, 4.4])
    np.testing.assert_allclose(value(bg), [7.05, 11.7])
    np.testing.assert_allclose(bo.jac[0].toarray(), 0.1*np.eye(2))
    np.testing.assert_allclose(bo.jac[1].toarray(), 0.2*np.eye(2))
    np.testing.assert_allclose(bg.jac[0].toarray(), 0.4*np.eye(2))
    np.testing.assert_allclose(bg.jac[2].toarray(), 0.5*np.eye(2))


@pytest.mark.parametrize("phase", ["oil", "gas"])
def test_missing_fluid_property_has_targeted_error(phase):
    model = SimpleNamespace(fluid=SimpleNamespace(), disgas=True, vapoil=True)
    function = get_b_o_bo_ad if phase == "oil" else get_b_g_bo_ad
    with pytest.raises(AttributeError, match=f"b_{phase[0]}"):
        function(model, np.array([1.0]), np.array([0.0]), np.array([False]))
