from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils import get_equil_pc


class _PC:
    def hasJFunctionScaler(self, pair):
        return pair == "OW"

    def getJFunctionStaticRatio(self, model):
        return np.array([2.0, 3.0, 5.0])

    def getSurfaceTension(self, pair):
        assert pair == "OW"
        return 0.04


class _Model:
    water = True
    oil = True
    gas = True
    fluid = SimpleNamespace(
        pcOW=[lambda s: 10 * (1 - s), lambda s: 20 * (1 - s)],
        pcOG=[lambda s: 30 * s, lambda s: 40 * s],
    )
    FlowPropertyFunctions = SimpleNamespace(CapillaryPressure=_PC())

    def getActivePhases(self):
        return np.ones(3, dtype=bool)

    def getPhaseIndex(self, phase):
        return {"W": 0, "O": 1, "G": 2}[phase]


def test_region_curves_signs_and_j_function_scaling():
    curves, signs, scalers = get_equil_pc(_Model(), 1, [2, 0])
    saturation = np.array([0.25, 0.75])
    assert np.allclose(curves[0](saturation), [15.0, 5.0])
    assert np.allclose(curves[1](saturation), 0.0)
    assert np.allclose(curves[2](saturation), [10.0, 30.0])
    assert np.array_equal(signs, [-1.0, 1.0, 1.0])
    assert np.allclose(scalers[:, 0], [0.2, 0.08])
    assert np.allclose(scalers[:, 1:], 1.0)


def test_water_gas_fallback_and_missing_water_curve():
    model = _Model()
    model.water = True
    model.oil = False
    model.gas = True
    model.fluid = SimpleNamespace(pcWG=lambda s: 7 * s)
    model.FlowPropertyFunctions = None
    model.getActivePhases = lambda: np.ones(2, dtype=bool)
    model.getPhaseIndex = lambda phase: {"W": 0, "G": 1}[phase]
    curves, signs, scalers = get_equil_pc(model, 0, [0, 1])
    saturation = np.array([0.2, 0.6])
    assert np.allclose(curves[0](saturation), 0.0)
    assert np.allclose(curves[1](saturation), [1.4, 4.2])
    assert np.array_equal(signs, [-1.0, 1.0])
    assert np.allclose(scalers, 1.0)
