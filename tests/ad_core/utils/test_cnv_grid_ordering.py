from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils import (
    cnv_mb_convergence_report,
    get_grid_symrcm_ordering,
)


class _FlowProperties:
    def getProperty(self, model, state, name):
        assert name == "ShrinkageFactors"
        return [np.array([2.0, 4.0]), np.array([0.5, 0.5])]


class _Model:
    operators = SimpleNamespace(pv=np.array([10.0, 20.0]))
    FlowPropertyFunctions = _FlowProperties()
    toleranceCNV = 0.5
    toleranceMB = 0.25

    def getPhaseNames(self):
        return ["W", "O"], ["water", "oil"]


class _Problem:
    state = {}
    dt = 2.0
    equations = [
        np.array([99.0]),
        np.array([1.0, -3.0]),
        np.array([2.0, 2.0]),
    ]

    def indexOfEquationName(self, name):
        return {"water": 1, "oil": 2}[name]


def test_cnv_mb_model_problem_wrapper():
    converged, values, evaluated, names = cnv_mb_convergence_report(
        _Model(), _Problem()
    )
    # B averages are 0.375 and 2.0.
    assert np.allclose(values, [0.1125, 0.8, 0.05, 16.0 / 30.0])
    assert np.array_equal(converged, [True, False, True, False])
    assert np.array_equal(evaluated, [False, True, True])
    assert names == ["CNV_W", "CNV_O", "MB_W", "MB_O"]


def test_grid_symrcm_ordering_native_and_imported():
    native = SimpleNamespace(
        faces=SimpleNamespace(neighbors=np.array([
            [-1, 0], [0, 1], [1, 2], [1, 3], [3, 4], [4, -1]
        ])),
        cells=SimpleNamespace(num=5),
    )
    imported = SimpleNamespace(
        faces=SimpleNamespace(neighbors=np.array([
            [0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 0]
        ])),
        cells=SimpleNamespace(num=5),
    )
    native_order = get_grid_symrcm_ordering(native)
    imported_order = get_grid_symrcm_ordering(imported)
    assert np.array_equal(native_order, imported_order)
    assert np.array_equal(np.sort(native_order), np.arange(5))
