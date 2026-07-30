"""Validate ``CNV_MBConvergence`` and ``getGridSYMRCMOrdering`` against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.ad_core.utils import (
    cnv_mb_convergence_report,
    get_grid_symrcm_ordering,
)


class FlowProperties:
    def getProperty(self, model, state, name):
        return [np.array([2.0, 4.0]), np.array([0.5, 0.5])]


class Model:
    operators = SimpleNamespace(pv=np.array([10.0, 20.0]))
    toleranceMB = 0.25
    toleranceCNV = 0.5
    FlowPropertyFunctions = FlowProperties()

    def getPhaseNames(self):
        return ["W", "O"], ["water", "oil"]


class Problem:
    state = {}
    dt = 2.0
    equations = [
        np.array([99.0]),
        np.array([1.0, -3.0]),
        np.array([2.0, 2.0]),
    ]

    def indexOfEquationName(self, name):
        return {"water": 1, "oil": 2}[name]


def _strings(value):
    return [str(item) for item in np.atleast_1d(value).ravel()]


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_cnv_grid_ordering.mat",
        squeeze_me=True,
    )
    converged, values, evaluated, names = cnv_mb_convergence_report(
        Model(), Problem()
    )
    grid = SimpleNamespace(
        faces=SimpleNamespace(neighbors=np.array([
            [-1, 0], [0, 1], [1, 2], [1, 3], [3, 4], [4, -1]
        ])),
        cells=SimpleNamespace(num=5),
    )
    checks = {
        "converged": (converged, ref["converged"]),
        "values": (values, ref["values"]),
        "evaluated": (evaluated, ref["evaluated"]),
        "ordering": (
            get_grid_symrcm_ordering(grid) + 1,
            ref["ordering"],
        ),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual, dtype=float).squeeze()
        expected = np.asarray(expected, dtype=float).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    expected_names = _strings(ref["names"])
    names_ok = names == expected_names
    print(f"{'ok  ' if names_ok else 'FAIL'} names: {names!r}")
    failed += not names_ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
