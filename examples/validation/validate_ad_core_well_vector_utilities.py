"""Validate ``padRatesAndCompi``, ``wellSolToVector`` and ``recoverVars``."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    pad_rates_and_compi,
    recover_vars,
    well_sol_to_vector,
)
from pymrst.params.wells_and_bc import Well


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_well_vector_utilities.mat",
        squeeze_me=True,
    )
    model = SimpleNamespace(getActivePhases=lambda: np.array([False, True, True]))
    well = Well(
        cells=np.array([0]), type="rate", val=1.0, r=np.array([0.1]),
        dir="z", WI=np.array([1.0]), dZ=np.array([0.0]), name="W",
        compi=np.array([0.25, 0.75]), refDepth=0.0, sign=1,
    )
    rates, wells, active = pad_rates_and_compi(
        [np.array([2.0, 3.0]), np.array([4.0, 5.0])], [well], model
    )
    steps = [
        [
            {"sign": 1, "qWs": 1, "qOs": 2, "qGs": 3, "bhp": 100},
            {"sign": -1, "qWs": -4, "qOs": -5, "qGs": -6, "bhp": 90},
        ],
        [
            {"sign": 1, "qWs": 7, "qOs": 8, "qGs": 9, "bhp": 110},
            {"sign": -1, "qWs": -10, "qOs": -11, "qGs": -12, "bhp": 95},
        ],
    ]
    qw, qo, qg, bhp = well_sol_to_vector(steps)
    eq = SimpleNamespace(
        val=np.array([7.0, -1.0]),
        jac=[
            sp.csr_matrix([[2.0, 0.0], [1.0, 3.0]]),
            sp.csr_matrix([[4.0, 1.0], [0.0, 2.0]]),
            sp.csr_matrix([[1.0], [2.0]]),
        ],
    )
    x = recover_vars(eq, 1, [np.array([0.5, -1.0]), np.array([2.0])])
    checks = {
        "qW": (rates[0], ref["qW"]), "qO": (rates[1], ref["qO"]),
        "qG": (rates[2], ref["qG"]), "compi": (wells[0].compi, ref["compi"]),
        "active": (active, ref["active"]), "qWs": (qw, ref["qWs"]),
        "qOs": (qo, ref["qOs"]), "qGs": (qg, ref["qGs"]),
        "bhp": (bhp, ref["bhp"]), "recovered x": (x, ref["x"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
