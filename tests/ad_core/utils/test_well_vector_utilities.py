from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    pad_rates_and_compi,
    recover_vars,
    well_sol_to_vector,
)
from pymrst.params.wells_and_bc import Well


def _well(compi):
    return Well(
        cells=np.array([0]), type="rate", val=1.0, r=np.array([0.1]),
        dir="z", WI=np.array([1.0]), dZ=np.array([0.0]), name="W",
        compi=np.asarray(compi, float), refDepth=0.0, sign=1,
    )


def test_pad_rates_and_compositions_for_oil_gas_model():
    model = SimpleNamespace(getActivePhases=lambda: np.array([False, True, True]))
    rates, wells, active = pad_rates_and_compi(
        [np.array([2.0, 3.0]), np.array([4.0, 5.0])],
        [_well([0.25, 0.75])], model,
    )
    assert np.array_equal(active, [False, True, True])
    assert np.array_equal(rates[0], [0.0, 0.0])
    assert np.array_equal(rates[1], [2.0, 3.0])
    assert np.array_equal(rates[2], [4.0, 5.0])
    assert np.array_equal(wells[0].compi, [0.0, 0.25, 0.75])


def test_well_sol_to_vector_applies_well_sign():
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
    assert np.array_equal(qw, [[1, 4], [7, 10]])
    assert np.array_equal(qo, [[2, 5], [8, 11]])
    assert np.array_equal(qg, [[3, 6], [9, 12]])
    assert np.array_equal(bhp, [[100, 90], [110, 95]])


def test_recover_vars_satisfies_eliminated_equation():
    eq = SimpleNamespace(
        val=np.array([7.0, -1.0]),
        jac=[
            sp.csr_matrix([[2.0, 0.0], [1.0, 3.0]]),
            sp.csr_matrix([[4.0, 1.0], [0.0, 2.0]]),
            sp.csr_matrix([[1.0], [2.0]]),
        ],
    )
    known0 = np.array([0.5, -1.0])
    known2 = np.array([2.0])
    x = recover_vars(eq, 1, [known0, known2])
    residual = eq.val + eq.jac[0] @ known0 + eq.jac[1] @ x + eq.jac[2] @ known2
    assert np.allclose(residual, 0.0)
