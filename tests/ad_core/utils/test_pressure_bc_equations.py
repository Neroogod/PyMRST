from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    bc_to_ad_bc,
    combine_equations,
    pressure_bc_contrib,
    pressure_bc_contrib_adi,
)
from pymrst.utils.adi import init_variables_adi


def _fixture():
    grid = SimpleNamespace(
        faces=SimpleNamespace(
            neighbors=np.array([[-1, 0], [0, 1], [1, -1]]),
            centroids=np.array([[0, 0, 0], [0, 0, 1.5], [0, 0, 3.0]]),
        ),
        cells=SimpleNamespace(
            num=2, centroids=np.array([[0, 0, 1.0], [0, 0, 2.0]])
        ),
    )
    operators = SimpleNamespace(T_all=np.array([2.0, 4.0, 3.0]))
    boundary = SimpleNamespace(
        face=np.array([0, 2]),
        type=np.array(["pressure", "pressure"]),
        value=np.array([120.0, 80.0]),
    )
    return grid, operators, bc_to_ad_bc(grid, boundary)


def test_pressure_bc_legacy_blocks_inflow_and_adi_preserves_it():
    grid, operators, boundary = _fixture()
    pressure = [np.array([100.0, 100.0])]
    density = [np.array([2.0, 3.0])]
    mobility = [np.array([0.5, 0.25])]
    bfactor = [np.array([1.2, 1.5])]
    legacy, cells = pressure_bc_contrib(
        grid, operators, pressure, density, mobility, bfactor, boundary, gravity=10
    )
    # First face has boundary inflow and is blocked; second is production.
    assert np.allclose(legacy[0], [0.0, 56.25])
    assert np.array_equal(cells, [0, 1])

    (p_ad,) = init_variables_adi(pressure[0])
    adi, _ = pressure_bc_contrib_adi(
        grid, operators, [p_ad], density, mobility, bfactor, boundary, gravity=10
    )
    assert np.allclose(adi[0].val, [-48.0, 56.25])
    assert adi[0].jac[0].shape == (2, 2)


def test_combine_equations_numeric_and_ad():
    assert np.array_equal(combine_equations([1, 2], 3), [1, 2, 3])
    pressure, bhp = init_variables_adi(np.array([1.0, 2.0]), 5.0)
    combined = combine_equations(3 * pressure + bhp, 2 * bhp)
    assert np.array_equal(combined.val, [8.0, 11.0, 10.0])
    assert np.array_equal(
        combined.jac[0].toarray(),
        [[3, 0, 1], [0, 3, 1], [0, 0, 2]],
    )
