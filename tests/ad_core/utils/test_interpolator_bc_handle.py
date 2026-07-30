from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    HandleStruct,
    bc_to_ad_bc,
    get_multi_dim_interpolator,
)
from pymrst.utils.adi import ADI


def test_multidimensional_interpolator_numeric_ad_and_descending_axis():
    x = np.array([2.0, 1.0, 0.0])
    y = np.array([0.0, 3.0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    table = 2.0 * xx + 3.0 * yy
    interpolate, _, epsilon = get_multi_dim_interpolator([x, y], table)
    assert np.allclose(epsilon, [0.5, 1.5])
    assert np.allclose(interpolate([0.5, 1.5], [1.0, 2.0]), [4.0, 9.0])

    x_ad = ADI(np.array([0.5, 1.5]), [sp.eye(2, format="csr")])
    result = interpolate(x_ad, np.array([1.0, 2.0]))
    assert np.allclose(result.val, [4.0, 9.0])
    assert np.allclose(result.jac[0].toarray(), 2.0 * np.eye(2))


def test_bc_to_ad_bc_builds_reduced_incidence():
    grid = SimpleNamespace(
        faces=SimpleNamespace(neighbors=np.array([[-1, 0], [0, 1], [1, -1]])),
        cells=SimpleNamespace(num=2),
    )
    boundary = SimpleNamespace(face=np.array([0, 2]), value=np.array([10, 20]))
    converted = bc_to_ad_bc(grid, boundary)
    assert np.array_equal(converted.cell2bcface.toarray(), np.eye(2))
    assert np.array_equal(converted.bcface2cell.toarray(), np.eye(2))
    imported = SimpleNamespace(
        faces=SimpleNamespace(neighbors=np.array([[0, 1], [1, 2], [2, 0]])),
        cells=SimpleNamespace(num=2),
    )
    imported_bc = SimpleNamespace(face=np.array([0, 2]))
    assert np.array_equal(
        bc_to_ad_bc(imported, imported_bc).cell2bcface.toarray(), np.eye(2)
    )


def test_handle_struct_reduces_ad_and_cell_values_and_copies():
    ad = ADI(np.array([1.0, 2.0]), [sp.eye(2, format="csr")])
    wrapped = HandleStruct({"ad": ad, "parts": [np.array([3]), np.array([4, 5])]})
    assert wrapped.isfield("ad") and wrapped.struct_prop_evaluated("parts")
    reduced = wrapped.reduce()
    assert np.array_equal(reduced["ad"], [1.0, 2.0])
    assert np.array_equal(reduced["parts"], [3, 4, 5])
    copied = wrapped.copy()
    copied.ad.val[0] = 99
    assert wrapped.ad.val[0] == 1.0
