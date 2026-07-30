from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import (
    build_corner_pt_pillars,
    cart2active,
    check_and_repair_zcorn,
)


def test_build_corner_pt_pillars_collapsed_and_scaled():
    grdecl = {
        "COORD": np.array([
            0.0, 0.0, 0.0, 10.0, 20.0, 10.0,
            4.0, 5.0, 3.0, 4.0, 5.0, 3.0,
        ]),
        "ZCORN": np.array([2.0, 8.0]),
    }
    x, y, z = build_corner_pt_pillars(grdecl)
    assert np.array_equal(x, [[0.0, 10.0], [4.0, 4.0]])
    assert np.array_equal(y, [[0.0, 20.0], [5.0, 5.0]])
    assert np.array_equal(z, [[0.0, 10.0], [2.0, 8.0]])

    x, y, z = build_corner_pt_pillars(grdecl, scale=True)
    assert np.allclose(x, [[2.0, 8.4], [4.0, 4.0]])
    assert np.allclose(y, [[4.0, 20.0], [5.0, 5.0]])
    assert np.array_equal(z, np.tile([2.0, 8.0], (2, 1)))


def test_cart2active_omits_inactive_or_preserves_shape():
    grid = SimpleNamespace(
        cartDims=(2, 2, 2),
        cells=SimpleNamespace(indexMap=np.array([0, 2, 3, 6])),
    )
    query = np.array([[6, 1], [0, 3]])
    assert np.array_equal(cart2active(grid, query), [3, 0, 2])
    assert np.array_equal(
        cart2active(grid, query, preserve_shape=True),
        [[3, -1], [0, 2]],
    )


def _zcorn(top, bottom):
    z = np.empty((2, 2, 2), dtype=float)
    z[:, :, 0] = top
    z[:, :, 1] = bottom
    return z.ravel(order="F")


def test_check_and_repair_zcorn_repairs_cell_inversion():
    values = _zcorn(5.0, 3.0)
    repaired, counts = check_and_repair_zcorn(
        values, [1, 1, 1], return_counts=True
    )
    z = repaired.reshape((2, 2, 2), order="F")
    assert np.array_equal(z[:, :, 0], z[:, :, 1])
    assert counts == {
        "top_below_bottom": 4,
        "bottom_below_lower_top": 0,
    }


def test_check_and_repair_zcorn_repairs_active_layer_overlap_only():
    z = np.empty((2, 2, 4), dtype=float)
    z[:, :, 0] = 0.0
    z[:, :, 1] = 6.0
    z[:, :, 2] = 5.0
    z[:, :, 3] = 10.0
    values = z.ravel(order="F")

    repaired, counts = check_and_repair_zcorn(
        values, [1, 1, 2], return_counts=True
    )
    out = repaired.reshape(z.shape, order="F")
    assert np.all(out[:, :, 2] == 6.0)
    assert counts["bottom_below_lower_top"] == 4

    unchanged = check_and_repair_zcorn(
        values, [1, 1, 2], active=[1, 0]
    )
    assert np.array_equal(unchanged, values)


def test_check_and_repair_zcorn_valid_input_is_unchanged():
    values = _zcorn(1.0, 2.0)
    assert np.array_equal(
        check_and_repair_zcorn(values, [1, 1, 1]), values
    )
