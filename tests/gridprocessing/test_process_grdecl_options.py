import numpy as np

from pymrst.gridprocessing import process_grdecl
from pymrst.gridprocessing.testgrids import simple_grdecl


def _disconnected_deck(*, pinch=False):
    deck = simple_grdecl([1, 1, 3], 0.0, flat=True)
    deck["ACTNUM"] = np.array([1, 0, 1])
    if pinch:
        deck["PINCH"] = [0.2, "GAP", 0.0, "TOPBOT", "TOP"]
    return deck


def test_process_grdecl_splits_disconnected_grid_by_default():
    result = process_grdecl(_disconnected_deck())
    assert isinstance(result, list)
    assert [part.cells.num for part in result] == [1, 1]
    assert all(part.cartDims == (1, 1, 3) for part in result)


def test_process_grdecl_can_keep_disconnected_grid_whole():
    grid = process_grdecl(_disconnected_deck(), split_disconnected=False)
    grid.check()
    assert grid.cells.num == 2


def test_process_grdecl_pinch_nnc_prevents_component_split():
    grid = process_grdecl(_disconnected_deck(pinch=True))
    grid.check()
    assert grid.cells.num == 2
    assert grid.nnc["cells"].shape == (1, 2)
    assert np.array_equal(grid.nnc["cells"], [[0, 1]])


def test_process_grdecl_rejects_negative_tolerance():
    with np.testing.assert_raises(ValueError):
        process_grdecl(_disconnected_deck(), tolerance=-1.0)


def test_process_grdecl_repair_zcorn_option_repairs_inverted_corner():
    deck = simple_grdecl([1, 1, 1], 0.0, flat=True)
    zcorn = np.asarray(deck["ZCORN"]).copy()
    zcorn[0] = zcorn[-1] + 1.0
    deck["ZCORN"] = zcorn
    grid = process_grdecl(
        deck, repair_zcorn=True, split_disconnected=False
    )
    grid.check()
    assert grid.cells.num == 1
