from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.params.rock.grdecl2rock import grdecl2rock


def test_permh_and_permz_produce_horizontal_diagonal_tensor():
    rock = grdecl2rock({"PERMH": [10, 20], "PERMZ": [1, 2]})
    np.testing.assert_allclose(rock.perm, [[10, 10, 1], [20, 20, 2]])


def test_long_diagonal_names_and_reverse_cross_aliases():
    deck = {
        "PERMXX": [10], "PERMYY": [20], "PERMZZ": [30],
        "PERMYX": [1], "PERMZX": [2], "PERMZY": [3],
    }
    rock = grdecl2rock(deck)
    np.testing.assert_allclose(rock.perm, [[10, 1, 2, 20, 3, 30]])


def test_later_reverse_aliases_have_mrst_precedence():
    deck = {
        "PERMX": [10], "PERMY": [20], "PERMZ": [30],
        "PERMXY": [1], "PERMYX": [4],
        "PERMXZ": [2], "PERMZX": [5],
        "PERMYZ": [3], "PERMZY": [6],
    }
    np.testing.assert_allclose(
        grdecl2rock(deck).perm, [[10, 4, 5, 20, 6, 30]]
    )


def test_active_map_grid_and_logical_extension_select_all_properties():
    deck = {
        "PERMX": [1, 2, 3, 4], "PORO": [.1, .2, .3, .4],
        "NTG": [.9, .8, .7, .6], "ROCKTYPE": [1, 2, 3, 4],
        "MULTX": [10, 20, 30, 40], "MULTX_": [11, 21, 31, 41],
    }
    grid = SimpleNamespace(cells=SimpleNamespace(indexMap=np.array([0, 2])))
    from_grid = grdecl2rock(deck, grid)
    from_mask = grdecl2rock(deck, [True, False, True, False])
    np.testing.assert_allclose(from_grid.perm, from_mask.perm)
    np.testing.assert_allclose(from_grid.poro.ravel(), [.1, .3])
    np.testing.assert_allclose(from_grid.ntg.ravel(), [.9, .7])
    np.testing.assert_array_equal(from_grid.rocktype.ravel(), [1, 3])
    np.testing.assert_array_equal(from_grid.multipliers["x"].ravel(), [10, 30])
    np.testing.assert_array_equal(from_grid.multipliers["x_"].ravel(), [11, 31])


def test_fault_data_is_copied_verbatim_only_when_both_keywords_exist():
    faults = [["F1", 1, 2]]
    multflt = {"F1": 0.5}
    rock = grdecl2rock({
        "PERMX": [1], "FAULTS": faults, "MULTFLT": multflt,
    })
    assert rock.faultdata == {"faults": faults, "multflt": multflt}
    assert not hasattr(grdecl2rock({"PERMX": [1], "FAULTS": faults}), "faultdata")


@pytest.mark.parametrize("deck", [
    {"PERMXY": [1]},
    {"PERMX": [1], "PERMY": [2], "PERMXY": [3]},
])
def test_incomplete_tensor_keywords_are_rejected(deck):
    with pytest.raises(ValueError, match="consistent|incomplete"):
        grdecl2rock(deck)


def test_property_lengths_and_active_indices_are_validated():
    with pytest.raises(ValueError, match="PERMY contains"):
        grdecl2rock({"PERMX": [1, 2], "PERMY": [1], "PERMZ": [1, 2]})
    with pytest.raises(ValueError, match="PORO contains"):
        grdecl2rock({"PERMX": [1, 2], "PORO": [.2]})
    with pytest.raises(IndexError, match="outside"):
        grdecl2rock({"PERMX": [1, 2]}, [2])
    with pytest.raises(ValueError, match="logical actmap"):
        grdecl2rock({"PERMX": [1, 2]}, [True])
