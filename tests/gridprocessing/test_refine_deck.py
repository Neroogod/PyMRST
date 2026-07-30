from copy import deepcopy

import numpy as np

from pymrst.gridprocessing import refine_deck


def _grid(nx=2, ny=1, nz=2):
    xyz = np.zeros((6, nx + 1, ny + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            xyz[:, i, j] = [i, j, 0.0, i, j, nz]
    z = np.empty((2 * nx, 2 * ny, 2 * nz))
    for k in range(nz):
        z[:, :, 2 * k] = k
        z[:, :, 2 * k + 1] = k + 1
    return {
        "cartDims": np.array([nx, ny, nz]),
        "COORD": xyz.ravel(order="F"),
        "ZCORN": z.ravel(order="F"),
        "PORO": np.arange(1, nx * ny * nz + 1, dtype=float) / 10,
    }


def _deck():
    return {
        "RUNSPEC": {"DIMENS": np.array([2, 1, 2])},
        "GRID": _grid(),
        "SOLUTION": {
            "PRESSURE": np.array([100.0, 200.0, 300.0, 400.0]),
            "SWAT": np.array([0.1, 0.2, 0.3, 0.4]),
        },
        "REGIONS": {"SATNUM": np.array([1, 2, 3, 4])},
        "PROPS": {"SWL": np.array([0.01, 0.02, 0.03, 0.04])},
        "SCHEDULE": {"control": [{
            "WELSPECS": np.array([
                ["WZ", "G", 1, 1],
                ["WX", "G", 2, 1],
            ], dtype=object),
            "COMPDAT": np.array([
                ["WZ", 1, 1, 1, 2, "OPEN", 0, 12.0, 0, 18.0, 0, 0, "Z"],
                ["WX", 2, 1, 1, 1, "OPEN", 0, 20.0, 0, 30.0, 0, 0, "X"],
            ], dtype=object),
        }]},
    }


def test_refine_deck_expands_grid_and_cell_fields_without_mutating_input():
    deck = _deck()
    original = deepcopy(deck)
    refined = refine_deck(deck, [2, 3, 2])
    assert refined["RUNSPEC"]["DIMENS"].tolist() == [4, 3, 4]
    assert refined["GRID"]["cartDims"].tolist() == [4, 3, 4]
    assert refined["SOLUTION"]["PRESSURE"].size == 48
    pressure = refined["SOLUTION"]["PRESSURE"].reshape((4, 3, 4), order="F")
    assert np.all(pressure[:2, :, :2] == 100.0)
    assert np.array_equal(deck["RUNSPEC"]["DIMENS"], original["RUNSPEC"]["DIMENS"])


def test_refine_deck_updates_vertical_and_horizontal_completions():
    refined = refine_deck(_deck(), [2, 3, 2])
    control = refined["SCHEDULE"]["control"][0]
    wells = control["WELSPECS"]
    compdat = control["COMPDAT"]
    assert wells[:, 2:4].tolist() == [[1, 2], [3, 2]]
    # Vertical completion expands through both child layers.
    assert compdat[0, 1:5].tolist() == [1, 2, 1, 4]
    # X completion repeats twice and WI/KH are divided by x refinement.
    assert compdat[1:3, 1].tolist() == [3, 4]
    assert compdat[1:3, 3:5].tolist() == [[1, 1], [1, 1]]
    assert np.allclose(compdat[1:3, 7].astype(float), 10.0)
    assert np.allclose(compdat[1:3, 9].astype(float), 15.0)


def test_default_well_resets_transmissibility_and_kh():
    refined = refine_deck(_deck(), [2, 1, 1], default_well=True)
    compdat = refined["SCHEDULE"]["control"][0]["COMPDAT"]
    assert np.all(compdat[:, 7] == "-1")
    assert np.all(compdat[:, 9] == "-1")
