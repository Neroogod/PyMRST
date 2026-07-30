from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import (
    cart_grid,
    remove_cells,
    split_disconnected_grid,
)


def _disconnected_strip():
    return remove_cells(cart_grid([5, 1, 1]), [1])


def test_nnc_can_connect_otherwise_disconnected_grid_dict():
    grid = _disconnected_strip()
    grid.nnc = {"cells": np.array([[0, 3]]), "trans": np.array([7.0])}
    parts = split_disconnected_grid(grid)
    assert len(parts) == 1
    assert parts[0] is grid


def test_nnc_can_connect_otherwise_disconnected_grid_namespace():
    grid = _disconnected_strip()
    grid.nnc = SimpleNamespace(cells=np.array([[0, 3]]), trans=np.array([7.0]))
    parts = split_disconnected_grid(grid)
    assert len(parts) == 1
    assert parts[0] is grid


def test_split_components_retain_only_local_remapped_nncs():
    grid = remove_cells(cart_grid([6, 1, 1]), [2])
    # Components are old cells [0,1] and [3,4,5]. These NNCs stay within
    # their respective components and therefore do not reconnect the pieces.
    grid.nnc = {
        "cells": np.array([[0, 1], [2, 4]]),
        "trans": np.array([11.0, 22.0]),
    }
    parts = split_disconnected_grid(grid)
    assert [part.cells.num for part in parts] == [3, 2]
    assert np.array_equal(parts[0].nnc["cells"], [[0, 2]])
    assert np.array_equal(parts[0].nnc["trans"], [22.0])
    assert np.array_equal(parts[1].nnc["cells"], [[0, 1]])
    assert np.array_equal(parts[1].nnc["trans"], [11.0])
    assert np.array_equal(parts[0].cells.global_, [2, 3, 4])
    assert np.array_equal(parts[1].cells.global_, [0, 1])


def test_verbose_option_is_accepted(capsys):
    parts = split_disconnected_grid(_disconnected_strip(), verbose=True)
    assert len(parts) == 2
    assert "2 disconnected components" in capsys.readouterr().out
