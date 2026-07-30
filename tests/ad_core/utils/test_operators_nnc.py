import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params import make_rock


def test_setup_operators_tpfa_includes_explicit_nnc():
    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    grid.nnc = {
        "cells": np.array([[0, 2]], dtype=np.int64),
        "trans": np.array([7.5]),
    }
    rock = make_rock(grid, np.ones((3, 1)), 0.2)
    operators = setup_operators_tpfa(grid, rock)
    assert np.array_equal(operators.N[-1], [0, 2])
    assert operators.T[-1] == 7.5
    assert np.array_equal(operators.grad(np.array([1.0, 4.0, 9.0])),
                          [3.0, 5.0, 8.0])
    assert np.array_equal(operators.div(np.array([0.0, 0.0, 2.0])),
                          [2.0, 0.0, -2.0])
