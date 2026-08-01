"""Runnable counterpart of MRST ``equil/test_simpleEquilibrium.m``."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.utils.equil.simple_equilibrium import simple_equilibrium


def test_simple_equilibrium_case():
    grid = compute_geometry(cart_grid([10, 1, 10], [1.0, 1.0, 1.0]))
    contacts = np.array([0.37, 0.8])
    vertical = simple_equilibrium(grid, contacts)
    diagonal = simple_equilibrium(grid, contacts, [1.0, 0.0, 1.0])
    assert vertical.shape == diagonal.shape == (100, 3)
    assert np.allclose(vertical.sum(axis=1), 1.0)
    assert np.allclose(diagonal.sum(axis=1), 1.0)
    assert not np.allclose(vertical, diagonal)
    return {"grid": grid, "contacts": contacts, "vertical": vertical, "diagonal": diagonal}


if __name__ == "__main__":
    test_simple_equilibrium_case()
    print("simpleEquilibrium example passed")
