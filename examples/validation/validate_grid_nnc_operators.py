"""Validate TPFA explicit-NNC operators against MATLAB/MRST."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params import make_rock


HERE = Path(__file__).resolve().parent


def main():
    ref = loadmat(HERE / "ref_grid_nnc_operators.mat", squeeze_me=True)
    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    grid.nnc = {"cells": np.array([[0, 2]]), "trans": np.array([7.5])}
    operators = setup_operators_tpfa(
        grid, make_rock(grid, np.ones((grid.cells.num, 1)), 0.2)
    )
    checks = {
        "neighbors": (operators.N + 1, ref["neighbors"]),
        "transmissibility": (operators.T, ref["transmissibility"]),
        "gradient": (
            operators.grad(np.array([1.0, 4.0, 9.0])), ref["gradient"],
        ),
        "divergence": (
            operators.div(np.array([0.0, 0.0, 2.0])), ref["divergence"],
        ),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
        passed = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-13, atol=1e-13
        )
        print(f"{'ok  ' if passed else 'FAIL'} {name}")
        failed |= not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
