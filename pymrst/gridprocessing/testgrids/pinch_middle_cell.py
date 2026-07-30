"""MRST ``pinchMiddleCell`` family of corner-point fixtures."""

from itertools import combinations

import numpy as np


def _coord():
    x, y, z = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0], indexing="ij")
    points = np.column_stack([
        x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")
    ])
    return np.column_stack([points, points]).ravel()


def pinch_middle_cell(k=1):
    """Return all ways to pinch ``k`` of the middle cell's four corners."""
    if int(k) != k or not 0 <= k <= 4:
        raise ValueError("k must be an integer between zero and four")
    grids = []
    for selected in combinations(range(4), int(k)):
        bound = np.ones(4)
        bound[list(selected)] = 2.0
        zcorn = np.concatenate([
            np.zeros(4), bound, bound, np.full(4, 2.0),
            np.full(4, 2.0), np.full(4, 3.0),
        ])
        grids.append({
            "cartDims": np.array([1, 1, 3]),
            "COORD": _coord(),
            "ZCORN": zcorn,
        })
    return grids


pinchMiddleCell = pinch_middle_cell
