"""MRST ``pinchedNode`` fixture."""

import numpy as np


def pinched_node():
    coord = np.array([
        [0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1],
        [2, 0, 0, 2, 0, 1], [0, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1], [2, 1, 0, 2, 1, 1],
    ], dtype=float)
    zcorn = np.array([
        [0, 0, 0, 1], [0, 0, 0, 0],
        [1, 1, 1, 1], [1, 1, 1, 1],
    ], dtype=float)
    return {
        "cartDims": np.array([2, 1, 1]),
        "COORD": coord.ravel(),
        "ZCORN": zcorn.ravel(),
        "ACTNUM": np.ones(2, dtype=np.int32),
    }


pinchedNode = pinched_node
