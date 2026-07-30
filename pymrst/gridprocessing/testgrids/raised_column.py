"""MRST ``raisedColumn`` fixture."""

import numpy as np


def raised_column():
    x, y, z = np.meshgrid(np.arange(3.0), np.arange(2.0), [0.0], indexing="ij")
    points = np.column_stack([
        x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")
    ])
    coord = np.column_stack([points, points]).ravel()
    zcorn = np.array([
        [0, 0, 0, 0], [0, 0, 0, 0],
        [1, 1, .5, .5], [1, 1, .5, .5],
        [1, 1, .5, .5], [1, 1, .5, .5],
        [2, 2, 1.5, 1.5], [2, 2, 1.5, 1.5],
    ], dtype=float)
    return {
        "cartDims": np.array([2, 1, 2]),
        "COORD": coord,
        "ZCORN": zcorn.ravel(),
    }


raisedColumn = raised_column
