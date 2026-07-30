"""MRST ``createPinchedColumn`` fixture."""

import numpy as np


def create_pinched_column():
    x, y, z = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0], indexing="ij")
    points = np.column_stack([
        x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")
    ])
    coord = np.column_stack([points, points]).ravel()
    d = 1.01
    zcorn = np.array([
        [0, 0], [0, 0], [1, 1], [1, 1],
        [1, 1], [1, 1], [d, d], [d, d],
        [d, d], [d, d], [2, 2], [2, 2],
    ], dtype=float)
    return {
        "cartDims": np.array([1, 1, 3]),
        "PINCH": [0.02, "GAP", np.inf, "TOPBOT", "TOP"],
        "COORD": coord,
        "ZCORN": zcorn.T.ravel(order="F"),
        "ACTNUM": np.array([1, 0, 1], dtype=np.int32),
    }


createPinchedColumn = create_pinched_column
