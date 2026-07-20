"""MRST ``C2D.m`` strain-energy tensor scaling."""

import numpy as np


def c2d(C: np.ndarray) -> np.ndarray:
    """Scale Voigt tensor entries to strain-energy form.

    MRST stores elasticity tensors in Voigt form and then applies the shear
    scaling used by the VEM strain-energy product. This supports both 2D
    ``[xx, yy, xy]`` tensors flattened to 9 entries and 3D tensors flattened
    to 36 entries.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.shape[1] == 9:
        factor = np.ones((3, 3))
        factor[:2, 2:] = 2.0
        factor[2:, :2] = 2.0
        factor[2:, 2:] = 4.0
        return C * factor.reshape(1, 9, order="F")

    factor = np.ones((6, 6))
    factor[:3, 3:] = 2.0
    factor[3:, :3] = 2.0
    factor[3:, 3:] = 4.0
    return C * factor.reshape(1, 36, order="F")

__all__ = ["c2d"]
