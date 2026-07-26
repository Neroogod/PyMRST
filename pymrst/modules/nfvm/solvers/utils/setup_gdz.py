"""Gravity differences for an installed NFVM gradient (MRST ``setup_gdz.m``)."""

import numpy as np


def setup_gdz(model, grad_op, gravity_vector=None):
    """Evaluate ``grad_op`` on coordinates and contract with gravity."""
    dim = model.G.griddim
    if gravity_vector is None:
        gravity_vector = np.zeros(dim)
        gravity_vector[-1] = float(getattr(model, "gravity_magnitude", 0.0))
    gravity_vector = np.asarray(gravity_vector, dtype=float).ravel()
    if gravity_vector.size != dim:
        raise ValueError("gravity vector dimension does not match grid")
    coordinate_gradients = np.column_stack([
        grad_op(np.asarray(model.G.cells.centroids)[:, axis])
        for axis in range(dim)
    ])
    return coordinate_gradients @ gravity_vector


setup_gdz.__name__ = "setup_gdz"
setupGDZ = setup_gdz

__all__ = ["setup_gdz", "setupGDZ"]
