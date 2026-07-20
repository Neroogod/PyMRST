"""Compatibility helper for MRST ``objUpliftAver.m``."""

import numpy as np


def uplift_average_objective(nodes, direction: int = 2):
    """Return an objective that averages uplift over a set of surface nodes."""
    nodes = np.asarray(nodes, dtype=np.int64).ravel()

    def objective(state, dt: float = 1.0) -> float:
        u = np.asarray(state["u"], dtype=float).reshape(-1, 3)
        # z points down in MRST/PyMRST grids, so uplift is negative z motion.
        return float(-np.mean(u[nodes, direction]) * dt)

    return objective


def obj_uplift_aver(state, nodes, direction: int = 2, dt: float = 1.0) -> float:
    """Evaluate the averaged uplift objective directly for one state."""
    return uplift_average_objective(nodes, direction)(state, dt)


__all__ = ["uplift_average_objective", "obj_uplift_aver"]
