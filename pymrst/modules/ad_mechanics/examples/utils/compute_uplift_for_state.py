"""Compatibility landing file for MRST ``computeUpliftForState.m`` subset."""

import numpy as np


def compute_uplift_for_state(state, node: int, direction: int = 2) -> float:
    """Return displacement component at one node from a state with full ``u``."""
    return float(np.asarray(state["u"]).reshape(-1, 3)[node, direction])


__all__ = ["compute_uplift_for_state"]
