"""Extract a nodal uplift component from a coupled mechanics state."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_uplift_for_state(
    model_or_state: Any,
    state_or_node: Any,
    topnode: int | None = None,
    *,
    direction: int | None = None,
    compute_partials: bool = False,
    one_based: bool = False,
):
    """Return uplift while retaining AD derivatives when ``xd`` is seeded.

    The preferred call is ``compute_uplift_for_state(model, state, node)``.
    The historical ``(state, node, direction=2)`` form remains supported.
    """
    if hasattr(model_or_state, "G"):
        model, state, node = model_or_state, state_or_node, topnode
        if node is None:
            raise TypeError("topnode is required")
        dim = int(model.G.griddim)
        node = int(node) - int(one_based)
        direction = dim - 1 if direction is None else int(direction)
        if not 0 <= node < int(model.G.nodes.num):
            raise IndexError("topnode is outside the grid")
        if not 0 <= direction < dim:
            raise IndexError("direction is outside the grid dimension")

        mech = getattr(model, "mech", None) or getattr(model, "mechModel", model)
        operator = mech.operators
        is_dir = getattr(operator, "is_dir", None)
        if is_dir is None:
            is_dir = getattr(operator, "isdirdofs")
        full_index = dim * node + direction
        if is_dir[full_index]:
            prescribed = getattr(operator, "u_dir", None)
            if prescribed is None:
                prescribed = getattr(operator, "V_dir")
            return np.asarray(prescribed)[full_index]
        free = np.flatnonzero(~np.asarray(is_dir, dtype=bool))
        reduced_index = int(np.searchsorted(free, full_index))
        if free[reduced_index] != full_index:
            raise RuntimeError("failed to map uplift degree of freedom")
        return state["xd"][reduced_index]

    state, node = model_or_state, int(state_or_node)
    dim = 3
    direction = 2 if direction is None else int(direction)
    return np.asarray(state["u"]).reshape(-1, dim)[node, direction]


computeUpliftForState = compute_uplift_for_state

__all__ = ["compute_uplift_for_state", "computeUpliftForState"]
