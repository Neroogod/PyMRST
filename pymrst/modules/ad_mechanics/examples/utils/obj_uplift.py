"""Time-weighted surface-uplift objective used by the adjoint example."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.modules.ad_mechanics.adjoint_impl import uplift_objective
from pymrst.modules.ad_mechanics.examples.utils.compute_uplift_for_state import compute_uplift_for_state


def _timesteps(schedule: Any) -> np.ndarray:
    if isinstance(schedule, dict):
        step = schedule.get("step", schedule)
        values = step.get("val") if isinstance(step, dict) else getattr(step, "val")
    else:
        step = getattr(schedule, "step", schedule)
        values = getattr(step, "val", step)
    return np.asarray(values, dtype=float).ravel()


def _node_displacement(state: Any, node: int, direction: int, dim: int):
    u = state["u"]
    index = int(node) * dim + int(direction)
    try:
        return u[index]
    except (TypeError, IndexError):
        return np.asarray(u).reshape(-1, dim)[node, direction]


def obj_uplift(
    model: Any,
    states: Sequence[Any],
    schedule: Any,
    topnode: int,
    *,
    compute_all_steps: bool = True,
    t_step: int | None = None,
    state: Any | None = None,
    compute_partials: bool = False,
    direction: int | None = None,
) -> list[Any]:
    """Evaluate MRST ``objUplift`` for all steps or one adjoint step."""
    dt = _timesteps(schedule)
    dim = int(model.G.griddim)
    direction = dim - 1 if direction is None else int(direction)
    if not 0 <= direction < dim:
        raise ValueError("uplift direction is outside the grid dimension")

    if compute_all_steps:
        selected_states = list(states)
        indices = np.arange(len(selected_states), dtype=int)
        if len(selected_states) != dt.size:
            raise ValueError("states and schedule timesteps must have equal length")
    else:
        if t_step is None:
            raise ValueError("t_step is required when compute_all_steps is false")
        index = int(t_step)
        # Accept Python zero-based indices and MRST-style one-based final index.
        if index == dt.size:
            index -= 1
        if not 0 <= index < dt.size:
            raise IndexError("t_step is outside the schedule")
        selected_states = [state if state is not None else states[0]]
        indices = np.array([index])

    if hasattr(model, "mech") or hasattr(model, "mechModel"):
        return [
            compute_uplift_for_state(
                model, st, topnode, direction=direction,
                compute_partials=compute_partials,
            ) * dt[index]
            for st, index in zip(selected_states, indices)
        ]
    # Historical lightweight form used full displacement with positive depth.
    return [-_node_displacement(st, topnode, direction, dim) * dt[index]
            for st, index in zip(selected_states, indices)]


objUplift = obj_uplift

__all__ = ["obj_uplift", "objUplift", "uplift_objective"]
