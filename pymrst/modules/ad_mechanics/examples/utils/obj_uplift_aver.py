"""Time-distributed positive-uplift objective for adjoint optimization."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.modules.ad_mechanics.examples.utils.compute_uplift_for_state import compute_uplift_for_state
from pymrst.modules.ad_mechanics.examples.utils.obj_uplift import _timesteps
from pymrst.utils.adi import ad_exp, ad_log, ad_max


def uplift_average_objective(nodes, direction: int = 2):
    """Legacy objective factory averaging upward motion over several nodes."""
    nodes = np.asarray(nodes, dtype=np.int64).ravel()

    def objective(state, dt: float = 1.0) -> float:
        u = np.asarray(state["u"], dtype=float).reshape(-1, 3)
        return float(-np.mean(u[nodes, direction]) * dt)

    return objective


def obj_uplift_aver(
    model_or_state: Any,
    states_or_nodes: Any,
    schedule: Any = None,
    topnode: int | None = None,
    *,
    compute_all_steps: bool = True,
    t_step: int | None = None,
    state: Any = None,
    compute_partials: bool = False,
    exponent: float = 1.0,
    normalization_constant: float = 1.0,
    direction: int | None = None,
    dt: float = 1.0,
):
    """Evaluate MRST ``objUpliftAver`` or the legacy direct objective."""
    if not hasattr(model_or_state, "G"):
        return uplift_average_objective(states_or_nodes, 2 if direction is None else direction)(model_or_state, dt)

    model, states = model_or_state, states_or_nodes
    if schedule is None or topnode is None:
        raise TypeError("schedule and topnode are required")
    steps = _timesteps(schedule)
    if compute_all_steps:
        selected = list(states)
        indices = np.arange(len(selected))
        if len(selected) != steps.size:
            raise ValueError("states and schedule timesteps must have equal length")
    else:
        if t_step is None:
            raise ValueError("t_step is required for a single objective step")
        index = int(t_step)
        if index == steps.size:
            index -= 1
        selected, indices = [state if state is not None else states[0]], np.array([index])

    p = float(exponent)
    C = float(normalization_constant)
    out = []
    for current, index in zip(selected, indices):
        uplift = compute_uplift_for_state(
            model, current, topnode, direction=direction,
            compute_partials=compute_partials,
        )
        positive = ad_max(0.0, uplift)
        value = ad_exp(p * ad_log(np.finfo(float).eps + C * positive))
        out.append(value * steps[index])
    return out


objUpliftAver = obj_uplift_aver

__all__ = ["uplift_average_objective", "obj_uplift_aver", "objUpliftAver"]
