"""MRST-style parameter sensitivities from the discrete adjoint sweep.

The current flow adjoint differentiates schedule well-control targets.  This
module adds MRST's parameter grouping surface: one parameter can select wells
and one or more control steps, with a single value shared across those steps.
Rock and initial-state parameters require residual parameter seeding and are
rejected explicitly until that backend support is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pymrst.modules.ad_core.simulators.compute_gradient_adjoint_ad import (
    AdjointResult,
    compute_gradient_adjoint,
)


@dataclass
class AdjointParameter:
    """A schedule-control parameter compatible with MRST ``ModelParameter``."""

    name: str
    control_type: str
    subset: Sequence[int] | None = None
    control_steps: Sequence[int] | None = None
    belongs_to: str = "schedule"
    lumping: Sequence[int] | None = None


def _get(value: Any, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _parameter_gradient(
    full: np.ndarray, parameter: AdjointParameter, well_types: Sequence[str]
) -> np.ndarray:
    belongs = _get(parameter, "belongs_to",
                   _get(parameter, "belongsTo", "schedule"))
    if str(belongs).lower() != "schedule":
        raise NotImplementedError(
            "adjoint seeding for model/state0 parameters is not yet available"
        )
    control_type = str(_get(parameter, "control_type",
                            _get(parameter, "controlType", ""))).lower()
    if control_type not in {"bhp", "rate", "wrat", "orat", "grat"}:
        raise ValueError(f"unsupported control parameter {control_type!r}")

    nw = full.shape[1]
    subset = _get(parameter, "subset")
    wells = np.arange(nw) if subset is None else np.asarray(subset, dtype=int)
    steps_value = _get(parameter, "control_steps",
                       _get(parameter, "controlSteps"))
    steps = (np.arange(full.shape[0]) if steps_value is None
             else np.asarray(steps_value, dtype=int))
    if wells.size and (wells.min() < 0 or wells.max() >= nw):
        raise IndexError("parameter well subset is outside the control matrix")
    if steps.size and (steps.min() < 0 or steps.max() >= full.shape[0]):
        raise IndexError("parameter control steps are outside the schedule")

    out = np.zeros(wells.size)
    for local, well in enumerate(wells):
        if str(well_types[well]).lower() == control_type:
            out[local] = np.sum(full[np.ix_(steps, [well])])

    lumping = _get(parameter, "lumping")
    if lumping is not None:
        groups = np.asarray(lumping, dtype=int)
        if groups.size != out.size:
            raise ValueError("lumping must have one entry per selected well")
        labels = np.unique(groups)
        out = np.array([out[groups == label].sum() for label in labels])
    return out


def compute_sensitivities_adjoint_ad(
    setup: Any,
    states: Sequence[Any] | None,
    parameters: Iterable[AdjointParameter],
    get_objective,
) -> dict[str, np.ndarray]:
    """Compute sensitivities for MRST-style schedule control parameters.

    ``setup`` must provide ``model``, ``state0``, ``dts`` and ``controls``.
    A parameter shared across several control steps receives the sum of those
    step derivatives, matching MRST's parameter-setting semantics.
    """
    model = _get(setup, "model")
    state0 = _get(setup, "state0")
    dts = _get(setup, "dts")
    controls = np.asarray(_get(setup, "controls"), dtype=float)
    if model is None or state0 is None or dts is None:
        raise ValueError("setup must provide model, state0, dts and controls")

    result: AdjointResult = compute_gradient_adjoint(
        model, state0, dts, controls, get_objective, states=states
    )
    well_types = [str(w.type) for w in model.wells]
    sensitivities: dict[str, np.ndarray] = {}
    for parameter in parameters:
        name = str(_get(parameter, "name"))
        if name in sensitivities:
            raise ValueError(f"duplicate sensitivity parameter name {name!r}")
        sensitivities[name] = _parameter_gradient(
            result.gradient, parameter, well_types
        )
    return sensitivities


computeSensitivitiesAdjointAD = compute_sensitivities_adjoint_ad

__all__ = [
    "AdjointParameter",
    "compute_sensitivities_adjoint_ad",
    "computeSensitivitiesAdjointAD",
]
