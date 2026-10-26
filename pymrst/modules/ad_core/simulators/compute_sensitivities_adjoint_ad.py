"""MRST-style parameter sensitivities from the discrete adjoint sweep.

The current flow adjoint differentiates schedule well-control targets.  This
module adds MRST's parameter grouping surface: one parameter can select wells
and one or more control steps, with a single value shared across those steps.
Initial-state (`state0`) parameters are supported too: their derivative is a
by-product of the same backward sweep, so it costs one extra
transpose-multiply rather than another simulation. Rock parameters are not -
they need the residual differentiated with respect to porosity and
permeability, which means seeding those as AD variables inside the model's
assembly, and that is a change to the model rather than to this module.
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


def _state0_gradient(result: AdjointResult,
                     parameter: AdjointParameter) -> np.ndarray:
    """dJ/dx_0 restricted to the cells this parameter selects.

    The initial-state sensitivity comes out of the same backward sweep as the
    control gradients - it is the first step's multiplier pushed back through
    that step's derivative with respect to the state it started from - so this
    costs nothing beyond the sweep already run.

    `subset` selects entries of the *state vector*, which for a two-phase
    model is `[pressure; sW; well unknowns]`. A parameter naming initial
    pressure therefore passes the pressure block's indices; there is no
    field-name lookup here because the state layout belongs to the model, not
    to this function, and inventing one would guess it.
    """
    gradient = result.initial_state_gradient
    if gradient is None:
        raise ValueError(
            "the adjoint sweep produced no initial-state gradient; a schedule "
            "with no timesteps has nothing to differentiate")
    subset = _get(parameter, "subset")
    if subset is None:
        return np.asarray(gradient, dtype=float)
    indices = np.asarray(subset, dtype=int)
    if indices.size and (indices.min() < 0 or indices.max() >= gradient.size):
        raise IndexError(
            f"parameter subset selects entries outside the {gradient.size}-"
            f"element state vector")
    return np.asarray(gradient, dtype=float)[indices]


def _parameter_gradient(
    full: np.ndarray, parameter: AdjointParameter,
    well_types: Sequence[str], result: AdjointResult | None = None
) -> np.ndarray:
    belongs = str(_get(parameter, "belongs_to",
                       _get(parameter, "belongsTo", "schedule"))).lower()
    if belongs == "state0":
        if result is None:
            raise ValueError("a state0 parameter needs the adjoint result")
        return _state0_gradient(result, parameter)
    if belongs != "schedule":
        # Rock parameters - porosity, permeability, transmissibility - need
        # the *residual* differentiated with respect to them, which means
        # seeding them as AD variables inside `assemble`. That is a change to
        # the model, not to this function, and no amount of post-processing
        # the control gradient can substitute for it.
        #
        # `state0` is different and is handled above: its derivative is
        # already a by-product of the backward sweep, which is why one is
        # available and the other is not.
        raise NotImplementedError(
            f"adjoint seeding for {belongs!r} parameters needs the residual "
            f"differentiated with respect to them - rock properties must be "
            f"AD variables in the model's assemble() for that. 'schedule' and "
            f"'state0' parameters are supported.")
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
            result.gradient, parameter, well_types, result
        )
    return sensitivities


computeSensitivitiesAdjointAD = compute_sensitivities_adjoint_ad

__all__ = [
    "AdjointParameter",
    "compute_sensitivities_adjoint_ad",
    "computeSensitivitiesAdjointAD",
]
