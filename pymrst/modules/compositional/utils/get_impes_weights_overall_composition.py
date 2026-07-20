"""IMPES pressure weights for overall-composition models.

This is the PyMRST port of MRST's
``getImpesWeightsOverallComposition.m``.  The utility exposes the same
numerical pressure-weight construction used internally by
``OverallCompositionModel.pressure_step``: assemble component accumulation terms
with AD variables and solve the transposed accumulation Jacobian for the
pressure-equation weights.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.utils.adi import ADI, init_variables_adi


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _call_update_after_convergence(model: Any, state: Any, dt: float):
    """Mirror MRST's EOSModel.updateAfterConvergence hook when present."""
    eos = _get(model, "EOSModel", None) or _get(model, "eos_model", None)
    hook = _get(eos, "updateAfterConvergence", None) or _get(eos, "update_after_convergence", None)
    if hook is None:
        return state
    try:
        return hook(state, state, dt, {})
    except TypeError:
        try:
            return hook(state, dt)
        except TypeError:
            return hook(state)


def _stack_weights(weights: Sequence[np.ndarray]) -> np.ndarray:
    if not weights:
        return np.zeros((0, 0), dtype=float)
    return np.column_stack([np.asarray(w, dtype=float).reshape(-1) for w in weights])


def _num_wells(model: Any, state: Any) -> int:
    wells = _get(model, "wells", ())
    if wells is not None:
        return len(wells)
    q_os = _get(state, "qOs", None)
    return 0 if q_os is None else np.asarray(q_os).size


def _state_array(state: Any, name: str, fallback, size: int | None = None) -> np.ndarray:
    value = _get(state, name, None)
    if value is None:
        if np.isscalar(fallback):
            if size is None:
                return np.asarray(fallback, dtype=float).reshape(-1)
            return np.full(size, fallback, dtype=float)
        return np.asarray(fallback, dtype=float)
    return np.asarray(value, dtype=float)


def _assemble_accumulation_terms(model: Any, state: Any, dt: float, *, bc=None, src=None):
    if not hasattr(model, "assemble"):
        raise TypeError("model must provide assemble(...) or pass acc=... explicitly")
    pressure = np.asarray(_get(state, "pressure"), dtype=float).reshape(-1)
    z = np.asarray(_get(state, "z", _get(state, "components", None)), dtype=float)
    if z.ndim != 2:
        raise ValueError("state must provide z/components as an (ncell, ncomp) array")
    ncomp = z.shape[1]
    nw = _num_wells(model, state)
    values = [pressure, *[z[:, i] for i in range(ncomp - 1)]]
    water = bool(_get(model, "water", False))
    if water:
        s_w = _state_array(state, "sW", 0.0, pressure.size)
        values.append(s_w)
        values.append(_state_array(state, "qWs", 0.0, nw))
    values.extend([
        _state_array(state, "qOs", 0.0, nw),
        _state_array(state, "qGs", 0.0, nw),
        _state_array(state, "bhp", 0.0, nw),
    ])
    prim = init_variables_adi(*values)
    idx = 1 + ncomp - 1
    sw_ad = prim[idx] if water else None
    idx += int(water)
    qw_ad = prim[idx] if water else None
    idx += int(water)
    qo, qg, bhp = prim[idx:idx + 3]
    eqs, _well_eqs, _ctrl, weights = model.assemble(
        state,
        prim[0],
        list(prim[1:1 + ncomp - 1]),
        qo,
        qg,
        bhp,
        dt,
        sw_ad,
        qw_ad,
        pressure_reduction=True,
        return_pressure_weights=True,
        bc=bc,
        src=src,
    )
    _ = eqs
    return weights


def getImpesWeightsOverallComposition(model, state, dt, acc=None, **kwargs) -> np.ndarray:
    """Return MRST-style IMPES pressure weights as ``(ncell, neq)``.

    Parameters mirror MRST's ``getImpesWeightsOverallComposition(model, state,
    dt)``.  If ``acc`` is supplied, it must be the list of AD accumulation terms
    and the function delegates directly to ``model.partial_volume_weights``.
    Otherwise the current state is assembled with AD primary variables and the
    existing PyMRST accumulation-Jacobian solver is used through
    ``assemble(..., pressure_reduction=True)``.
    """
    bc = kwargs.pop("bc", None)
    src = kwargs.pop("src", None)
    update_after_convergence = kwargs.pop("updateAfterConvergence", kwargs.pop("update_after_convergence", True))
    as_list = kwargs.pop("asList", kwargs.pop("as_list", False))
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected option(s): {unknown}")
    if acc is not None:
        if not hasattr(model, "partial_volume_weights"):
            raise TypeError("model must provide partial_volume_weights(acc) when acc is supplied")
        weights = model.partial_volume_weights(acc)
    else:
        work_state = _call_update_after_convergence(model, state, dt) if update_after_convergence else state
        weights = _assemble_accumulation_terms(model, work_state, float(dt), bc=bc, src=src)
    if as_list:
        return [np.asarray(w, dtype=float).reshape(-1) for w in weights]
    return _stack_weights(weights)


def get_impes_weights_overall_composition(model, state, dt, acc=None, **kwargs) -> np.ndarray:
    """Pythonic alias for :func:`getImpesWeightsOverallComposition`."""
    return getImpesWeightsOverallComposition(model, state, dt, acc=acc, **kwargs)


__all__ = ["getImpesWeightsOverallComposition", "get_impes_weights_overall_composition"]
