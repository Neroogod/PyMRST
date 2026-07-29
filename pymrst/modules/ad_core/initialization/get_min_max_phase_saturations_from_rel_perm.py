"""MRST ``getMinMaxPhaseSaturationsFromRelPerm``."""

from __future__ import annotations

import numpy as np


def _phase_index(model, phase: str) -> int:
    if hasattr(model, "get_phase_index"):
        return int(model.get_phase_index(phase))
    if hasattr(model, "getPhaseIndex"):
        index = int(model.getPhaseIndex(phase))
        return index - 1 if index >= 1 else index
    order = [name for name, active in (
        ("W", getattr(model, "water", False)),
        ("O", getattr(model, "oil", False)),
        ("G", getattr(model, "gas", False)),
    ) if active]
    return order.index(phase)


def _regional_function(function, region: int):
    if isinstance(function, (list, tuple)):
        return function[int(region) - 1]
    return function


def _evaluate(model, name: str, saturation, region: int):
    fluid = model.fluid
    function = getattr(fluid, name, None)
    if function is None and isinstance(fluid, dict):
        function = fluid.get(name)
    if function is None:
        raise AttributeError(f"fluid has no relative-permeability function {name}")
    return np.asarray(_regional_function(function, region)(saturation), dtype=float)


def _endpoints(saturation, relperm):
    zero = np.flatnonzero(relperm == 0)
    plateau = np.flatnonzero(relperm == relperm[-1])
    s_min = saturation[zero[-1]] if zero.size else saturation[0]
    s_max = saturation[plateau[0]] if plateau.size else saturation[-1]
    return s_min, s_max


def get_min_max_phase_saturations_from_rel_perm(
    model, tol: float = 1e-6, region: int = 1
):
    """Infer phase saturation endpoints by sampling relative permeability."""
    if tol <= 0 or tol > 1:
        raise ValueError("tol must be in (0, 1]")
    saturation = np.arange(0.0, 1.0 + 0.5 * tol, tol)
    active = [
        phase for phase in ("W", "O", "G")
        if bool(getattr(model, {"W": "water", "O": "oil", "G": "gas"}[phase], False))
    ]
    s_min = np.zeros(len(active))
    s_max = np.ones(len(active))

    if "W" in active:
        ix = _phase_index(model, "W")
        s_min[ix], s_max[ix] = _endpoints(
            saturation, _evaluate(model, "krW", saturation, region)
        )
    if "O" in active:
        ix = _phase_index(model, "O")
        try:
            kro = _evaluate(model, "krO", saturation, region)
        except AttributeError:
            krow = (
                _evaluate(model, "krOW", saturation, region)
                if "W" in active else np.ones_like(saturation)
            )
            krog = (
                _evaluate(model, "krOG", saturation, region)
                if "G" in active else np.ones_like(saturation)
            )
            kro = np.minimum(krow, krog)
        s_min[ix], s_max[ix] = _endpoints(saturation, kro)
    if "G" in active:
        ix = _phase_index(model, "G")
        s_min[ix], s_max[ix] = _endpoints(
            saturation, _evaluate(model, "krG", saturation, region)
        )
    return s_min, s_max


getMinMaxPhaseSaturationsFromRelPerm = (
    get_min_max_phase_saturations_from_rel_perm
)

__all__ = [
    "get_min_max_phase_saturations_from_rel_perm",
    "getMinMaxPhaseSaturationsFromRelPerm",
]
