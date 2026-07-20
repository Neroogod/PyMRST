"""Construct EOS component objects (MRST ``getEOSComponent.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.compositional.properties.components import EquationOfStateComponent
from pymrst.modules.compositional.utils.standalone_flash import standaloneFlash


_PHASE_ALIASES = {
    "water": ("water", "w", "W"),
    "oil": ("oil", "o", "O", "liquid", "L"),
    "gas": ("gas", "g", "G", "vapour", "vapor", "V"),
}


def _eos_model(model: Any):
    eos = getattr(model, "EOSModel", None)
    if eos is None and isinstance(model, dict):
        eos = model.get("EOSModel")
    return model if eos is None else eos


def _mixture(eos):
    mix = getattr(eos, "CompositionalMixture", None)
    if mix is None:
        mix = getattr(eos, "mixture", None)
    if mix is None:
        raise ValueError("model/EOSModel must expose CompositionalMixture")
    return mix


def _call_or_attr(obj: Any, camel: str, snake: str | None = None, default=None):
    if obj is None:
        return default
    if hasattr(obj, camel):
        value = getattr(obj, camel)
        return value() if callable(value) else value
    if snake and hasattr(obj, snake):
        value = getattr(obj, snake)
        return value() if callable(value) else value
    if isinstance(obj, dict):
        if camel in obj:
            value = obj[camel]
            return value() if callable(value) else value
        if snake and snake in obj:
            value = obj[snake]
            return value() if callable(value) else value
    return default


def _phase_names(model: Any, nph: int) -> tuple[str, ...]:
    names = _call_or_attr(model, "getPhaseNames", "get_phase_names")
    if names is None:
        names = getattr(model, "phase_names", None)
    if names is None:
        defaults = ("liquid", "vapor", "water", "oil", "gas")
        return defaults[:nph]
    return tuple(str(v) for v in names)


def _extra_phase_names(model: Any) -> tuple[str, ...]:
    extra = _call_or_attr(model, "getNonEoSPhaseNames", "get_non_eos_phase_names", default=())
    if extra is None:
        return ()
    if isinstance(extra, tuple) and len(extra) == 2 and not isinstance(extra[0], str):
        extra = extra[1]
    elif isinstance(extra, list) and len(extra) == 2 and isinstance(extra[0], (list, tuple)):
        extra = extra[1]
    if isinstance(extra, str):
        return (extra,)
    return tuple(str(v) for v in extra)


def _phase_index(model: Any, method: str, snake: str, default: int) -> int:
    value = _call_or_attr(model, method, snake, default=default)
    if value is None:
        return default
    return int(value)


def _number_of_phases(model: Any, liquid_index: int, vapor_index: int, extra: tuple[str, ...]) -> int:
    nph = _call_or_attr(model, "getNumberOfPhases", "get_number_of_phases")
    if nph is None:
        nph = getattr(model, "n_phases", None)
    if nph is None:
        nph = max(liquid_index, vapor_index, 1) + 1 + len(extra)
    return int(nph)


def _density_accessor(fluid: Any, phase: str):
    keys = []
    aliases = (phase,)
    low = phase.lower()
    for canonical, names in _PHASE_ALIASES.items():
        if low in {v.lower() for v in names}:
            aliases = names
            break
    for alias in aliases:
        token = str(alias)
        keys.extend([
            f"rho{token}S",
            f"rho{token.upper()}S",
            f"rho{token.capitalize()}S",
            f"rho_{token.lower()}s",
            f"rho_{token.lower()}_s",
        ])
    for key in keys:
        if isinstance(fluid, dict) and key in fluid:
            return fluid[key]
        if hasattr(fluid, key):
            return getattr(fluid, key)
    return None


def _surface_density(fluid: Any, phase: str) -> float | None:
    value = _density_accessor(fluid, phase)
    if value is None:
        return None
    if callable(value):
        try:
            value = value(1)
        except TypeError:
            value = value()
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else None


def getEOSComponent(model, p, T, name: str, ci: int | None = None, *, mrst_indexing: bool = False) -> EquationOfStateComponent:
    """Return an :class:`EquationOfStateComponent` for one EOS component.

    The calculation follows MRST ``getEOSComponent.m``: flash a pure-component
    stream at separator conditions, convert mole phase fractions to mass phase
    fractions using EOS phase densities, then store per-phase surface fractions
    and pure phase densities in model phase order.

    PyMRST component indices are zero-based. Pass ``mrst_indexing=True`` when
    feeding in MRST's one-based loop index directly.
    """
    eos = _eos_model(model)
    mix = _mixture(eos)
    names = tuple(str(v) for v in mix.names)
    if name not in names:
        raise ValueError(f"component {name!r} not found in EOS mixture {names!r}")
    hcpos = names.index(name)
    component_index = hcpos if ci is None else int(ci) - (1 if mrst_indexing else 0)
    if component_index < 0:
        raise ValueError("component index must be non-negative after indexing conversion")

    z = np.zeros((1, len(names)), dtype=float)
    z[0, hcpos] = 1.0
    flash = standaloneFlash(p, T, z, eos)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    T_arr = np.asarray(T, dtype=float).reshape(-1)
    p0 = float(p_arr[0])
    T0 = float(T_arr[0])

    rhoL = np.asarray(eos.PropertyModel.computeDensity(eos, np.array([p0]), flash.x, flash.Z_L, np.array([T0]), isLiquid=True), dtype=float).reshape(-1)
    rhoV = np.asarray(eos.PropertyModel.computeDensity(eos, np.array([p0]), flash.y, flash.Z_V, np.array([T0]), isLiquid=False), dtype=float).reshape(-1)
    rho_l = float(rhoL[0])
    rho_v = float(rhoV[0])

    vapor_mole = float(np.asarray(flash.V, dtype=float).reshape(-1)[0])
    liquid_mole = 1.0 - vapor_mole
    denom = rho_l * liquid_mole + rho_v * vapor_mole
    if denom <= 0.0 or not np.isfinite(denom):
        liquid_mass = 1.0 if liquid_mole >= vapor_mole else 0.0
    else:
        liquid_mass = rho_l * liquid_mole / denom
    vapor_mass = 1.0 - liquid_mass

    liquid_index = _phase_index(model, "getLiquidIndex", "get_liquid_index", 0)
    vapor_index = _phase_index(model, "getVaporIndex", "get_vapor_index", 1)
    extra = _extra_phase_names(model)
    nph = _number_of_phases(model, liquid_index, vapor_index, extra)
    frac = np.zeros(nph, dtype=float)
    rho = np.zeros(nph, dtype=float)
    frac[liquid_index] = liquid_mass
    frac[vapor_index] = vapor_mass
    rho[liquid_index] = rho_l
    rho[vapor_index] = rho_v

    phase_names = _phase_names(model, nph)
    fluid = getattr(model, "fluid", None)
    if isinstance(model, dict):
        fluid = model.get("fluid", fluid)
    for phase in extra:
        try:
            idx = phase_names.index(str(phase))
        except ValueError:
            lower = [item.lower() for item in phase_names]
            idx = lower.index(str(phase).lower()) if str(phase).lower() in lower else None
        if idx is None or idx >= nph:
            continue
        density = _surface_density(fluid, phase)
        if density is not None:
            rho[idx] = density

    return EquationOfStateComponent(name, p0, T0, component_index, frac, rho, float(mix.molarMass[hcpos]))


def get_eos_component(model, p, T, name: str, ci: int | None = None, *, mrst_indexing: bool = False) -> EquationOfStateComponent:
    """Pythonic alias for :func:`getEOSComponent`."""
    return getEOSComponent(model, p, T, name, ci, mrst_indexing=mrst_indexing)


__all__ = ["getEOSComponent", "get_eos_component"]
