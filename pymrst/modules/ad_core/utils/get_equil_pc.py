"""Capillary-pressure curves used by equilibrium initialization."""

from __future__ import annotations

import numpy as np


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _call(value, *names, args=()):
    for name in names:
        method = getattr(value, name, None)
        if method is not None:
            return method(*args)
    raise AttributeError(f"{type(value).__name__} has none of {names}")


def _phase_index(model, phase):
    return int(_call(model, "get_phase_index", "getPhaseIndex", args=(phase,)))


def _region_function(fluid, names, satnum):
    function = None
    for name in names:
        function = _field(fluid, name)
        if function is not None:
            break
    if isinstance(function, (list, tuple)):
        function = function[int(satnum)]
    elif isinstance(function, np.ndarray) and function.dtype == object:
        function = function.ravel()[int(satnum)]
    return function


def get_equil_pc(model, satnum, cells):
    """Return phase curves, signs, and per-cell J-function scale factors.

    Phase indices follow the model's native (zero-based) convention. ``satnum``
    is likewise zero-based when selecting a Python sequence of region curves.
    """
    active = np.asarray(
        _call(model, "get_active_phases", "getActivePhases"), dtype=bool
    )
    nph = int(active.sum())
    fluid = _field(model, "fluid")
    cells = np.asarray(cells, dtype=np.int64).ravel()

    curves = [lambda saturation: 0 * saturation for _ in range(nph)]
    signs = np.ones(nph)
    scalers = np.ones((cells.size, nph))

    grouping = _field(model, "FlowPropertyFunctions")
    pc_impl = _field(grouping, "CapillaryPressure") if grouping is not None else None

    def install(phase, curve_names, pair, sign):
        index = _phase_index(model, phase)
        signs[index] = sign
        curve = _region_function(fluid, curve_names, satnum)
        if curve is not None:
            curves[index] = curve
            if pc_impl is not None and _call(
                pc_impl, "has_j_function_scaler", "hasJFunctionScaler", args=(pair,)
            ):
                ratio = np.asarray(
                    _call(
                        pc_impl,
                        "get_j_function_static_ratio",
                        "getJFunctionStaticRatio",
                        args=(model,),
                    )
                ).ravel()
                tension = _call(
                    pc_impl,
                    "get_surface_tension",
                    "getSurfaceTension",
                    args=(pair,),
                )
                scalers[:, index] = ratio[cells] * tension

    if bool(_field(model, "water", False)):
        install("W", ("pcOW", "pc_ow"), "OW", -1.0)

    if bool(_field(model, "oil", False)):
        curves[_phase_index(model, "O")] = lambda saturation: 0 * saturation

    if bool(_field(model, "gas", False)):
        gas_names = ("pcOG", "pc_og")
        if _region_function(fluid, gas_names, satnum) is not None:
            install("G", gas_names, "OG", 1.0)
        elif not bool(_field(model, "oil", False)):
            install("G", ("pcWG", "pc_wg"), "WG", 1.0)

    return curves, signs, scalers


getEquilPC = get_equil_pc


__all__ = ["get_equil_pc", "getEquilPC"]
