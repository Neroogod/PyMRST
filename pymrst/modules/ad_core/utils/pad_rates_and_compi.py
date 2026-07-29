"""Pad active-phase rates and well compositions to W/O/G layout."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np

from pymrst.utils.adi import value


def _active_phases(model) -> np.ndarray:
    for name in ("getActivePhases", "get_active_phases"):
        method = getattr(model, name, None)
        if callable(method):
            active = method()
            if isinstance(active, tuple) and len(active) == 2:
                active = active[1]
            return np.asarray(active, dtype=bool).ravel()
    names = getattr(model, "getPhaseNames", lambda: ())()
    present = {str(name).upper()[0] for name in names}
    return np.array([phase in present for phase in ("W", "O", "G")])


def pad_rates_and_compi(q_s: Sequence[Any], wells: Sequence, model: Any):
    """Port MRST ``padRatesAndCompi``; return ``(q_wog, wells, is_active)``."""
    q_s = list(q_s)
    active = _active_phases(model)
    if active.size != 3:
        raise ValueError("active phase mask must have W/O/G length three")
    if len(q_s) != int(active.sum()):
        raise ValueError("one rate array is required per active phase")
    if not q_s:
        raise ValueError("at least one phase rate is required")
    dummy = np.zeros_like(np.asarray(value(q_s[0]), dtype=float))
    padded = [dummy.copy(), dummy.copy(), dummy.copy()]
    source = iter(q_s)
    for index, enabled in enumerate(active):
        if enabled:
            padded[index] = next(source)
    out_wells = deepcopy(list(wells))
    for well in out_wells:
        compi = np.asarray(well.compi, dtype=float).ravel()
        if compi.size != int(active.sum()):
            raise ValueError(f"well {well.name!r} composition does not match active phases")
        full = np.zeros(3)
        full[active] = compi
        well.compi = full
    return padded, out_wells, active


padRatesAndCompi = pad_rates_and_compi

__all__ = ["pad_rates_and_compi", "padRatesAndCompi"]
