"""Assign directly controlled values to AD well-solution rows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _set(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _phase_names(model) -> tuple[str, ...]:
    if hasattr(model, "getPhaseNames"):
        return tuple(str(x).upper()[0] for x in model.getPhaseNames())
    phases = getattr(getattr(model, "fluid", None), "phases", ())
    aliases = {"WATER": "W", "OIL": "O", "GAS": "G"}
    return tuple(aliases.get(str(x).upper(), str(x).upper()[0]) for x in phases)


def assign_well_values_from_control(
    model: Any,
    well_sol: Sequence[Any],
    wells: Sequence,
    wi: int = 0,
    oi: int = 1,
    gi: int = 2,
    *,
    copy_rows: bool = True,
):
    """Port MRST ``assignWellValuesFromControl`` using zero-based phase indices."""
    rows = deepcopy(list(well_sol)) if copy_rows else list(well_sol)
    if len(rows) != len(wells):
        raise ValueError("well_sol and wells must have equal length")
    active = set(_phase_names(model))
    phase_index = {"W": wi, "O": oi, "G": gi}
    for row, well in zip(rows, wells):
        if not bool(_get(row, "status", True)):
            continue
        control = str(_get(row, "type", well.type)).lower()
        value = float(_get(row, "val", well.val))
        sign = int(_get(row, "sign", well.sign))
        if control == "bhp":
            _set(row, "bhp", value)
        elif control == "rate":
            if sign < 1:
                continue
            compi = np.asarray(well.compi, dtype=float).ravel()
            for phase in ("W", "O", "G"):
                index = phase_index[phase]
                if phase in active and index < compi.size:
                    _set(row, f"q{phase}s", value * float(compi[index]))
        elif control in {"orat", "wrat", "grat"}:
            _set(row, f"q{control[0].upper()}s", value)
        elif control in {"lrat", "thp", "resv", "vrat", "resv_history"}:
            continue
        else:
            raise ValueError(f"unknown well control mode {control!r}")
    return rows


assignWellValuesFromControl = assign_well_values_from_control

__all__ = ["assign_well_values_from_control", "assignWellValuesFromControl"]
