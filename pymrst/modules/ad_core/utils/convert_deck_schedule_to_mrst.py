"""Convert a parsed ECLIPSE/MRST schedule to PyMRST's schedule objects.

This ports the structural part of MRST ``convertDeckScheduleToMRST.m``:
step/control conversion, active-phase composition mapping, compositional
defaults, step limiting and consistent well ordering between control periods.
The input controls must contain processed ``W`` records (``Well`` objects or
MRST well dictionaries). Raw ``WELSPECS``/``COMPDAT`` processing belongs to
the deck-format parser and is deliberately rejected here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.utils.schedule import Control, Schedule
from pymrst.params.wells_and_bc import Well


def _get(obj: Any, *names, default=None):
    if isinstance(obj, Mapping):
        for name in names:
            for key in (name, name.lower(), name.upper()):
                if key in obj:
                    return obj[key]
    return default


def _list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.ravel())
    return [value]


def _phase_names(model) -> tuple[str, ...]:
    if hasattr(model, "getPhaseNames"):
        return tuple(str(x).upper()[0] for x in model.getPhaseNames())
    phases = getattr(getattr(model, "fluid", None), "phases", ())
    aliases = {"WATER": "W", "OIL": "O", "GAS": "G", "SOLVENT": "S"}
    return tuple(aliases.get(str(x).upper(), str(x).upper()[0]) for x in phases)


def _well(value, index: int) -> Well:
    if isinstance(value, Well):
        return deepcopy(value)
    if not isinstance(value, Mapping):
        raise TypeError("processed control W entries must be Well objects or mappings")
    data = dict(value)
    cells = np.atleast_1d(np.asarray(_get(data, "cells", "cell", default=[]), dtype=np.int64))
    # Processed Python deck records use zero-based cells. Normalized MATLAB
    # structs may opt into conversion explicitly with ``one_based=True``.
    if bool(_get(data, "one_based", default=False)):
        cells = cells - 1

    def array(*names, default=0.0):
        raw = np.atleast_1d(np.asarray(_get(data, *names, default=default), dtype=float))
        return np.full(cells.size, float(raw[0])) if raw.size == 1 and cells.size > 1 else raw

    components = _get(data, "components")
    cstatus = _get(data, "cstatus")
    return Well(
        cells=cells,
        type=str(_get(data, "type", default="unknown")).lower(),
        val=float(np.asarray(_get(data, "val", default=0.0)).reshape(-1)[0]),
        r=array("r", "radius"),
        dir=str(_get(data, "dir", default="z")).lower(),
        WI=array("WI", "wi"),
        dZ=array("dZ", "dz"),
        name=str(_get(data, "name", default=f"W{index + 1}")),
        compi=np.atleast_1d(np.asarray(_get(data, "compi", default=np.ones(1)), float)),
        refDepth=float(np.asarray(_get(data, "refDepth", "ref_depth", default=0.0)).reshape(-1)[0]),
        sign=int(np.asarray(_get(data, "sign", default=0)).reshape(-1)[0]),
        status=bool(np.asarray(_get(data, "status", default=True)).reshape(-1)[0]),
        cstatus=None if cstatus is None else np.atleast_1d(np.asarray(cstatus, bool)),
        components=None if components is None else np.atleast_1d(np.asarray(components, float)),
        lims=_get(data, "lims"),
    )


def _map_composition(well: Well, phase_names: Sequence[str]) -> Well:
    deck_order = {"W": 0, "O": 1, "G": 2, "S": 3}
    raw = np.pad(np.asarray(well.compi, dtype=float).ravel(), (0, 4))[:4]
    mapped = np.array([raw[deck_order[name]] for name in phase_names], dtype=float)
    total = float(mapped.sum())
    if total <= 0:
        raise ValueError(f"well {well.name!r} has zero composition in active phases")
    well.compi = mapped / total
    return well


def _component_count(model) -> int:
    eos = getattr(model, "EOSModel", None)
    if eos is None:
        return 0
    getter = getattr(eos, "getNumberOfComponents", None)
    return int(getter()) if callable(getter) else 0


def _processed_wells(ctrl: Mapping[str, Any], model) -> list[Well]:
    if _get(ctrl, "W") is None:
        raw_keys = ("WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE")
        if any(_get(ctrl, key) is not None for key in raw_keys):
            raise NotImplementedError(
                "raw WELSPECS/COMPDAT controls require processWells support; "
                "supply processed control.W records"
            )
        return []
    phases = _phase_names(model)
    n_hc = _component_count(model)
    wells = []
    for index, item in enumerate(_list(_get(ctrl, "W"))):
        well = _map_composition(_well(item, index), phases)
        if n_hc and well.components is None:
            well.components = np.full(n_hc, 1.0 / n_hc)
        wells.append(well)
    return wells


def _consistent_controls(controls: list[Control]) -> list[Control]:
    """Keep MRST's invariant: every control has every well in one name order."""
    names = []
    templates = {}
    for ctrl in controls:
        for well in ctrl.W or []:
            if well.name not in templates:
                names.append(well.name)
                templates[well.name] = deepcopy(well)
    out = []
    for ctrl in controls:
        present = {well.name: well for well in ctrl.W or []}
        wells = []
        for name in names:
            if name in present:
                wells.append(present[name])
            else:
                inactive = deepcopy(templates[name])
                inactive.status = False
                inactive.val = 0.0
                if inactive.cstatus is not None:
                    inactive.cstatus = np.zeros_like(inactive.cstatus, dtype=bool)
                wells.append(inactive)
        out.append(replace(ctrl, W=wells))
    return out


def convert_deck_schedule_to_mrst(
    model,
    schedule_deck,
    *,
    step_limit: Optional[int] = None,
    ensure_consistent: bool = True,
) -> Schedule:
    """Convert deck-style ``step``/``control`` data into a :class:`Schedule`.

    MATLAB one-based control indices are converted to Python zero-based indices.
    """
    nested = _get(schedule_deck, "SCHEDULE")
    if nested is not None:
        schedule_deck = nested
    step = _get(schedule_deck, "step")
    if step is None and _get(schedule_deck, "step_val") is not None:
        step = {
            "val": _get(schedule_deck, "step_val"),
            "control": _get(schedule_deck, "step_control"),
        }
    if step is None:
        raise KeyError("schedule deck must contain step.val and step.control")
    values = np.asarray(_get(step, "val"), dtype=float).ravel()
    indices = np.asarray(_get(step, "control"), dtype=np.int64).ravel()
    if values.size != indices.size:
        raise ValueError("step.val and step.control must have equal length")

    raw_controls = _list(_get(schedule_deck, "control", "controls", default=[]))
    controls = []
    for ctrl in raw_controls:
        if not isinstance(ctrl, Mapping):
            raise TypeError("each deck control must be a mapping")
        controls.append(Control(
            W=_processed_wells(ctrl, model),
            bc=_get(ctrl, "bc"),
            src=_get(ctrl, "src"),
        ))
    if indices.size and indices.min() >= 1 and indices.max() <= len(controls):
        indices = indices - 1
    if step_limit is not None:
        if step_limit < 0:
            raise ValueError("step_limit must be non-negative")
        values, indices = values[:step_limit], indices[:step_limit]
    if ensure_consistent:
        controls = _consistent_controls(controls)
    return Schedule(values, indices, controls)


convertDeckScheduleToMRST = convert_deck_schedule_to_mrst

__all__ = ["convert_deck_schedule_to_mrst", "convertDeckScheduleToMRST"]
