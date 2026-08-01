"""Convert a parsed ECLIPSE/MRST schedule to PyMRST's schedule objects.

This ports the structural part of MRST ``convertDeckScheduleToMRST.m``:
step/control conversion, active-phase composition mapping, compositional
defaults, step limiting and consistent well ordering between control periods.
Controls may contain processed ``W`` records or raw ``WELSPECS``/``COMPDAT``
geometry with ``WCON*`` operating records. The raw path reuses PyMRST's shared
MRST record parser so compositional and black-oil schedules have one set of
defaulting and completion rules.
"""

from __future__ import annotations

from copy import deepcopy
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


def _processed_wells(
    ctrl: Mapping[str, Any], model, base_wells: Sequence[Well]
) -> list[Well]:
    if _get(ctrl, "W") is None:
        raw_keys = (
            "WELSPECS", "COMPDAT", "WCONPROD", "WCONHIST",
            "WCONINJE", "WCONINJ", "WCONINJH", "WELOPEN",
            "WELTARG", "WELLTARG",
        )
        if not any(_get(ctrl, key) is not None for key in raw_keys):
            return deepcopy(list(base_wells))
        grid = getattr(model, "G", None)
        rock = getattr(model, "rock", None)
        if grid is None or rock is None:
            raise ValueError("model.G and model.rock are required for raw well records")
        # These parsers already implement MRST processWells record defaulting,
        # completion expansion, WI construction, and control/limit semantics.
        from pymrst.modules.compositional.facility import (
            apply_deck_well_controls,
            wells_from_deck_completion_records,
        )
        gravity = float(getattr(
            model, "gravity_magnitude", getattr(model, "gravity", 0.0)
        ))
        geometry = None
        if _get(ctrl, "WELSPECS") is not None or _get(ctrl, "COMPDAT") is not None:
            geometry = wells_from_deck_completion_records(
                grid, rock, ctrl, compi=np.array([1.0, 0.0, 0.0]),
                gravity=np.array([0.0, 0.0, gravity]),
            )
        wells = list(base_wells)
        if geometry:
            by_name = {well.name: deepcopy(well) for well in wells}
            by_name.update({well.name: well for well in geometry})
            order = [well.name for well in wells]
            order.extend(well.name for well in geometry if well.name not in order)
            wells = [by_name[name] for name in order]
        wells = apply_deck_well_controls(wells, ctrl, grid=grid)
        phases = _phase_names(model)
        n_hc = _component_count(model)
        for well in wells:
            _map_composition(well, phases)
            if n_hc and well.components is None:
                well.components = np.full(n_hc, 1.0 / n_hc)
        return wells
    phases = _phase_names(model)
    n_hc = _component_count(model)
    wells = []
    for index, item in enumerate(_list(_get(ctrl, "W"))):
        well = _map_composition(_well(item, index), phases)
        if n_hc and well.components is None:
            well.components = np.full(n_hc, 1.0 / n_hc)
        wells.append(well)
    return wells


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
    previous_wells: list[Well] = []
    for ctrl in raw_controls:
        if not isinstance(ctrl, Mapping):
            raise TypeError("each deck control must be a mapping")
        controls.append(Control(
            W=_processed_wells(ctrl, model, previous_wells),
            bc=_get(ctrl, "bc"),
            src=_get(ctrl, "src"),
        ))
        previous_wells = deepcopy(list(controls[-1].W or []))
    if indices.size and indices.min() >= 1 and indices.max() <= len(controls):
        indices = indices - 1
    if step_limit is not None:
        if step_limit < 0:
            raise ValueError("step_limit must be non-negative")
        values, indices = values[:step_limit], indices[:step_limit]
    schedule = Schedule(values, indices, controls)
    if ensure_consistent:
        from pymrst.modules.ad_core.utils.make_schedule_consistent import (
            make_schedule_consistent,
        )
        schedule = make_schedule_consistent(schedule)
    return schedule


convertDeckScheduleToMRST = convert_deck_schedule_to_mrst

__all__ = ["convert_deck_schedule_to_mrst", "convertDeckScheduleToMRST"]
