"""MRST-style compositional facility helpers.

This module handles the small but important bridge between deck/schedule
facility records and PyMRST's native ``FacilityGroupControl`` objects. It
expects already-parsed records shaped like MRST's ``schedule.control`` fields
(``GRUPTREE``, ``GCONPROD``, ``GCONINJE``, ``GECON``, ``GRUPNET``,
``WELSPECS`` and ``WGRUPCON``), not raw ECLIPSE deck text.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import numpy as np

from pymrst.modules.compositional.model import FacilityGroupControl
from pymrst.params.wells_and_bc import add_well
from pymrst.params.wells_and_bc import Well
from pymrst.utils.units import ATM, BARSA


_MISSING = object()


def _struct_field_names(obj: Any) -> list[str]:
    if isinstance(obj, Mapping):
        return [str(k) for k in obj.keys()]
    dtype = getattr(obj, "dtype", None)
    if dtype is not None and getattr(dtype, "names", None):
        return [str(name) for name in dtype.names]
    names = getattr(obj, "_fieldnames", None)
    if names is not None:
        return [str(name) for name in names]
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return [str(k) for k in data.keys() if not str(k).startswith("_")]
    return []


def _struct_value(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    dtype = getattr(obj, "dtype", None)
    if dtype is not None and getattr(dtype, "names", None) and name in dtype.names:
        return obj[name]
    return getattr(obj, name)


def _struct_to_mapping(obj: Any) -> Any:
    if isinstance(obj, Mapping) or obj is None:
        return obj
    names = _struct_field_names(obj)
    if not names:
        return obj
    return {name: _struct_value(obj, name) for name in names}


def _records(obj: Any) -> list[Any]:
    if obj is None:
        return []
    obj = _struct_to_mapping(obj)
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return []
        if obj.dtype.names:
            return [_struct_to_mapping(item) for item in obj.ravel().tolist()]
        if obj.dtype == object:
            if obj.ndim == 0:
                return _records(obj.item())
            flat = obj.ravel().tolist()
            if obj.ndim == 1 and any(isinstance(_struct_to_mapping(item), Mapping) for item in flat):
                return [_struct_to_mapping(item) for item in flat]
        if obj.ndim == 1:
            return [obj.tolist()]
        return [row.tolist() for row in obj]
    if isinstance(obj, Mapping):
        return [obj]
    if isinstance(obj, (str, bytes)):
        return [obj]
    return list(obj)


def _field(record: Any, names: Sequence[str], index: int, default: Any = _MISSING) -> Any:
    record = _struct_to_mapping(record)
    if isinstance(record, Mapping):
        lower = {str(k).lower(): v for k, v in record.items()}
        for name in names:
            key = name.lower()
            if key in lower:
                return lower[key]
        if default is _MISSING:
            raise KeyError(f"record is missing {names[0]!r}")
        return default
    if isinstance(record, (str, bytes)):
        if index == 0:
            return record.decode() if isinstance(record, bytes) else record
        if default is _MISSING:
            raise IndexError(index)
        return default
    try:
        return record[index]
    except (IndexError, TypeError):
        if default is _MISSING:
            raise
        return default


def _name(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        x = x.decode()
    return str(x).strip()


def _status(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    text = _name(value).lower()
    if text in ("", "default", "yes", "open", "on", "true", "1"):
        return True
    if text in ("no", "shut", "stop", "off", "false", "0"):
        return False
    return default


def _float(value: Any, default: float = np.inf) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in ("", "default", "none"):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _finite_limit(value: Any) -> float | None:
    v = _float(value)
    return v if isfinite(v) else None


def _raw_record(record: Any, columns: Sequence[tuple[Sequence[str], int, Any]]) -> list[Any]:
    """Return MRST-column-ordered raw data for mapping records."""
    record = _struct_to_mapping(record)
    if isinstance(record, Mapping):
        return [_field(record, names, index, default) for names, index, default in columns]
    return list(record)


def _gconprod_limit(record: Any) -> dict[str, float]:
    """Map MRST ``GCONPROD`` columns to PyMRST control limit names."""
    pairs = [
        ("orat", ("orat", "oil_rate", "oil", "oil_target"), 2),
        ("wrat", ("wrat", "water_rate", "water", "water_target"), 3),
        ("grat", ("grat", "gas_rate", "gas", "gas_target"), 4),
        ("lrat", ("lrat", "liquid_rate", "liquid", "liquid_target"), 5),
    ]
    lims: dict[str, float] = {}
    for mode, names, index in pairs:
        value = _finite_limit(_field(record, names, index, np.inf))
        if value is not None:
            lims[mode] = value
    resv = _finite_limit(_field(record, ("resv", "reservoir_rate", "vrat"), 13, np.inf))
    if resv is not None:
        lims["vrat"] = resv
    nrat = _finite_limit(_field(record, ("nrat", "network_rate"), 18, np.inf))
    if nrat is not None:
        lims["nrat"] = nrat
    if isinstance(record, Mapping):
        total = _finite_limit(_field(record, ("rate", "total_rate", "target"), 8, np.inf))
        if total is not None:
            lims["rate"] = total
    return lims


def _gconprod_metadata(record: Any) -> dict[str, Any]:
    """Preserve MRST ``GCONPROD`` fields beyond native limit enforcement."""
    return {
        "raw": _raw_record(record, [
            (("group", "name", "gname"), 0, "Default"),
            (("control", "type", "mode"), 1, "NONE"),
            (("orat", "oil_rate", "oil", "oil_target"), 2, np.inf),
            (("wrat", "water_rate", "water", "water_target"), 3, np.inf),
            (("grat", "gas_rate", "gas", "gas_target"), 4, np.inf),
            (("lrat", "liquid_rate", "liquid", "liquid_target"), 5, np.inf),
            (("exceed_action", "exceed", "exccntrl"), 6, "NONE"),
            (("status", "open"), 7, "YES"),
            (("guide_rate", "guide"), 8, np.inf),
            (("guide_phase", "phase"), 9, ""),
            (("damping",), 10, "NONE"),
            (("free_group_control", "free_control"), 11, "NONE"),
            (("region", "region_control"), 12, "NONE"),
            (("resv", "reservoir_rate", "vrat"), 13, np.inf),
            (("voidage", "voidage_rate"), 14, np.inf),
            (("wet_gas", "wetgas"), 15, np.inf),
            (("calorific", "calorific_rate"), 16, np.inf),
            (("crat", "calorific_volume_rate"), 17, np.inf),
            (("nrat", "network_rate"), 18, np.inf),
            (("trgp", "target_pressure"), 19, np.inf),
            (("procedure", "proc"), 20, "NONE"),
        ]),
        "group": _name(_field(record, ("group", "name", "gname"), 0, "")),
        "control": _control_mode(_field(record, ("control", "type", "mode"), 1, "NONE")),
        "orat": _float(_field(record, ("orat", "oil_rate", "oil", "oil_target"), 2, np.inf)),
        "wrat": _float(_field(record, ("wrat", "water_rate", "water", "water_target"), 3, np.inf)),
        "grat": _float(_field(record, ("grat", "gas_rate", "gas", "gas_target"), 4, np.inf)),
        "lrat": _float(_field(record, ("lrat", "liquid_rate", "liquid", "liquid_target"), 5, np.inf)),
        "exceed_action": _name(_field(record, ("exceed_action", "exceed", "exccntrl"), 6, "NONE")),
        "status": _status(_field(record, ("status", "open"), 7, "YES")),
        "guide_rate": _float(_field(record, ("guide_rate", "guide"), 8, np.inf)),
        "guide_phase": _name(_field(record, ("guide_phase", "phase"), 9, "")),
        "damping": _name(_field(record, ("damping",), 10, "NONE")),
        "free_group_control": _name(_field(record, ("free_group_control", "free_control"), 11, "NONE")),
        "region": _name(_field(record, ("region", "region_control"), 12, "NONE")),
        "resv": _float(_field(record, ("resv", "reservoir_rate", "vrat"), 13, np.inf)),
        "voidage": _float(_field(record, ("voidage", "voidage_rate"), 14, np.inf)),
        "wet_gas": _float(_field(record, ("wet_gas", "wetgas"), 15, np.inf)),
        "calorific": _float(_field(record, ("calorific", "calorific_rate"), 16, np.inf)),
        "crat": _float(_field(record, ("crat", "calorific_volume_rate"), 17, np.inf)),
        "nrat": _float(_field(record, ("nrat", "network_rate"), 18, np.inf)),
        "trgp": _float(_field(record, ("trgp", "target_pressure"), 19, np.inf)),
        "procedure": _name(_field(record, ("procedure", "proc"), 20, "NONE")),
    }


def _gconinje_limit(record: Any) -> dict[str, float]:
    """Map MRST ``GCONINJE`` columns to PyMRST injection limit names."""
    phase = _name(_field(record, ("phase", "type"), 1, "")).lower()
    surf = _finite_limit(_field(record, ("surface", "surface_rate", "rate", "target"), 3, np.inf))
    resv = _finite_limit(_field(record, ("reservoir", "resv", "reservoir_rate"), 4, np.inf))
    lims: dict[str, float] = {}
    if surf is not None:
        if phase.startswith("wat"):
            lims["wrat"] = surf
        elif phase.startswith("oil"):
            lims["orat"] = surf
        elif phase.startswith("gas"):
            lims["grat"] = surf
        else:
            lims["rate"] = surf
    if resv is not None:
        lims["vrat"] = resv
    return lims


def _gconinje_metadata(record: Any) -> dict[str, Any]:
    """Preserve MRST ``GCONINJE`` fields beyond native limit enforcement."""
    return {
        "raw": _raw_record(record, [
            (("group", "name", "gname"), 0, "Default"),
            (("phase", "type"), 1, "Default"),
            (("control", "mode"), 2, "NONE"),
            (("surface", "surface_rate", "rate", "target"), 3, np.inf),
            (("reservoir", "resv", "reservoir_rate"), 4, np.inf),
            (("reinject_target", "target_reinject"), 5, np.inf),
            (("voidage_target", "target_voidage"), 6, np.inf),
            (("status", "open"), 7, "YES"),
            (("guide_rate", "guide"), 8, np.nan),
            (("guide_phase", "phase_guide"), 9, ""),
            (("trgp", "target_pressure"), 10, np.nan),
            (("vrep", "voidage_replacement"), 11, np.nan),
            (("wet_gas", "wetgas"), 12, np.inf),
        ]),
        "group": _name(_field(record, ("group", "name", "gname"), 0, "")),
        "phase": _name(_field(record, ("phase", "type"), 1, "")),
        "control": _control_mode(_field(record, ("control", "mode"), 2, "NONE")),
        "surface_rate": _float(_field(record, ("surface", "surface_rate", "rate", "target"), 3, np.inf)),
        "reservoir_rate": _float(_field(record, ("reservoir", "resv", "reservoir_rate"), 4, np.inf)),
        "reinject_target": _float(_field(record, ("reinject_target", "target_reinject"), 5, np.inf)),
        "voidage_target": _float(_field(record, ("voidage_target", "target_voidage"), 6, np.inf)),
        "status": _status(_field(record, ("status", "open"), 7, "YES")),
        "guide_rate": _float(_field(record, ("guide_rate", "guide"), 8, np.nan), np.nan),
        "guide_phase": _name(_field(record, ("guide_phase", "phase_guide"), 9, "")),
        "trgp": _float(_field(record, ("trgp", "target_pressure"), 10, np.nan), np.nan),
        "vrep": _float(_field(record, ("vrep", "voidage_replacement"), 11, np.nan), np.nan),
        "wet_gas": _float(_field(record, ("wet_gas", "wetgas"), 12, np.inf)),
    }


def _replace_duplicate_gconinje(records: Sequence[Any]) -> list[Any]:
    """Mirror MRST ``readGConInje`` replacement by ``(group, phase)``."""
    out: list[Any] = []
    index_by_key: dict[tuple[str, str], int] = {}
    for record in records:
        group = _name(_field(record, ("group", "name", "gname"), 0, "")).lower()
        phase = _name(_field(record, ("phase", "type"), 1, "")).lower()
        key = (group, phase)
        if key in index_by_key:
            out[index_by_key[key]] = record
        else:
            index_by_key[key] = len(out)
            out.append(record)
    return out


def _replace_duplicate_by_group(records: Sequence[Any]) -> list[Any]:
    """Mirror MRST keyword replacement by group name."""
    out: list[Any] = []
    index_by_group: dict[str, int] = {}
    for record in records:
        group = _name(_field(record, ("group", "name", "gname"), 0, "")).lower()
        if group in index_by_group:
            out[index_by_group[group]] = record
        else:
            index_by_group[group] = len(out)
            out.append(record)
    return out


def _gecon_metadata(record: Any) -> dict[str, Any]:
    """Preserve MRST ``GECON`` economics fields on imported group controls."""
    return {
        "raw": _raw_record(record, [
            (("group", "name", "gname"), 0, "Default"),
            (("min_oil_rate", "min_oil", "oil"), 1, 0.0),
            (("min_gas_rate", "min_gas", "gas"), 2, 0.0),
            (("max_water_cut", "water_cut", "wcut"), 3, 0.0),
            (("max_gas_oil_ratio", "max_gor", "gas_oil_ratio", "gor"), 4, 0.0),
            (("max_water_gas_ratio", "max_wgr", "water_gas_ratio", "wgr"), 5, 0.0),
            (("workover", "workover_procedure"), 6, "NONE"),
            (("end_run", "endrun"), 7, "NO"),
            (("follow_on", "followon"), 8, 0.0),
        ]),
        "min_oil_rate": _float(_field(record, ("min_oil_rate", "min_oil", "oil"), 1, np.inf)),
        "min_gas_rate": _float(_field(record, ("min_gas_rate", "min_gas", "gas"), 2, np.inf)),
        "max_water_cut": _float(_field(record, ("max_water_cut", "water_cut", "wcut"), 3, np.inf)),
        "max_gas_oil_ratio": _float(_field(record, ("max_gor", "gas_oil_ratio", "gor"), 4, np.inf)),
        "max_water_gas_ratio": _float(_field(record, ("max_wgr", "water_gas_ratio", "wgr"), 5, np.inf)),
        "workover": _name(_field(record, ("workover", "workover_procedure"), 6, "NONE")),
        "end_run": _status(_field(record, ("end_run", "endrun"), 7, "NO"), default=False),
        "follow_on": _float(_field(record, ("follow_on", "followon"), 8, 0.0), 0.0),
    }


def _grupnet_metadata(record: Any) -> dict[str, Any]:
    """Preserve MRST ``GRUPNET`` network-control fields on imported groups."""
    return {
        "raw": _raw_record(record, [
            (("group", "name", "gname"), 0, "Default"),
            (("guide_rate", "guide"), 1, -1.0),
            (("min_vfp_table", "vfp_min", "min_vfp"), 2, 0.0),
            (("max_vfp_table", "vfp_max", "max_vfp"), 3, 0.0),
            (("use_network", "network"), 4, "NO"),
            (("use_vfp", "vfp"), 5, "NO"),
            (("terminal", "node"), 6, "NONE"),
        ]),
        "guide_rate": _float(_field(record, ("guide_rate", "guide"), 1, -1.0), -1.0),
        "min_vfp_table": _float(_field(record, ("min_vfp_table", "vfp_min", "min_vfp"), 2, 0.0), 0.0),
        "max_vfp_table": _float(_field(record, ("max_vfp_table", "vfp_max", "max_vfp"), 3, 0.0), 0.0),
        "use_network": _status(_field(record, ("use_network", "network"), 4, "NO"), default=False),
        "use_vfp": _status(_field(record, ("use_vfp", "vfp"), 5, "NO"), default=False),
        "terminal": _name(_field(record, ("terminal", "node"), 6, "NONE")),
    }


def _wgrupcon_status(record: Any) -> bool:
    """Return the active flag from a MRST-like ``WGRUPCON`` row."""
    return _status(_field(record, ("status", "open", "active"), 1, "YES"))


def _wgrupcon_metadata(record: Any) -> dict[str, Any]:
    """Preserve MRST ``WGRUPCON`` guide-control fields on imported groups."""
    guide_rate = _float(_field(record, ("guide_rate", "guide", "target"), 2, -1.0), -1.0)
    guide_multiplier = _float(
        _field(record, ("guide_multiplier", "multiplier", "guide_factor", "factor"), 4, 1.0),
        1.0,
    )
    return {
        "raw": _raw_record(record, [
            (("well", "name", "wname"), 0, "Default"),
            (("status", "open", "active"), 1, "YES"),
            (("guide_rate", "guide", "target"), 2, -1.0),
            (("group", "group_name", "grp"), 3, ""),
            (("guide_multiplier", "multiplier", "guide_factor", "factor"), 4, 1.0),
        ]),
        "well": _name(_field(record, ("well", "name", "wname"), 0, "")),
        "status": _wgrupcon_status(record),
        "guide_rate": guide_rate,
        "guide_multiplier": guide_multiplier,
        "effective_guide_rate": guide_rate * guide_multiplier,
        "group": _name(_field(record, ("group", "group_name", "grp"), 3, "")),
    }


def _control_mode(value: Any) -> str:
    mode = _name(value).lower()
    aliases = {
        "resv": "vrat",
        "volume": "vrat",
    }
    return aliases.get(mode, mode)


def _phase_compi(phase: Any, nph: int) -> np.ndarray:
    """MRST-style injection stream phase selector for existing PyMRST wells."""
    phase = _name(phase).lower()
    out = np.zeros(max(int(nph), 1), dtype=float)
    if phase.startswith("wat"):
        out[0] = 1.0
    elif phase.startswith("oil"):
        out[0 if out.size == 1 else min(1, out.size - 1)] = 1.0
    elif phase.startswith("gas"):
        out[-1] = 1.0
    else:
        out[:] = 1.0
    if out.sum() <= 0.0:
        out[:] = 1.0
    return out / out.sum()


def _wconinjh_history_rate_record(record: Any) -> bool:
    """Return true for MRST-shaped WCONINJH rows.

    MRST's ``process_wconinjh`` does not have a control-mode column: column 4 is
    the historical injection rate and the well is always rate controlled.  Older
    PyMRST analogue examples used a WCONINJE-like textual control mode in that
    slot; those remain distinguishable because column 4 is not numeric.
    """
    if isinstance(record, Mapping):
        lower = {str(k).lower(): v for k, v in record.items()}
        if any(k in lower for k in ("control", "type", "mode")):
            return False
        return any(k in lower for k in ("rate", "surface_rate", "target"))
    return isfinite(_float(_field(record, ("rate", "surface_rate"), 3, np.nan), np.nan))


def _full_wconhist_record(record: Any) -> bool:
    return not isinstance(record, Mapping) and len(record) > 9


def _wconhist_resv_target_and_compi(record: Any, nph: int = 3) -> tuple[float, np.ndarray]:
    """MRST ``process_wconhist`` RESV target and phase split.

    MRST reads historical oil/water/gas rates from columns 4--6, negates them
    for producers, sums them into the active target, and stores WOG phase
    fractions after reordering from the row's OWG columns.
    """
    oil = _float(_field(record, ("orat", "oil_rate", "oil"), 3, 0.0), 0.0)
    water = _float(_field(record, ("wrat", "water_rate", "water"), 4, 0.0), 0.0)
    gas = _float(_field(record, ("grat", "gas_rate", "gas"), 5, 0.0), 0.0)
    rates_owg = -np.asarray([oil, water, gas], dtype=float)
    val = float(rates_owg.sum())
    if abs(val) > 0.0:
        wog = rates_owg[[1, 0, 2]] / val
    else:
        wog = np.ones(3, dtype=float) / 3.0

    nph = max(int(nph), 1)
    if nph >= 3:
        compi = wog
    elif nph == 2:
        oil_gas = np.maximum(np.asarray([wog[1], wog[2]], dtype=float), 0.0)
        compi = oil_gas / oil_gas.sum() if oil_gas.sum() > 0.0 else np.ones(2, dtype=float) / 2.0
    else:
        compi = np.ones(1, dtype=float)
    return val, compi


def _project_wog_compi(wog: Sequence[float], nph: int) -> np.ndarray:
    """Project MRST WOG phase fractions onto the native well component length."""
    wog_arr = np.asarray(wog, dtype=float).ravel()
    nph = max(int(nph), 1)
    if nph >= 3:
        out = np.zeros(nph, dtype=float)
        out[:3] = wog_arr[:3]
    elif nph == 2:
        out = np.maximum(np.asarray([wog_arr[1], wog_arr[2]], dtype=float), 0.0)
        if out.sum() <= 0.0:
            out[:] = 1.0
    else:
        out = np.ones(1, dtype=float)
    if out.sum() <= 0.0:
        out[:] = 1.0
    return out / out.sum()


def _wconhist_lrat_target_and_compi(record: Any, nph: int = 3) -> tuple[float, np.ndarray]:
    oil = _float(_field(record, ("orat", "oil_rate", "oil"), 3, 0.0), 0.0)
    water = _float(_field(record, ("wrat", "water_rate", "water"), 4, 0.0), 0.0)
    rates_wo = -np.asarray([water, oil], dtype=float)
    val = float(rates_wo.sum())
    if abs(val) > 0.0:
        wog = np.asarray([rates_wo[0], rates_wo[1], 0.0], dtype=float) / val
    else:
        wog = np.asarray([0.5, 0.5, 0.0], dtype=float)
    return val, _project_wog_compi(wog, nph)


def _wconhist_compi(record: Any, mode: str, nph: int) -> np.ndarray | None:
    if not _full_wconhist_record(record):
        return None
    mode = str(mode).lower()
    if mode == "wrat":
        return _project_wog_compi([1.0, 0.0, 0.0], nph)
    if mode in {"orat", "bhp"}:
        return _project_wog_compi([0.0, 1.0, 0.0], nph)
    if mode == "grat":
        return _project_wog_compi([0.0, 0.0, 1.0], nph)
    if mode == "lrat":
        _, compi = _wconhist_lrat_target_and_compi(record, nph)
        return compi
    if mode == "resv_history":
        _, compi = _wconhist_resv_target_and_compi(record, nph)
        return compi
    return None


def _wconprod_compi(mode: str, nph: int) -> np.ndarray | None:
    """Return MRST's default phase-composition vector for WCONPROD controls."""
    mode = str(mode).lower()
    if mode == "wrat":
        return _project_wog_compi([1.0, 0.0, 0.0], nph)
    if mode == "orat":
        return _project_wog_compi([0.0, 1.0, 0.0], nph)
    if mode == "grat":
        return _project_wog_compi([0.0, 0.0, 1.0], nph)
    if mode == "lrat":
        return _project_wog_compi([1.0, 1.0, 0.0], nph)
    if mode in {"vrat", "bhp"}:
        return _project_wog_compi([1.0, 1.0, 1.0], nph)
    return None


def _vfp_index(record: Any, *, keyword: str) -> int | None:
    kw = str(keyword).upper()
    index = 10 if kw in {"WCONPROD", "WCONINJ"} else 8
    value = _field(record, ("vfp_index", "vfp", "vfp_table_index", "table"), index, None)
    if value is None:
        return None
    try:
        value_float = float(np.asarray(value).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(value_float):
        return None
    return int(value_float)


def _prod_lims(record: Any, *, keyword: str = "WCONPROD") -> dict[str, float]:
    lims: dict[str, float] = {}
    kw = str(keyword).upper()
    compact_wconhist = kw == "WCONHIST" and (not isinstance(record, Mapping)) and len(record) <= 9
    for mode, names, index in [
        ("orat", ("orat", "oil_rate", "oil"), 3),
        ("wrat", ("wrat", "water_rate", "water"), 4),
        ("grat", ("grat", "gas_rate", "gas"), 5),
        ("lrat", ("lrat", "liquid_rate", "liquid"), 6),
        ("vrat", ("vrat", "resv", "reservoir_rate"), 7),
        ("bhp", ("bhp", "bottom_hole_pressure"), 8 if compact_wconhist or kw != "WCONHIST" else 9),
        ("thp", ("thp", "tubing_head_pressure"), 9),
    ]:
        if kw == "WCONHIST" and mode == "thp":
            continue
        default = ATM if mode == "bhp" else np.inf
        value = _finite_limit(_field(record, names, index, default))
        if value is not None:
            lims[mode] = -value if mode in {"orat", "wrat", "grat", "lrat", "vrat"} else value
    if kw == "WCONHIST" and not compact_wconhist and not isinstance(record, Mapping):
        oil = _finite_limit(_field(record, ("orat", "oil_rate", "oil"), 3, np.inf))
        water = _finite_limit(_field(record, ("wrat", "water_rate", "water"), 4, np.inf))
        if oil is not None and water is not None:
            lims["lrat"] = -(oil + water)
        if _name(_field(record, ("control", "type", "mode"), 2, "")).lower() == "resv":
            val, _ = _wconhist_resv_target_and_compi(record)
            lims["resv_history"] = val
    return lims


def _inje_lims(record: Any, *, keyword: str = "WCONINJE") -> dict[str, float]:
    lims: dict[str, float] = {}
    kw = str(keyword).upper()
    if kw == "WCONINJH" and _wconinjh_history_rate_record(record):
        columns = [
            ("rate", ("rate", "surface_rate", "target"), 3, np.inf),
            ("bhp", ("bhp", "bottom_hole_pressure"), 6, 6895.0 * BARSA),
        ]
    elif kw == "WCONINJ":
        # Legacy WCONINJ uses a different MRST column layout from WCONINJE:
        # RATE/RESV are columns 5/6, BHP/THP are columns 9/10.  Older PyMRST
        # analogue rows sometimes used a compact 8-column shape; keep that
        # accepted for native examples while preferring the MRST layout.
        compact = (not isinstance(record, Mapping)) and len(record) <= 8
        columns = [
            ("rate", ("rate", "surface_rate"), 4, np.inf),
            ("vrat", ("vrat", "resv", "reservoir_rate"), 5, np.inf),
            ("bhp", ("bhp", "bottom_hole_pressure"), 6 if compact else 8, -1.0),
            ("thp", ("thp", "tubing_head_pressure"), 7 if compact else 9, np.inf),
        ]
    else:
        columns = [
            ("rate", ("rate", "surface_rate"), 4, np.inf),
            ("vrat", ("vrat", "resv", "reservoir_rate"), 5, np.inf),
            ("bhp", ("bhp", "bottom_hole_pressure"), 6, 6895.0 * BARSA),
            ("thp", ("thp", "tubing_head_pressure"), 7, np.inf),
        ]
    for mode, names, index, default in columns:
        value = _finite_limit(_field(record, names, index, default))
        if value is not None:
            lims[mode] = value
    return lims


_PRODUCER_CONTROL_KEYWORDS = ("WCONPROD", "WCONHIST")
_INJECTOR_CONTROL_KEYWORDS = ("WCONINJE", "WCONINJ", "WCONINJH")
_WELL_CONTROL_KEYWORDS = (*_PRODUCER_CONTROL_KEYWORDS, *_INJECTOR_CONTROL_KEYWORDS)


def _control_records(control: Mapping[str, Any], *names: str) -> list[Any]:
    control = _struct_to_mapping(control)
    out: list[Any] = []
    for name in names:
        out.extend(_records(control.get(name, control.get(name.lower()))))
    return out


def _record_well_name(record: Any) -> str:
    return _name(_field(record, ("well", "name", "wname"), 0, ""))


def _replace_duplicate_well_records(records: Sequence[Any]) -> list[Any]:
    """Mirror MRST ``unique_well_records(..., 'last')`` for one keyword."""
    out: list[Any] = []
    index_by_well: dict[str, int] = {}
    for record in records:
        well = _record_well_name(record).lower()
        if well in index_by_well:
            out[index_by_well[well]] = record
        else:
            index_by_well[well] = len(out)
            out.append(record)
    return out


def _well_control_records_by_keyword(control: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Build MRST-style active well-control keyword tables.

    MRST ``assignControlRecords`` keeps only the last row for a well inside one
    keyword table and removes that well from all other well-control keyword
    tables. Python ports use mapping insertion order as the deck keyword order.
    """
    records_by_kw: dict[str, list[Any]] = {kw: [] for kw in _WELL_CONTROL_KEYWORDS}
    for key, value in control.items():
        kw = str(key).upper()
        if kw not in records_by_kw:
            continue
        incoming = _replace_duplicate_well_records(_records(value))
        incoming_names = {_record_well_name(record).lower() for record in incoming}
        for other_kw in _WELL_CONTROL_KEYWORDS:
            records_by_kw[other_kw] = [
                record for record in records_by_kw[other_kw]
                if _record_well_name(record).lower() not in incoming_names
            ]
        records_by_kw[kw].extend(incoming)
    return records_by_kw


def _apply_weltarg(records_by_kw: dict[str, list[Any]], targets: Sequence[Any]) -> None:
    """Apply MRST ``WELTARG``/``WELLTARG`` records to parsed control records."""
    prod_columns = {"orat": 3, "wrat": 4, "grat": 5, "lrat": 6, "crat": 18,
                    "resv": 7, "vrat": 7, "bhp": 8, "thp": 9}
    inje_columns = {"rate": 4, "resv": 5, "vrat": 5, "bhp": 6, "thp": 7}
    for target in targets:
        well = _name(_field(target, ("well", "name", "wname"), 0, ""))
        mode = _control_mode(_field(target, ("mode", "control", "target"), 1, ""))
        value = _float(_field(target, ("value", "val"), 2, np.inf), np.inf)
        if not well or not isfinite(value):
            continue
        for keywords, columns in (
            (_PRODUCER_CONTROL_KEYWORDS, prod_columns),
            (_INJECTOR_CONTROL_KEYWORDS, inje_columns),
        ):
            col = columns.get(mode)
            if col is None:
                continue
            for kw in keywords:
                for record in records_by_kw.get(kw, []):
                    if _name(_field(record, ("well", "name", "wname"), 0, "")) != well:
                        continue
                    record_col = col
                    if (
                        kw == "WCONHIST"
                        and mode == "bhp"
                        and not isinstance(record, Mapping)
                        and len(record) > 9
                    ):
                        record_col = 9
                    elif kw == "WCONINJH" and mode == "rate" and _wconinjh_history_rate_record(record):
                        record_col = 3
                    if isinstance(record, Mapping):
                        record[mode] = value
                    else:
                        while len(record) <= record_col:
                            record.append(np.inf)
                        record[record_col] = value


def _welopen_is_whole_well(record: Any) -> bool:
    """Return true for MRST ``WELOPEN`` rows that target the whole well."""
    columns = [
        (("i", "i1", "ix", "col2"), 2),
        (("j", "j1", "jy", "col3"), 3),
        (("k", "k1", "k_lower", "col4"), 4),
        (("lower", "lower_completion", "first_completion", "col5"), 5),
        (("upper", "upper_completion", "last_completion", "col6"), 6),
    ]
    for names, index in columns:
        value = _field(record, names, index, -1)
        if _float(value, -1.0) >= 0.0:
            return False
    return True


def _set_control_record_status(record: Any, status: str, column: int) -> None:
    if isinstance(record, Mapping):
        record["status"] = status
        return
    while len(record) <= column:
        record.append("OPEN")
    record[column] = status


def _apply_welopen(records_by_kw: dict[str, list[Any]], welopen: Sequence[Any]) -> None:
    """Apply whole-well MRST ``WELOPEN`` status updates to active controls."""
    status_columns = {
        **{kw: 1 for kw in _PRODUCER_CONTROL_KEYWORDS},
        **{kw: 2 for kw in _INJECTOR_CONTROL_KEYWORDS},
    }
    for record in welopen:
        if not _welopen_is_whole_well(record):
            continue
        well = _record_well_name(record).lower()
        status = _name(_field(record, ("status", "open"), 1, "OPEN")).upper()
        if not well or status not in {"OPEN", "SHUT", "STOP"}:
            continue
        for kw, column in status_columns.items():
            for control_record in records_by_kw.get(kw, []):
                if _record_well_name(control_record).lower() == well:
                    _set_control_record_status(control_record, status, column)


def _welopen_completion_bounds(record: Any, nperf: int) -> tuple[int, int] | None:
    lo = int(_float(_field(record, ("lower", "lower_completion", "first_completion", "col5"), 5, -1), -1.0))
    hi = int(_float(_field(record, ("upper", "upper_completion", "last_completion", "col6"), 6, -1), -1.0))
    if lo < 1:
        lo = 1
    if hi < 1:
        hi = nperf
    lo = max(lo, 1)
    hi = min(hi, nperf)
    if hi < lo:
        return None
    return lo - 1, hi


def _well_cartesian_ijk(well: Well, grid: Any) -> np.ndarray | None:
    cart_dims = getattr(grid, "cartDims", None)
    cells_obj = getattr(grid, "cells", None)
    if cart_dims is None or cells_obj is None:
        return None
    dims = list(np.asarray(cart_dims, dtype=np.int64).ravel()) + [1, 1, 1]
    nx, ny, _ = (int(dims[0]), int(dims[1]), int(dims[2]))
    cells = np.asarray(well.cells, dtype=np.int64).ravel()
    index_map = getattr(cells_obj, "indexMap", None)
    linear = cells if index_map is None else np.asarray(index_map, dtype=np.int64).ravel()[cells]
    i = linear % nx
    j = (linear // nx) % ny
    k = linear // (nx * ny)
    return np.column_stack([i, j, k]) + 1


def _welopen_ijk_mask(record: Any, well: Well, grid: Any | None) -> np.ndarray | None:
    values = np.array([
        _float(_field(record, names, index, -1), -1.0)
        for names, index in [
            (("i", "i1", "ix", "col2"), 2),
            (("j", "j1", "jy", "col3"), 3),
            (("k", "k1", "k_lower", "col4"), 4),
        ]
    ], dtype=float)
    nperf = len(well.cells)
    if np.all(values == 0.0):
        return np.ones(nperf, dtype=bool)
    if np.any(values > 0.0):
        ijk = None if grid is None else _well_cartesian_ijk(well, grid)
        if ijk is None:
            return None
        mask = np.ones(nperf, dtype=bool)
        for axis, value in enumerate(values):
            if value == 0.0:
                continue
            if value < 0.0:
                return np.zeros(nperf, dtype=bool)
            mask &= ijk[:, axis] == int(value)
        return mask
    if np.any(values < 0.0):
        return None
    return np.ones(nperf, dtype=bool)


def _well_cstatus_array(well: Well) -> np.ndarray:
    nperf = len(well.cells)
    cstatus = getattr(well, "cstatus", None)
    if cstatus is None:
        return np.ones(nperf, dtype=bool)
    out = np.asarray(cstatus, dtype=bool).ravel().copy()
    if out.size == 1 and nperf > 1:
        return np.full(nperf, bool(out[0]), dtype=bool)
    if out.size != nperf:
        raise ValueError(f"well {well.name!r} has {out.size} completion statuses for {nperf} perforations")
    return out


def _deck_cartesian_cell(grid: Any, i: int, j: int, k: int) -> int | None:
    cart_dims = getattr(grid, "cartDims", None)
    cells_obj = getattr(grid, "cells", None)
    if cart_dims is None or cells_obj is None:
        return None
    dims = list(np.asarray(cart_dims, dtype=np.int64).ravel()) + [1, 1, 1]
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    if i < 1 or j < 1 or k < 1 or i > nx or j > ny or k > nz:
        return None
    global_cell = (i - 1) + nx * (j - 1) + nx * ny * (k - 1)
    index_map = getattr(cells_obj, "indexMap", None)
    if index_map is None:
        return int(global_cell)
    matches = np.flatnonzero(np.asarray(index_map, dtype=np.int64).ravel() == global_cell)
    if matches.size == 0:
        return None
    return int(matches[0])


def _compdat_well_name(record: Any) -> str:
    return _name(_field(record, ("well", "name", "wname"), 0, ""))


def _welspecs_by_name(records: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for record in records:
        name = _compdat_well_name(record)
        if name:
            out[name.lower()] = record
    return out


def _welspecs_ij(record: Any | None) -> tuple[int, int]:
    if record is None:
        return -1, -1
    i = int(_float(_field(record, ("i", "I", "ix", "col2"), 2, -1), -1.0))
    j = int(_float(_field(record, ("j", "J", "jy", "col3"), 3, -1), -1.0))
    return i, j


def _welspecs_ref_depth(record: Any | None) -> float | None:
    if record is None:
        return None
    value = _float(_field(record, ("ref_depth", "refDepth", "datum_depth", "col4"), 4, np.nan), np.nan)
    return float(value) if isfinite(value) else None


def _welspecs_group(record: Any | None) -> str | None:
    if record is None:
        return None
    group = _name(_field(record, ("group", "group_name", "col1"), 1, ""))
    return group or None


def _grid_nlayers(grid: Any) -> int:
    cart_dims = getattr(grid, "cartDims", None)
    if cart_dims is None:
        cells_obj = getattr(grid, "cells", None)
        num = getattr(cells_obj, "num", None)
        return int(num) if num is not None else 1
    dims = list(np.asarray(cart_dims, dtype=np.int64).ravel()) + [1, 1, 1]
    return int(max(dims[2], 1))


def _default_compdat_record(name: str, nlayers: int) -> list[Any]:
    """MRST ``insertDefaultCOMPDAT`` row for WELSPECS-only vertical wells."""
    return [name, -1, -1, 1, int(nlayers), "OPEN", 0, 0.0, 0.1, -1.0, 0.0, "Default", "Z", -1]


def _completion_status_text(value: Any, default: str = "OPEN") -> str:
    text = _name(value).upper()
    if text in {"", "DEFAULT"}:
        text = default
    return text if text in {"OPEN", "SHUT", "STOP"} else default


def _compdat_status_text(record: Any) -> str:
    return _completion_status_text(_field(record, ("status", "open", "col5"), 5, "OPEN"))


def _compdat_status(record: Any) -> bool:
    return _status(_compdat_status_text(record))


def _compdat_direction(record: Any, default: str = "z") -> str:
    direction = _name(_field(record, ("dir", "direction", "well_dir", "col12"), 12, default)).lower()
    if direction in {"x", "y", "z"}:
        return direction
    return default


def _compdat_radius(record: Any, default: float = 0.1) -> float:
    radius = _float(_field(record, ("radius", "r", "well_radius"), -1, np.nan), np.nan)
    if isfinite(radius) and radius > 0.0:
        return float(radius)
    diameter = _float(_field(record, ("diameter", "diam", "col8"), 8, np.nan), np.nan)
    if isfinite(diameter) and diameter > 0.0:
        return 0.5 * float(diameter)
    return float(default)


def _compdat_wi(record: Any) -> float | None:
    value = _float(
        _field(record, ("wi", "WI", "connection_transmissibility", "transmissibility", "col7"), 7, np.nan),
        np.nan,
    )
    return float(value) if isfinite(value) and value >= 0.0 else None


def _compdat_satnum(record: Any) -> int:
    return int(_float(_field(record, ("satnum", "sat_num", "sat_region", "col6"), 6, 0), 0.0))


def _compdat_kh(record: Any) -> float:
    value = _float(_field(record, ("kh", "KH", "perm_thickness", "col9"), 9, -1.0), -1.0)
    return float(value) if isfinite(value) and value > 0.0 else -1.0


def _compdat_skin(record: Any) -> float:
    return float(_float(_field(record, ("skin", "col10"), 10, 0.0), 0.0))


def _compdat_d_factor(record: Any) -> float:
    return float(_float(_field(record, ("d_factor", "dfactor", "d", "col11"), 11, 0.0), 0.0))


def _last_completion_indices(cells: Sequence[int]) -> np.ndarray:
    """Return surviving completion indices, matching MRST ``unique(...,'last')``."""
    last: dict[int, int] = {}
    for index, cell in enumerate(cells):
        last[int(cell)] = index
    return np.asarray(sorted(last.values()), dtype=np.int64)


def wells_from_deck_completion_records(
    grid: Any,
    rock: Any,
    control: Mapping[str, Any],
    *,
    well_type: str = "bhp",
    val: float = 0.0,
    compi: Sequence[float] | None = None,
    gravity: Any | None = None,
    default_radius: float = 0.1,
) -> list[Well]:
    """Build native wells from MRST-like ``WELSPECS``/``COMPDAT`` records.

    This ports the geometry part that ``apply_deck_well_controls`` deliberately
    does not create: one or more ``COMPDAT`` rows are expanded into perforated
    cells, completion status is stored as ``Well.cstatus``, and optional
    connection WI/radius/skin/direction metadata is passed to ``add_well``.
    Parsed records are expected to be in PyMRST/native units.
    """
    control = _struct_to_mapping(control)
    if not isinstance(control, Mapping):
        raise TypeError("deck completion control must be a mapping or MATLAB-style struct")
    welspecs = _welspecs_by_name(_records(control.get("WELSPECS", control.get("welspecs"))))
    compdat = _records(control.get("COMPDAT", control.get("compdat")))
    by_well: dict[str, list[Any]] = defaultdict(list)
    for record in compdat:
        name = _compdat_well_name(record)
        if name:
            by_well[name].append(record)
    present = {name.lower() for name in by_well}
    nlayers = _grid_nlayers(grid)
    for spec in welspecs.values():
        name = _compdat_well_name(spec)
        if name and name.lower() not in present:
            by_well[name].append(_default_compdat_record(name, nlayers))
            present.add(name.lower())

    wells: list[Well] = []
    compi_arr = None if compi is None else np.asarray(compi, dtype=float).ravel()
    for name, records in by_well.items():
        spec = welspecs.get(name.lower())
        spec_i, spec_j = _welspecs_ij(spec)
        cells: list[int] = []
        radii: list[float] = []
        wi_values: list[float] = []
        explicit_wi: list[bool] = []
        skins: list[float] = []
        kh_values: list[float] = []
        satnums: list[int] = []
        d_factors: list[float] = []
        cstatus: list[bool] = []
        cstatus_text: list[str] = []
        directions: list[str] = []
        for record in records:
            i = int(_float(_field(record, ("i", "I", "ix", "col1"), 1, spec_i), spec_i))
            j = int(_float(_field(record, ("j", "J", "jy", "col2"), 2, spec_j), spec_j))
            if i <= 0:
                i = spec_i
            if j <= 0:
                j = spec_j
            k1 = int(_float(_field(record, ("k1", "k_lower", "lower", "col3"), 3, 1), 1.0))
            k2 = int(_float(_field(record, ("k2", "k_upper", "upper", "col4"), 4, k1), float(k1)))
            if k2 < k1:
                continue
            for k in range(k1, k2 + 1):
                cell = _deck_cartesian_cell(grid, i, j, k)
                if cell is None:
                    continue
                cells.append(cell)
                radii.append(_compdat_radius(record, default_radius))
                wi = _compdat_wi(record)
                if wi is None:
                    wi_values.append(np.nan)
                    explicit_wi.append(False)
                else:
                    wi_values.append(wi)
                    explicit_wi.append(True)
                skins.append(_compdat_skin(record))
                kh_values.append(_compdat_kh(record))
                satnums.append(_compdat_satnum(record))
                d_factors.append(_compdat_d_factor(record))
                status_text = _compdat_status_text(record)
                cstatus.append(_status(status_text))
                cstatus_text.append(status_text)
                directions.append(_compdat_direction(record))
        if not cells:
            continue
        keep = _last_completion_indices(cells)
        cells_arr = np.asarray(cells, dtype=np.int64)[keep]
        radii_arr = np.asarray(radii, dtype=float)[keep]
        wi_arr = np.asarray(wi_values, dtype=float)[keep]
        explicit_wi_arr = np.asarray(explicit_wi, dtype=bool)[keep]
        all_wi_explicit = bool(explicit_wi_arr.all())
        kh_arr = np.asarray(kh_values, dtype=float)[keep]
        skin: float | np.ndarray = np.asarray(skins, dtype=float)[keep]
        cstatus_arr = np.asarray(cstatus, dtype=bool)[keep]
        cstatus_text_arr = np.asarray(cstatus_text, dtype=object)[keep]
        satnum_arr = np.asarray(satnums, dtype=np.int64)[keep]
        d_factor_arr = np.asarray(d_factors, dtype=float)[keep]
        directions_arr = np.asarray(directions, dtype=object)[keep]
        well_dir = str(directions_arr[0]) if directions_arr.size else "z"
        if np.allclose(skin, skin[0]):
            skin = float(skin[0])
        wells = add_well(
            wells,
            grid,
            rock,
            cells_arr,
            well_type=well_type,
            val=val,
            radius=radii_arr,
            well_dir=well_dir,
            name=name,
            skin=skin,
            kh=None if all_wi_explicit else kh_arr,
            wi=wi_arr if all_wi_explicit else None,
            compi=compi_arr,
            ref_depth=_welspecs_ref_depth(spec),
            gravity=gravity,
        )
        if not all_wi_explicit and explicit_wi_arr.any():
            wells[-1].WI[explicit_wi_arr] = wi_arr[explicit_wi_arr]
        wells[-1].cstatus = cstatus_arr
        setattr(wells[-1], "cstatus_text", cstatus_text_arr)
        setattr(wells[-1], "satnum", satnum_arr)
        setattr(wells[-1], "kh", kh_arr)
        setattr(wells[-1], "d_factor", d_factor_arr)
        group = _welspecs_group(spec)
        if group is not None:
            setattr(wells[-1], "group", group)
        setattr(wells[-1], "compdat", [list(r) if not isinstance(r, Mapping) else dict(r) for r in records])
        if spec is not None:
            setattr(wells[-1], "welspecs", list(spec) if not isinstance(spec, Mapping) else dict(spec))
    return wells


def _apply_completion_welopen(wells: Sequence[Well], welopen: Sequence[Any], grid: Any | None = None) -> None:
    """Apply MRST-like ``WELOPEN`` completion ranges to wells."""
    by_name = {well.name.lower(): well for well in wells}
    for record in welopen:
        if _welopen_is_whole_well(record):
            continue
        well = by_name.get(_record_well_name(record).lower())
        if well is None:
            continue
        bounds = _welopen_completion_bounds(record, len(well.cells))
        if bounds is None:
            continue
        ijk_mask = _welopen_ijk_mask(record, well, grid)
        if ijk_mask is None:
            continue
        lo, hi = bounds
        range_mask = np.zeros(len(well.cells), dtype=bool)
        range_mask[lo:hi] = True
        cstatus = _well_cstatus_array(well)
        status_text = _completion_status_text(_field(record, ("status", "open"), 1, "OPEN"))
        mask = range_mask & ijk_mask
        cstatus[mask] = _status(status_text)
        well.cstatus = cstatus
        if hasattr(well, "cstatus_text"):
            current = np.asarray(getattr(well, "cstatus_text"), dtype=object).ravel().copy()
            if current.size == 1 and len(well.cells) > 1:
                current = np.full(len(well.cells), str(current[0]), dtype=object)
            if current.size == len(well.cells):
                current[mask] = status_text
                setattr(well, "cstatus_text", current)


def apply_deck_well_controls(
    wells: Sequence[Well],
    control: Mapping[str, Any],
    *,
    copy_wells: bool = True,
    grid: Any | None = None,
) -> list[Well]:
    """Apply MRST well-control records to wells.

    The geometry-defining well objects are assumed to already exist (from
    ``WELSPECS``/``COMPDAT`` or a native PyMRST setup). This helper mirrors the
    control part of MRST's ``processWells``: it updates matching well names with
    status, sign, active control type/value, stream selector, and operating
    limits. Unknown well names are ignored, matching MRST's schedule-update
    spirit where geometry and controls are separate concerns. Supported
    keywords are ``WCONPROD``, ``WCONHIST``, ``WCONINJE``, ``WCONINJ``,
    ``WCONINJH``, whole-well and completion-level ``WELOPEN``, ``WELTARG`` and
    ``WELLTARG``. Grid-aware positive-I/J/K ``WELOPEN`` matching is available
    when a Cartesian ``grid`` is supplied; otherwise only wildcard-I/J/K
    completion ranges are applied.
    """
    import copy

    control = _struct_to_mapping(control)
    if not isinstance(control, Mapping):
        raise TypeError("deck well control must be a mapping or MATLAB-style struct")
    out = copy.deepcopy(list(wells)) if copy_wells else list(wells)
    by_name = {w.name: w for w in out}
    records_by_kw = _well_control_records_by_keyword(control)
    welopen = _records(control.get("WELOPEN", control.get("welopen", control.get("wellopen"))))
    _apply_welopen(records_by_kw, welopen)
    targets = _records(
        control.get("WELTARG", control.get("WELLTARG", control.get("weltarg", control.get("welltarg"))))
    )
    _apply_weltarg(records_by_kw, targets)

    for keyword in _PRODUCER_CONTROL_KEYWORDS:
        for record in records_by_kw.get(keyword, []):
            name = _name(_field(record, ("well", "name", "wname"), 0, ""))
            w = by_name.get(name)
            if w is None:
                continue
            raw_mode = _name(_field(record, ("control", "type", "mode"), 2, "bhp")).lower()
            if keyword == "WCONHIST" and _full_wconhist_record(record) and raw_mode == "resv":
                mode = "resv_history"
            else:
                mode = _control_mode(raw_mode)
            lims = _prod_lims(record, keyword=keyword)
            w.sign = -1
            w.status = _status(_field(record, ("status", "open"), 1, "OPEN"))
            w.lims = lims
            w.type = mode
            w.val = float(lims.get(mode, 0.0))
            if mode in {"orat", "wrat", "grat", "lrat", "vrat", "resv_history"} and w.val > 0.0:
                w.val = -w.val
            if mode == "resv_history":
                w.val, w.compi = _wconhist_resv_target_and_compi(record, np.asarray(w.compi).size)
            elif keyword == "WCONHIST":
                compi = _wconhist_compi(record, mode, np.asarray(w.compi).size)
                if compi is not None:
                    w.compi = compi
            elif keyword == "WCONPROD":
                compi = _wconprod_compi(mode, np.asarray(w.compi).size)
                if compi is not None:
                    w.compi = compi
                w.vfp_index = _vfp_index(record, keyword=keyword)

    for keyword in _INJECTOR_CONTROL_KEYWORDS:
        for record in records_by_kw.get(keyword, []):
            name = _name(_field(record, ("well", "name", "wname"), 0, ""))
            w = by_name.get(name)
            if w is None:
                continue
            if keyword == "WCONINJH" and _wconinjh_history_rate_record(record):
                mode = "rate"
            else:
                mode = _control_mode(_field(record, ("control", "type", "mode"), 3, "rate"))
            lims = _inje_lims(record, keyword=keyword)
            w.sign = 1
            w.status = _status(_field(record, ("status", "open"), 2, "OPEN"))
            w.lims = lims
            w.type = mode
            w.val = float(lims.get(mode, 0.0))
            w.compi = _phase_compi(_field(record, ("phase", "fluid"), 1, ""), np.asarray(w.compi).size)
            if keyword in {"WCONINJ", "WCONINJE"}:
                w.vfp_index = _vfp_index(record, keyword=keyword)

    _apply_completion_welopen(out, welopen, grid=grid)

    return out


def _allocation_from_wgrupcon(
    records: Sequence[Any],
    wells: Sequence[str],
) -> tuple[str, list[float] | None, dict[str, Any] | None]:
    if not records:
        return "rate", None, None
    guide_by_well: dict[str, float] = {}
    guide_records: list[dict[str, Any]] = []
    for record in records:
        metadata = _wgrupcon_metadata(record)
        if not _wgrupcon_status(record):
            guide_records.append({
                "member": str(metadata.get("well", "")),
                "well": str(metadata.get("well", "")),
                "guide_rate": float(metadata.get("guide_rate", -1.0)),
                "guide_multiplier": float(metadata.get("guide_multiplier", 1.0)),
                "effective_guide_rate": float(metadata.get("effective_guide_rate", 0.0)),
                "used": False,
                "status": False,
            })
            continue
        wname = str(metadata.get("well", ""))
        guide = float(metadata.get("effective_guide_rate", 0.0))
        used = bool(wname and guide > 0.0)
        guide_records.append({
            "member": wname,
            "well": wname,
            "guide_rate": float(metadata.get("guide_rate", -1.0)),
            "guide_multiplier": float(metadata.get("guide_multiplier", 1.0)),
            "effective_guide_rate": guide,
            "used": used,
            "status": True,
        })
        if wname and guide > 0.0:
            guide_by_well[wname] = guide
    if not guide_by_well:
        return "rate", None, None
    fractions = [max(float(guide_by_well.get(w, 0.0)), 0.0) for w in wells]
    return "fraction", fractions, {
        "source_keyword": "WGRUPCON",
        "allocation": "fraction",
        "members": guide_records,
        "fractions": list(fractions),
    }


def _allocation_from_grupnet_guides(
    members: Sequence[str],
    extra_by_group: Mapping[str, dict[str, Any]],
) -> tuple[str, list[float] | None, dict[str, Any] | None]:
    """Use imported ``GRUPNET`` child guide rates as parent fractions.

    This is a conservative native counterpart to MRST's richer network guide
    rate machinery: if a parent group points at child groups that carry finite
    positive ``GRUPNET`` guide rates with network use enabled, those guide rates
    become explicit parent allocation fractions. Direct wells and child groups
    without guide rates receive zero before the usual active-member
    renormalisation in the model.
    """
    if not members:
        return "rate", None, None
    fractions: list[float] = []
    guided_members: list[dict[str, Any]] = []
    any_guided = False
    for member in members:
        group_name = None
        if isinstance(member, str) and member.startswith("group:"):
            group_name = member.split(":", 1)[1]
        elif isinstance(member, str) and member in extra_by_group:
            group_name = member
        metadata = extra_by_group.get(group_name or "", {}).get("GRUPNET", {})
        guide = 0.0
        if isinstance(metadata, Mapping) and metadata.get("use_network", False):
            guide = _float(metadata.get("guide_rate"), -1.0)
        guided_members.append(
            {
                "member": str(member),
                "group": group_name,
                "guide_rate": float(guide),
                "used": bool(guide > 0.0),
                "terminal": metadata.get("terminal") if isinstance(metadata, Mapping) else None,
                "use_vfp": bool(metadata.get("use_vfp", False)) if isinstance(metadata, Mapping) else False,
                "min_vfp_table": metadata.get("min_vfp_table") if isinstance(metadata, Mapping) else None,
                "max_vfp_table": metadata.get("max_vfp_table") if isinstance(metadata, Mapping) else None,
            }
        )
        if guide > 0.0:
            any_guided = True
            fractions.append(float(guide))
        else:
            fractions.append(0.0)
    if not any_guided:
        return "rate", None, None
    return "fraction", fractions, {
        "source_keyword": "GRUPNET",
        "allocation": "fraction",
        "members": guided_members,
        "fractions": list(fractions),
    }


def facility_group_controls_from_deck_control(control: Mapping[str, Any]) -> list[FacilityGroupControl]:
    """Create ``FacilityGroupControl`` objects from one MRST schedule control.

    Supported record fields are intentionally the MRST deck/schedule names:
    ``GRUPTREE`` defines child -> parent group edges; ``WELSPECS`` and
    ``WGRUPCON`` define well-to-group membership and optional guide rates; and
    ``GCONPROD``/``GCONINJE`` define group limits. ``GECON`` and ``GRUPNET``
    are preserved as ``FacilityGroupControl.extra`` metadata so exact deck
    ports do not lose economics/network rows before full enforcement exists.
    """
    control = _struct_to_mapping(control)
    if not isinstance(control, Mapping):
        raise TypeError("deck group control must be a mapping or MATLAB-style struct")
    gruptree = _records(control.get("GRUPTREE", control.get("gruptree")))
    gconprod = _records(control.get("GCONPROD", control.get("gconprod")))
    gconinje = _replace_duplicate_gconinje(_records(control.get("GCONINJE", control.get("gconinje"))))
    gecon = _replace_duplicate_by_group(_records(control.get("GECON", control.get("gecon"))))
    grupnet = _replace_duplicate_by_group(_records(control.get("GRUPNET", control.get("grupnet"))))
    welspecs = _records(control.get("WELSPECS", control.get("welspecs")))
    wgrupcon = _records(control.get("WGRUPCON", control.get("wgrupcon")))

    parent_by_child: dict[str, str] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    groups: set[str] = set()
    for record in gruptree:
        child = _name(_field(record, ("child", "group", "name"), 0, ""))
        parent = _name(_field(record, ("parent", "parent_group"), 1, ""))
        if not child or not parent:
            continue
        parent_by_child[child] = parent
        children_by_parent[parent].append(child)
        groups.update((child, parent))

    wells_by_group: dict[str, list[str]] = defaultdict(list)
    for record in welspecs:
        well = _name(_field(record, ("well", "name", "wname"), 0, ""))
        group = _name(_field(record, ("group", "group_name", "grp"), 1, ""))
        if well and group:
            wells_by_group[group].append(well)
            groups.add(group)

    wgrup_by_group: dict[str, list[Any]] = defaultdict(list)
    extra_by_group: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in wgrupcon:
        well = _name(_field(record, ("well", "name", "wname"), 0, ""))
        group = _name(_field(record, ("group", "group_name", "grp"), 3, ""))
        if group:
            if well and well not in wells_by_group[group]:
                wells_by_group[group].append(well)
            wgrup_by_group[group].append(record)
            extra_by_group[group].setdefault("WGRUPCON", []).append(_wgrupcon_metadata(record))
            groups.add(group)

    lims_by_group: dict[str, dict[str, float]] = defaultdict(dict)
    status_by_group: dict[str, bool] = {}
    for record in gconprod:
        group = _name(_field(record, ("group", "name", "gname"), 0, ""))
        if not group:
            continue
        groups.add(group)
        lims_by_group[group].update(_gconprod_limit(record))
        status_by_group[group] = _status(_field(record, ("status", "open"), 7, "YES"))
        extra_by_group[group].setdefault("GCONPROD", []).append(_gconprod_metadata(record))
    for record in gconinje:
        group = _name(_field(record, ("group", "name", "gname"), 0, ""))
        if not group:
            continue
        groups.add(group)
        lims_by_group[group].update(_gconinje_limit(record))
        status_by_group[group] = _status(_field(record, ("status", "open"), 7, "YES"))
        extra_by_group[group].setdefault("GCONINJE", []).append(_gconinje_metadata(record))

    for record in gecon:
        group = _name(_field(record, ("group", "name", "gname"), 0, ""))
        if not group:
            continue
        groups.add(group)
        extra_by_group[group]["GECON"] = _gecon_metadata(record)
    for record in grupnet:
        group = _name(_field(record, ("group", "name", "gname"), 0, ""))
        if not group:
            continue
        groups.add(group)
        extra_by_group[group]["GRUPNET"] = _grupnet_metadata(record)

    def depth(group: str, seen: tuple[str, ...] = ()) -> int:
        if group in seen:
            cycle = " -> ".join(seen + (group,))
            raise ValueError(f"cyclic deck group tree: {cycle}")
        kids = children_by_parent.get(group, [])
        if not kids:
            return 0
        return 1 + max(depth(kid, seen + (group,)) for kid in kids)

    out: list[FacilityGroupControl] = []
    for group in sorted(groups, key=lambda g: (depth(g), g)):
        members: list[str] = []
        members.extend(wells_by_group.get(group, []))
        members.extend(f"group:{child}" for child in children_by_parent.get(group, []))
        if (
            not members
            and group in parent_by_child
            and group not in lims_by_group
            and group not in extra_by_group
        ):
            continue
        allocation, fractions, wgrupcon_allocation = _allocation_from_wgrupcon(
            wgrup_by_group.get(group, []), members
        )
        group_extra = dict(extra_by_group.get(group, {}))
        if wgrupcon_allocation is not None:
            group_extra["WGRUPCON_ALLOCATION"] = wgrupcon_allocation
        if fractions is None:
            allocation, fractions, grupnet_allocation = _allocation_from_grupnet_guides(
                members, extra_by_group
            )
            if grupnet_allocation is not None:
                group_extra["GRUPNET_ALLOCATION"] = grupnet_allocation
        out.append(
            FacilityGroupControl(
                group,
                wells=members,
                lims=dict(lims_by_group.get(group, {})),
                status=status_by_group.get(group, True),
                allocation=allocation,
                fractions=fractions,
                extra=group_extra,
            )
        )
    return out


def facility_group_controls_from_deck_schedule(schedule: Mapping[str, Any]) -> list[list[FacilityGroupControl]]:
    """Convert every control block in a MRST-like schedule mapping."""
    schedule = _struct_to_mapping(schedule)
    if not isinstance(schedule, Mapping):
        raise TypeError("deck schedule must be a mapping or MATLAB-style struct")
    controls = schedule.get("control", schedule.get("CONTROL", []))
    return [facility_group_controls_from_deck_control(ctrl) for ctrl in _records(controls)]


# MATLAB-style aliases
applyDeckWellControls = apply_deck_well_controls
facilityGroupControlsFromDeckControl = facility_group_controls_from_deck_control
facilityGroupControlsFromDeckSchedule = facility_group_controls_from_deck_schedule
wellsFromDeckCompletionRecords = wells_from_deck_completion_records


__all__ = [
    "applyDeckWellControls",
    "apply_deck_well_controls",
    "facilityGroupControlsFromDeckControl",
    "facilityGroupControlsFromDeckSchedule",
    "facility_group_controls_from_deck_control",
    "facility_group_controls_from_deck_schedule",
    "wellsFromDeckCompletionRecords",
    "wells_from_deck_completion_records",
]
