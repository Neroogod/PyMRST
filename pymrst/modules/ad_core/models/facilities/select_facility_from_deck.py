"""MRST ``ad-core/models/facilities/selectFacilityFromDeck.m``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.facilities.vfp import VFPTable


def _get(obj, *names, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        lower = {str(k).lower(): v for k, v in obj.items()}
        for name in names:
            key = str(name).lower()
            if key in lower:
                return lower[key]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _controls_from_deck(deck) -> list:
    schedule = _get(deck, "SCHEDULE", "schedule", default={})
    controls = _get(schedule, "control", "controls", default=[])
    if controls is None:
        return []
    if isinstance(controls, Mapping):
        return [controls]
    return list(controls)


def _is_vfp_deck_table(obj) -> bool:
    if not isinstance(obj, Mapping):
        return False
    keys = {str(k).upper() for k in obj.keys()}
    return "FLO" in keys and "THP" in keys and ("Q" in keys or "BHP" in keys)


def _as_vfp_table(obj) -> VFPTable:
    if isinstance(obj, VFPTable):
        return obj
    return VFPTable.from_deck(obj)


def _iter_numbered_tables(records):
    if records is None:
        return
    if isinstance(records, VFPTable) or _is_vfp_deck_table(records):
        yield 1, records
        return
    if isinstance(records, Mapping):
        for key, table in records.items():
            if table is None:
                continue
            if isinstance(table, Sequence) and not isinstance(table, (str, bytes, Mapping)) and len(table) == 0:
                continue
            yield int(key), table
        return
    for table_no, table in enumerate(list(records), start=1):
        if table is None:
            continue
        if isinstance(table, Sequence) and not isinstance(table, (str, bytes, Mapping)) and len(table) == 0:
            continue
        yield table_no, table


def _set_table(tables: list, table_no: int, table, *, warned: list[bool]) -> None:
    if table_no <= 0:
        raise ValueError("VFP table numbers are one-based and must be positive")
    while len(tables) < table_no:
        tables.append(None)
    if tables[table_no - 1] is not None and not warned[0]:
        warnings.warn(
            "MRST does not presently support overwriting VFP tables",
            RuntimeWarning,
            stacklevel=3,
        )
        warned[0] = True
    tables[table_no - 1] = _as_vfp_table(table)


def vfp_tables_from_deck(deck) -> tuple[list, list]:
    """Collect injector and producer VFP tables from all deck controls.

    The output lists preserve MRST's one-based table numbering with ``None``
    placeholders for missing table numbers.
    """
    injector: list[Any] = []
    producer: list[Any] = []
    warned = [False]
    for control in _controls_from_deck(deck):
        for table_no, table in _iter_numbered_tables(_get(control, "VFPINJ", "vfpinj", default=None)) or ():
            _set_table(injector, table_no, table, warned=warned)
        for table_no, table in _iter_numbered_tables(_get(control, "VFPPROD", "vfpprod", default=None)) or ():
            _set_table(producer, table_no, table, warned=warned)
    return injector, producer


def group_controls_from_deck(deck) -> list[list[Any]]:
    """Collect MRST schedule group controls for native facilities."""
    schedule = _get(deck, "SCHEDULE", "schedule", default={})
    try:
        from pymrst.modules.compositional.facility import (
            facility_group_controls_from_deck_schedule,
        )
    except Exception:
        return []
    try:
        return facility_group_controls_from_deck_schedule(schedule)
    except Exception:
        return []


def _existing_facility(model):
    return _get(model, "FacilityModel", "facility_model", "facility", default=None)


def _model_wells(model):
    return list(_get(model, "wells", "W", default=[]))


def _attach_facility(model, facility: FacilityModel) -> None:
    if model is None:
        return
    for name in ("FacilityModel", "facility_model", "facility"):
        try:
            setattr(model, name, facility)
            return
        except Exception:
            continue


def select_facility_from_deck(
    deck,
    model=None,
    *,
    wells=None,
    disgas: bool = False,
    verbose: bool = False,
    vapoil: bool = False,
) -> FacilityModel:
    """Pick and configure a native facility model from a deck.

    This mirrors MRST's ``selectFacilityFromDeck`` for PyMRST's native
    ``FacilityModel``: reuse an existing facility when the model has one,
    otherwise create one from the model wells (or the explicit ``wells``
    argument), then import any ``VFPINJ``/``VFPPROD`` tables from schedule
    controls and rebind wells with positive ``vfp_index`` values.
    """
    injector, producer = vfp_tables_from_deck(deck)
    group_schedule = group_controls_from_deck(deck)
    first_groups = group_schedule[0] if group_schedule else []
    facility = _existing_facility(model)
    if facility is None:
        facility = FacilityModel(
            _model_wells(model) if wells is None else list(wells),
            disgas=disgas,
            verbose=verbose,
            vapoil=vapoil,
            vfp_tables_injector=injector,
            vfp_tables_producer=producer,
            group_controls=first_groups,
        )
        _attach_facility(model, facility)
    elif injector or producer:
        facility.vfp_tables_injector = injector
        facility.vfp_tables_producer = producer
        facility._bind_vfp_tables()
    if group_schedule:
        facility.group_control_schedule = group_schedule
        facility.group_controls = list(first_groups)
    return facility


selectFacilityFromDeck = select_facility_from_deck


__all__ = [
    "selectFacilityFromDeck",
    "group_controls_from_deck",
    "select_facility_from_deck",
    "vfp_tables_from_deck",
]
