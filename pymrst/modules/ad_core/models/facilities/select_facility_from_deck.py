"""MRST ``ad-core/models/facilities/selectFacilityFromDeck.m``.

`select_facility_from_deck` is the port, for the AD-OO models in
`ad_core.ad_oo`: the model's facility (a `GenericFacilityModel` for the
generic black-oil model), given the deck's VFP tables.
`select_facility_from_deck_ad` is PyMRST's earlier version for the legacy
`FacilityModel` (wells, group controls), kept for its callers.
"""

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


def select_facility_from_deck_ad(
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


def select_facility_from_deck(deck, model):
    """MRST `selectFacilityFromDeck`: the model's facility, with VFP tables.

    MRST keeps an injector table only where the list already has that slot,
    and the list starts empty, so injector tables are never stored; producer
    tables are stored and the list grows to fit. Both are reproduced.
    """
    facility = model.FacilityModel
    if facility is None:
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            GenericBlackOilModel,
        )
        from pymrst.modules.ad_core.ad_oo.facility_model import (
            FacilityModel as AdOoFacilityModel,
        )
        from pymrst.modules.ad_core.ad_oo.generic_facility_model import (
            GenericFacilityModel,
        )
        facility = (GenericFacilityModel(model) if isinstance(model, GenericBlackOilModel)
                    else AdOoFacilityModel(model))
    controls = _controls_from_deck(deck)
    if not controls:
        return facility
    runspec = _get(deck, "RUNSPEC", default={})
    first = controls[0]
    if (_get(runspec, "VFPIDIMS") is not None or _get(first, "VFPPROD") is not None
            or _get(first, "VFPINJ") is not None):
        facility.VFPTablesInjector = []
        facility.VFPTablesProducer = []
        has_warned = False
        for control in controls:
            for tno, table in enumerate(_table_list(_get(control, "VFPINJ"))):
                if _is_empty(table) or len(facility.VFPTablesInjector) <= tno:
                    continue
                if not _is_empty(facility.VFPTablesInjector[tno]) and not has_warned:
                    warnings.warn("MRST does not presently support overwriting VFP tables")
                    has_warned = True
                facility.VFPTablesInjector[tno] = _as_vfp_table(table)
            for tno, table in enumerate(_table_list(_get(control, "VFPPROD"))):
                if _is_empty(table):
                    continue
                tables = facility.VFPTablesProducer
                if len(tables) > tno and not _is_empty(tables[tno]) and not has_warned:
                    warnings.warn("MRST does not presently support overwriting VFP tables")
                    has_warned = True
                tables.extend([None]*(tno + 1 - len(tables)))
                tables[tno] = _as_vfp_table(table)
    return facility


def _table_list(records) -> list:
    """A control's VFPINJ/VFPPROD cell array as a list (table `k` at `k-1`)."""
    if records is None:
        return []
    if isinstance(records, VFPTable) or _is_vfp_deck_table(records):
        return [records]
    if isinstance(records, Mapping):
        out: list[Any] = []
        for key, table in records.items():
            out.extend([None]*(int(key) - len(out)))
            out[int(key) - 1] = table
        return out
    return list(records)


def _is_empty(item) -> bool:
    """MATLAB `isempty` for a table slot."""
    if item is None:
        return True
    if isinstance(item, (Sequence, Mapping)) and not isinstance(item, str):
        return len(item) == 0
    return False


selectFacilityFromDeck = select_facility_from_deck


__all__ = [
    "selectFacilityFromDeck",
    "group_controls_from_deck",
    "select_facility_from_deck",
    "select_facility_from_deck_ad",
    "vfp_tables_from_deck",
]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _table():
        return object.__new__(VFPTable)

    def _deck(*controls, runspec=None):
        return {"RUNSPEC": runspec or {}, "SCHEDULE": {"control": list(controls)}}

    def test_the_models_facility_is_kept():
        facility = SimpleNamespace(VFPTablesInjector=["kept"], VFPTablesProducer=[])
        model = SimpleNamespace(FacilityModel=facility)
        assert select_facility_from_deck(_deck({"WCONPROD": []}), model) is facility
        assert facility.VFPTablesInjector == ["kept"], "no VFP keywords: untouched"
        print("ok test_the_models_facility_is_kept")

    def test_producer_tables_are_stored_at_their_numbers():
        facility = SimpleNamespace(VFPTablesInjector=[], VFPTablesProducer=[])
        model = SimpleNamespace(FacilityModel=facility)
        second = _table()
        select_facility_from_deck(_deck({"VFPPROD": [None, second]}), model)
        assert facility.VFPTablesProducer == [None, second]
        print("ok test_producer_tables_are_stored_at_their_numbers")

    def test_injector_tables_are_never_stored_as_in_mrst():
        """MRST stores an injector table only into an existing slot, and the
        list starts empty."""
        facility = SimpleNamespace(VFPTablesInjector=[], VFPTablesProducer=[])
        model = SimpleNamespace(FacilityModel=facility)
        select_facility_from_deck(_deck({"VFPINJ": [_table()]}), model)
        assert facility.VFPTablesInjector == []
        print("ok test_injector_tables_are_never_stored_as_in_mrst")

    test_the_models_facility_is_kept()
    test_producer_tables_are_stored_at_their_numbers()
    test_injector_tables_are_never_stored_as_in_mrst()
    print("\nAll tests passed!")
