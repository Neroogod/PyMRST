"""Combine regular and multisegment wells with a uniform field layout."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List


def _fields(item: Any) -> set[str]:
    if isinstance(item, dict):
        return set(item)
    if hasattr(item, "__dict__"):
        return set(vars(item))
    slots = getattr(type(item), "__slots__", ())
    return {name for name in slots if hasattr(item, name)}


def _get(item: Any, name: str):
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


def _normalized(item: Any, fields: set[str], is_ms: bool):
    item_fields = _fields(item)
    values = {name: (_get(item, name) if name in item_fields else None)
              for name in fields}
    values["isMS"] = bool(is_ms)
    if isinstance(item, dict):
        return values
    return SimpleNamespace(**values)


def _as_list(wells: Iterable[Any] | Any) -> List[Any]:
    if wells is None:
        return []
    if isinstance(wells, (dict, SimpleNamespace)) or hasattr(wells, "cells"):
        return [wells]
    return list(wells)


def combine_ms_with_regular_wells(
    regular_wells: Iterable[Any] | Any,
    multisegment_wells: Iterable[Any] | Any,
) -> List[Any]:
    """Return regular wells followed by MS wells with the union of all fields.

    Missing MATLAB struct fields are represented by ``None`` (the Python
    equivalent of MRST's empty value). Inputs are copied and never mutated.
    """
    regular = _as_list(regular_wells)
    ms = _as_list(multisegment_wells)
    all_fields: set[str] = set()
    for well in regular + ms:
        all_fields.update(_fields(well))
    all_fields.discard("isMS")
    return ([_normalized(w, all_fields, False) for w in regular]
            + [_normalized(w, all_fields, True) for w in ms])


combineMSwithRegularWells = combine_ms_with_regular_wells

__all__ = ["combine_ms_with_regular_wells", "combineMSwithRegularWells"]
