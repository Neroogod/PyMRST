"""MRST ``ad-core/models/facilities/combineMSwithRegularWells.m``.

One well list from regular and multisegment wells: every well gets every
field any of them has (missing ones empty, `None` here, as MATLAB's struct
arrays fill them with `[]`), the regular wells come first, and `isMS` marks
the rest. The wells keep their own type; the inputs are not changed.

`combine_ms_with_regular_wells_ad`, below, is PyMRST's earlier version (it
rebuilds the wells as namespaces or dicts).
"""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any, Iterable, List


def combine_ms_with_regular_wells(W_regular: Any, W_ms: Any) -> List[Any]:
    """MRST `W = combineMSwithRegularWells(W_regular, W_ms)`."""
    W_regular = [copy(w) for w in _as_list(W_regular)]
    W_ms = [copy(w) for w in _as_list(W_ms)]
    allflds = sorted(set().union(*[_fields(w) for w in W_ms + W_regular]))
    for fld in allflds:
        for w in W_regular + W_ms:
            if not hasattr(w, fld):
                setattr(w, fld, None)
    W = W_regular + W_ms
    for i, w in enumerate(W):
        w.isMS = i >= len(W_regular)
    return W


combineMSwithRegularWells = combine_ms_with_regular_wells


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


def combine_ms_with_regular_wells_ad(
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


__all__ = ["combine_ms_with_regular_wells", "combine_ms_with_regular_wells_ad",
           "combineMSwithRegularWells"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.params.wells_and_bc.types import Well

    def _well(name: str, cells) -> Well:
        return Well(cells=np.asarray(cells), type="rate", val=1.0, r=np.array([0.1]),
                    dir="z", WI=np.ones(len(cells)), dZ=np.zeros(len(cells)), name=name,
                    compi=np.array([1.0, 0, 0]), refDepth=0.0)

    def test_regular_first_with_the_union_of_fields():
        ms = _well("MS", [2, 3])
        ms.nodes = SimpleNamespace(depth=np.array([0.0, 1.0]))
        out = combine_ms_with_regular_wells(_well("R", [0]), ms)
        assert [w.name for w in out] == ["R", "MS"]
        assert [w.isMS for w in out] == [False, True]
        assert out[0].nodes is None and out[1].nodes is ms.nodes
        assert type(out[0]) is Well
        print("ok test_regular_first_with_the_union_of_fields")

    def test_the_inputs_are_not_changed():
        regular, ms = _well("R", [0]), SimpleNamespace(name="MS", cells=[1], segments=1)
        combine_ms_with_regular_wells([regular], [ms])
        assert not hasattr(regular, "isMS") and not hasattr(regular, "segments")
        assert not hasattr(ms, "WI")
        print("ok test_the_inputs_are_not_changed")

    test_regular_first_with_the_union_of_fields()
    test_the_inputs_are_not_changed()
    print("\nAll tests passed!")
