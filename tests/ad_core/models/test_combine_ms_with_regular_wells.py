from types import SimpleNamespace

from pymrst.modules.ad_core.models.facilities import (
    combine_ms_with_regular_wells_ad,
)


def test_combines_and_normalizes_heterogeneous_fields_without_mutation():
    regular = SimpleNamespace(name="R1", cells=[0], WI=[1.0])
    ms = SimpleNamespace(
        name="MS1", cells=[1, 2], nodes={"depth": [0.0, 10.0]},
        segments={"topo": [[0, 1]]},
    )
    out = combine_ms_with_regular_wells_ad(regular, ms)
    assert [w.name for w in out] == ["R1", "MS1"]
    assert [w.isMS for w in out] == [False, True]
    assert out[0].nodes is None and out[0].segments is None
    assert out[1].WI is None
    assert not hasattr(regular, "isMS")
    assert not hasattr(ms, "WI")


def test_dict_inputs_stay_dicts_and_empty_families_work():
    out = combine_ms_with_regular_wells_ad(
        [{"name": "R", "cells": [0]}],
        [{"name": "M", "cells": [1], "topo": [[0, 1]]}],
    )
    assert all(isinstance(w, dict) for w in out)
    assert out[0]["topo"] is None
    assert out[1]["isMS"]
    assert combine_ms_with_regular_wells_ad([], []) == []
