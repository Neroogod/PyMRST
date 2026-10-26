"""MRST ``spe10/getSPE10_model_1_relperm.m``: the official Model I relative
permeabilities as a reduced ECLIPSE-style deck whose `PROPS.SGOF` holds the
benchmark table, read from the dataset's `model1_data.mat`.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pymrst.modules.spe10.get_spe10_model_1_rock import _model1_data


def _struct(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return {name: _struct(getattr(value, name)) for name in value._fieldnames}
    return value


def get_spe10_model_1_relperm() -> Dict[str, Any]:
    """MRST `kr_deck = getSPE10_model_1_relperm()`; `PROPS.SGOF` is a list of
    tables, as MATLAB's cell array."""
    kr_deck = _struct(_model1_data()["kr_deck"])
    sgof = kr_deck["PROPS"]["SGOF"]
    tables = [sgof] if np.ndim(sgof) == 2 and np.asarray(sgof).dtype != object else list(sgof)
    kr_deck["PROPS"]["SGOF"] = [np.asarray(t, dtype=float) for t in tables]
    return kr_deck


getSPE10_model_1_relperm = get_spe10_model_1_relperm
__all__ = ["getSPE10_model_1_relperm", "get_spe10_model_1_relperm"]


if __name__ == "__main__":
    def test_model_one_sgof():
        kr_deck = get_spe10_model_1_relperm()
        sgof = kr_deck["PROPS"]["SGOF"]
        assert len(sgof) == 1 and sgof[0].shape == (35, 4)
        assert sgof[0][0, 1] == 0 and sgof[0][-1, 2] == 0
        assert {"GRID", "PROPS", "REGIONS"} <= set(kr_deck)
        print("ok test_model_one_sgof")

    test_model_one_sgof()
    print("\nAll tests passed!")
