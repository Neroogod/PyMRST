"""MRST ``ad-core/initialization/getMinMaxPhaseSaturations.m``.

Per cell and phase, the lowest and highest saturation for equilibration:
from the rock's drainage end-point scaling (`rock.krscale`) when there is
one, else from the fluid's table end points (`krPts`, by saturation
region), else from where the relative permeabilities start and stop
changing. Gas may vanish (`s_min = 0`); water is held at connate
saturation (`fluid.sWcon`, or `rock.sw` with end-point scaling) and may fill
a cell, and gas may then fill only what connate water leaves.

`get_min_max_phase_saturations_ad` (in `regions.py`) is PyMRST's earlier
version. `satnum` is MRST's 1-based region number; `cells` are 0-based.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def get_min_max_phase_saturations(model: Any, satnum: Any = 1,
                                  cells: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `[s_min, s_max] = getMinMaxPhaseSaturations(model, satnum, cells)`."""
    satnum = np.atleast_1d(np.asarray(satnum, dtype=np.int64))
    cells = None if cells is None else np.atleast_1d(np.asarray(cells, dtype=np.int64))
    nc = satnum.size
    rock = model.rock
    has_endpoints = _present(rock, "krscale")
    fluid = model.fluid
    if has_endpoints:
        assert cells is not None and cells.size
        pts = _get(_get(rock, "krscale"), "drainage")
        s_min, s_max = _get_bounds(model, pts, cells)
    else:
        if _present(fluid, "krPts"):
            s_min, s_max = _get_bounds(model, _get(fluid, "krPts"), satnum - 1)
        else:
            from pymrst.modules.ad_core.initialization.get_min_max_phase_saturations_from_rel_perm import (
                get_min_max_phase_saturations_from_rel_perm,
            )
            from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
            model = value_copy_model(model).validate_model()
            relperm = model.FlowPropertyFunctions.get_state_function("RelativePermeability")
            regions = getattr(relperm, "regions", None)
            sat_reg = 1 if regions is None else int(np.asarray(regions)[cells[0]])
            s_min, s_max = get_min_max_phase_saturations_from_rel_perm(model, 1e-6, sat_reg)
        s_min, s_max = np.atleast_2d(s_min).astype(float), np.atleast_2d(s_max).astype(float)
        if s_min.shape[0] == 1:
            s_min = np.repeat(s_min, nc, axis=0)
        if s_max.shape[0] == 1:
            s_max = np.repeat(s_max, nc, axis=0)
    wix = _index(model, "W") if model.water else None
    gix = _index(model, "G") if model.gas else None
    if model.gas:
        s_min[:, gix] = 0
    if model.water:
        if _present(fluid, "sWcon") and not has_endpoints:
            swcon = np.atleast_1d(np.asarray(_get(fluid, "sWcon"), dtype=float))
            s_min[:, wix] = swcon[0] if swcon.size == 1 else swcon[satnum - 1]
        if _present(rock, "sw"):
            assert cells is not None and cells.size
            assert has_endpoints, "Requires endpoint scaling"
            s_min[:, wix] = np.asarray(_get(rock, "sw"), dtype=float)[cells]
        s_max[:, wix] = 1
        if model.gas:
            s_max[:, gix] = np.minimum(s_max[:, gix], 1 - s_min[:, wix])
    return s_min, s_max


def _get_bounds(model: Any, pts: Any, subs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    phases = model.get_phase_names()
    nph = len(phases)
    subs = np.atleast_1d(subs)
    s_min, s_max = np.ones((subs.size, nph)), np.ones((subs.size, nph))
    lo, hi = 0, 2                                       # MRST's min_sub = 1, max_sub = 3
    for i, letter in enumerate(phases):
        ph = letter.lower()
        if ph == "o" and not _present(pts, "o"):
            ow = np.atleast_2d(_get(pts, "ow")) if model.water else None
            og = np.atleast_2d(_get(pts, "og")) if model.gas else None
            if model.water and model.gas:
                s_min[:, i] = np.minimum(ow[subs, lo], og[subs, lo])
                s_max[:, i] = np.maximum(ow[subs, hi], og[subs, hi])
            elif model.water:
                s_min[:, i], s_max[:, i] = ow[subs, lo], ow[subs, hi]
            elif model.gas:
                s_min[:, i], s_max[:, i] = og[subs, lo], og[subs, hi]
        else:
            table = np.atleast_2d(_get(pts, ph))
            s_min[:, i], s_max[:, i] = table[subs, lo], table[subs, hi]
    return s_min, s_max


def _index(model: Any, phase: str) -> int:
    return int(model.get_phase_index(phase))


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _present(obj: Any, name: str) -> bool:
    """MATLAB `isfield`: here, present and not empty."""
    value = _get(obj, name)
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple)) or hasattr(value, "__dict__"):
        return bool(value) if isinstance(value, (dict, list, tuple)) else True
    return np.size(value) > 0


getMinMaxPhaseSaturations = get_min_max_phase_saturations

__all__ = ["getMinMaxPhaseSaturations", "get_min_max_phase_saturations"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _model(pts, sWcon=None):
        fluid = {"krPts": pts}
        if sWcon is not None:
            fluid["sWcon"] = sWcon
        return SimpleNamespace(water=True, oil=True, gas=False,
                               rock=SimpleNamespace(krscale=None, sw=None), fluid=fluid,
                               get_phase_names=lambda: ["W", "O"],
                               get_phase_index=lambda p: {"W": 0, "O": 1}[p])

    def test_table_end_points_bound_the_saturations():
        pts = {"w": np.array([[0.1, 0.2, 0.9, 0.7]]), "ow": np.array([[0.0, 0.15, 1.0, 0.8]])}
        s_min, s_max = get_min_max_phase_saturations(_model(pts), 1, np.arange(3))
        np.testing.assert_allclose(s_min, [[0.1, 0.0]]*1)
        np.testing.assert_allclose(s_max, [[1.0, 1.0]])
        print("ok test_table_end_points_bound_the_saturations")

    def test_connate_water_from_the_fluid():
        pts = {"w": np.array([[0.1, 0.2, 0.9, 0.7]]), "ow": np.array([[0.0, 0.15, 1.0, 0.8]])}
        s_min, _ = get_min_max_phase_saturations(_model(pts, sWcon=0.25), np.array([1, 1]))
        np.testing.assert_allclose(s_min[:, 0], [0.25, 0.25])
        print("ok test_connate_water_from_the_fluid")

    test_table_end_points_bound_the_saturations()
    test_connate_water_from_the_fluid()
    print("\nAll tests passed!")
