"""MRST ``ad-core/models/wells/updateSwitchedWellControls.m``: control
switching for the (deprecated) classic well model.

For each open well with limits, the first limit it breaks - other than the
control it is on - becomes its control: an injector checks `bhp` (above),
`rate` (above) and `vrat`; a producer `bhp` (below), `orat`, `lrat`, `grat`,
`wrat` (below, rates negative) and `vrat`, missing limits counting as
`-inf`. The well solution's `type` and `val` are switched and the change is
printed. Numeric limits (which must be infinite) never switch. Returns
`(sol, withinLims)`, one flag per well.
"""

from __future__ import annotations

import copy
from typing import Any, List, Sequence, Tuple

import numpy as np


def update_switched_well_controls(wellmodel: Any, model: Any, sol: Sequence[Any], pBH: Any,
                                  q_s: Sequence[Any]) -> Tuple[List[Any], np.ndarray]:
    """MRST `[sol, withinLims] = updateSwitchedWellControls(wellmodel, model, sol, pBH, q_s)`."""
    from pymrst.utils.adi import value
    W = list(wellmodel.W or [])
    sol = [copy.copy(s) for s in sol]
    if not W or _empty(_get(W[0], "lims")):
        return sol, np.ones(len(sol), dtype=bool)
    nwells = len(sol)
    within = np.ones(nwells, dtype=bool)
    pBH = np.asarray(value(pBH), dtype=float).ravel()
    q = np.column_stack([np.asarray(value(x), dtype=float).ravel() for x in q_s])
    for wnr in range(nwells):
        status = _get(W[wnr], "status")
        if status is not None and not status:
            continue
        lims = _get(W[wnr], "lims")
        pBHw = pBH[wnr]
        q_sw = q[wnr]
        qt_sw = q_sw.sum()
        if not isinstance(lims, (int, float, np.ndarray)):
            lims = dict(lims) if isinstance(lims, dict) else dict(vars(lims))
            lims["vrat"] = -np.inf
            if _get(sol[wnr], "sign") > 0:
                modes = ["bhp", "rate", "rate"]
                flags = [pBHw > lims["bhp"], qt_sw > lims["rate"], qt_sw < lims["vrat"]]
            else:
                modes = ["bhp", "orat", "lrat", "grat", "wrat", "vrat"]
                for f in modes:
                    lims.setdefault(f, -np.inf)
                flags = [pBHw < lims["bhp"], q_sw[1] < lims["orat"],
                         q_sw[0] + q_sw[1] < lims["lrat"], q_sw[2] < lims["grat"],
                         q_sw[0] < lims["wrat"], qt_sw > -lims["vrat"]]
        else:
            modes, flags = [], []
            assert np.all(np.isinf(lims))
        chk = [m != _get(sol[wnr], "type") for m in modes]
        candidates = [i for i, (c, f) in enumerate(zip(chk, flags)) if c and f]
        if candidates:
            within[wnr] = False
            switch_mode = modes[candidates[0]]
            print(f"Well {_get(sol[wnr], 'name')}: Control mode changed from "
                  f"{_get(sol[wnr], 'type')} to {switch_mode}.")
            _set(sol[wnr], "type", switch_mode)
            _set(sol[wnr], "val", lims[switch_mode])
    return sol, within


def _empty(x: Any) -> bool:
    return x is None or (isinstance(x, (list, tuple, np.ndarray)) and np.size(x) == 0)


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


updateSwitchedWellControls = update_switched_well_controls
__all__ = ["updateSwitchedWellControls", "update_switched_well_controls"]


if __name__ == "__main__":
    import contextlib
    import io
    from types import SimpleNamespace

    def test_injector_and_producer_switch_to_the_first_broken_limit():
        W = [SimpleNamespace(lims={"bhp": 300e5, "rate": 3e-3}, status=True),
             SimpleNamespace(lims={"bhp": 100e5, "orat": -1e-3}, status=True)]
        sol = [SimpleNamespace(name="I", type="rate", val=3e-3, sign=1),
               SimpleNamespace(name="P", type="bhp", val=100e5, sign=-1)]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out, within = update_switched_well_controls(
                SimpleNamespace(W=W), None, sol, np.array([320e5, 100e5]),
                [np.array([3e-3, -1e-3]), np.array([0.0, -2e-3]), np.array([0.0, 0.0])])
        assert within.tolist() == [False, False]
        assert (out[0].type, out[0].val) == ("bhp", 300e5) and (out[1].type, out[1].val) == ("orat", -1e-3)
        assert sol[0].type == "rate" and "Well I: Control mode changed from rate to bhp." in buf.getvalue()
        print("ok test_injector_and_producer_switch_to_the_first_broken_limit")

    def test_no_limits_or_a_shut_well_leave_everything():
        sol = [SimpleNamespace(name="P", type="bhp", val=1.0, sign=-1)]
        out, within = update_switched_well_controls(SimpleNamespace(W=[SimpleNamespace(lims=None)]),
                                                    None, sol, [1.0], [[0.0], [0.0], [0.0]])
        assert within.tolist() == [True] and out[0].type == "bhp"
        shut = SimpleNamespace(W=[SimpleNamespace(lims={"bhp": 2.0}, status=False)])
        out, within = update_switched_well_controls(shut, None, sol, [1.0], [[0.0], [0.0], [0.0]])
        assert within.tolist() == [True]
        print("ok test_no_limits_or_a_shut_well_leave_everything")

    test_injector_and_producer_switch_to_the_first_broken_limit()
    test_no_limits_or_a_shut_well_leave_everything()
    print("\nAll tests passed!")
