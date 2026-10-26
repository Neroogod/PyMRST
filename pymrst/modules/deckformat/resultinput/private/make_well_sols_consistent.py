"""MRST ``deckformat/resultinput/private/makeWellSolsConsistent.m``.

Give every step the same wells, in the order of the last step's: a well
absent at a step is present but shut (zero rates, bhp and reservoir rate,
no open connections); a well whose connection count differs from the last
step's has its rates and controls copied and its connection arrays filled
from the front. A well's zero sign is then replaced by its sign at the
nearest later step, else the latest earlier one.

Takes states (with `wellSol`) or a list of well solutions.
"""

from __future__ import annotations

import copy
from typing import Any, List

import numpy as np

ZFIELDS = ("qWs", "qOs", "qGs", "bhp", "resv")
EFIELDS = ("type", "val", "sign", "compi")


def make_well_sols_consistent(sols: List[Any]) -> List[Any]:
    """MRST `sols = makeWellSolsConsistent(sols)`."""
    state_input = isinstance(sols[0], dict) and "wellSol" in sols[0]
    well_sols = [s["wellSol"] for s in sols] if state_input else list(sols)
    tmp = copy.deepcopy(well_sols[-1])
    for w in tmp:
        nc = np.size(w.cells)
        for f in ZFIELDS + EFIELDS:
            setattr(w, f, 0)
        w.status = False
        w.cstatus = np.zeros(nc, dtype=bool)
        w.flux = np.zeros(nc)
        w.cqs = np.zeros((nc, 3))
    nr = len(well_sols)
    ws = [copy.deepcopy(tmp) for _ in range(nr)]
    for k in range(nr):
        names = [w.name for w in ws[k]]
        for wk in well_sols[k]:
            ix = names.index(wk.name)
            if np.size(ws[k][ix].cells) == np.size(wk.cells):
                ws[k][ix] = copy.deepcopy(wk)
            else:
                for f in ZFIELDS + EFIELDS:
                    setattr(ws[k][ix], f, getattr(wk, f))
                ws[k][ix].status = True
                na = np.size(wk.cells)
                ws[k][ix].cstatus[:na] = wk.cstatus
                ws[k][ix].flux[:na] = wk.flux
                ws[k][ix].cqs[:na, :] = wk.cqs
    wsign = [w.sign for w in ws[0]]
    for k in range(nr):
        for i, w in enumerate(ws[k]):
            if w.sign != 0:
                wsign[i] = w.sign
    for k in range(nr - 1, -1, -1):
        for i, w in enumerate(ws[k]):
            if w.sign == 0:
                w.sign = wsign[i]
            else:
                wsign[i] = w.sign
    if state_input:
        for k, s in enumerate(sols):
            s["wellSol"] = ws[k]
        return sols
    return ws


makeWellSolsConsistent = make_well_sols_consistent

__all__ = ["make_well_sols_consistent", "makeWellSolsConsistent"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _w(name, sign, ncell=1):
        return SimpleNamespace(name=name, cells=np.arange(ncell), qWs=1.0, qOs=1.0, qGs=1.0,
                               bhp=1.0, resv=1.0, type="bhp", val=1.0, sign=sign,
                               compi=np.array([0, 1, 0]), status=True,
                               cstatus=np.ones(ncell, dtype=bool), flux=np.ones(ncell),
                               cqs=np.ones((ncell, 3)))

    def test_missing_wells_are_shut_and_take_their_later_sign():
        out = make_well_sols_consistent([[_w("A", -1)], [_w("A", 0), _w("B", 1)]])
        assert [w.name for w in out[0]] == ["A", "B"]
        assert not out[0][1].status and out[0][1].bhp == 0
        assert out[0][1].sign == 1, "the shut well takes B's sign"
        assert out[1][0].sign == -1, "a zero sign takes the latest earlier one"
        print("ok test_missing_wells_are_shut_and_take_their_later_sign")

    def test_states_are_updated_in_place():
        states = [{"wellSol": [_w("A", 1)]}, {"wellSol": [_w("A", 1, 2)]}]
        out = make_well_sols_consistent(states)
        assert out[0]["wellSol"][0].cqs.shape == (2, 3)
        np.testing.assert_array_equal(out[0]["wellSol"][0].cstatus, [True, False])
        print("ok test_states_are_updated_in_place")

    test_missing_wells_are_shut_and_take_their_later_sign()
    test_states_are_updated_in_place()
    print("\nAll tests passed!")
