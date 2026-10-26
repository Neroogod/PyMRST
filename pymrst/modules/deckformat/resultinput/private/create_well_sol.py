"""MRST ``deckformat/resultinput/private/createWellSol.m``.

One restart step's wells as MRST well solutions: cells (from the
connection IJKs through `ijk2act`), well indices, reference depth and
connection depth differences, status, surface rates (oil/water/gas), bhp,
reservoir rate, connection reservoir fluxes and pressures, and connection
surface rates (water, oil, gas). An open well also gets its type, target
value, injection composition and sign from its control mode: producers by
`orat`, `wrat`, `grat`, `lrat`, `resv`, `unknown`, `bhp`; injectors by
`rate` for the first three. Units are the restart's (metric or field).

`ijk2act` maps 1-based IJK rows to 0-based active cells.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pymrst.modules.deckformat.resultinput.private.get_restart_well_info import (
    get_restart_well_info,
)
from pymrst.utils.units import BARSA, CENTI, DAY, FT, METER, POISE, PSIA, STB

PRODUCER_TYPES = ("orat", "wrat", "grat", "lrat", "resv", "unknown", "bhp")


def create_well_sol(rstrt: Dict[str, List[Any]], rep_step: int,
                    ijk2act: Callable[[np.ndarray], np.ndarray]
                    ) -> Optional[List[SimpleNamespace]]:
    """MRST `wellSol = createWellSol(rstrt, repStep, ijk2act)`."""
    w, ih = get_restart_well_info(rstrt, rep_step)
    u = _get_units(("metric", "field", "lab")[ih.unit - 1])
    if not w:
        return None
    well_sol = []
    for wk in w:
        ws = SimpleNamespace(
            name=wk.name, cells=ijk2act(wk.cijk), type="", val=None, WI=wk.cwi*u.trans,
            compi=None, refDepth=wk.depth*u.len, dZ=(wk.cdepth - wk.depth)*u.len,
            dir=wk.cdir, sign=0, status=wk.stat > 0, cstatus=wk.cstat > 0,
            qWs=wk.qWs*u.ql, qOs=wk.qOs*u.ql, qGs=wk.qGs*u.qg, bhp=wk.bhp*u.p,
            resv=wk.qr*u.qr, flux=wk.cqr*u.qr, press=wk.press*u.p,
            cqs=np.column_stack([wk.cqs[:, 1]*u.ql, wk.cqs[:, 0]*u.ql, wk.cqs[:, 2]*u.qg]))
        if wk.stat > 0:
            vals = [ws.qOs, ws.qWs, ws.qGs, ws.qOs + ws.qWs, ws.resv, np.nan, ws.bhp]
            compis = ([0, 1, 0], [1, 0, 0], [0, 0, 1])
            known = 0 < wk.cntr <= len(PRODUCER_TYPES)
            if wk.type == 1:
                sign, compi = -1, [0, 1, 0]
                kind = PRODUCER_TYPES[wk.cntr - 1] if known else ""
                val = vals[wk.cntr - 1] if known else np.nan
            else:
                sign = 1
                compi = (compis[wk.type - 2] if wk.type <= 4
                         else list(np.array(vals[:3])/np.sum(vals[:3])))
                if known:
                    val = vals[wk.cntr - 1]
                    kind = "rate" if wk.cntr <= 3 else PRODUCER_TYPES[wk.cntr - 1]
                else:
                    kind, val = "", np.nan
            ws.type, ws.val, ws.compi, ws.sign = kind, val, np.asarray(compi, float), sign
        well_sol.append(ws)
    return well_sol


createWellSol = create_well_sol


def _get_units(unit: str) -> SimpleNamespace:
    if unit == "metric":
        return SimpleNamespace(len=METER, p=BARSA, ql=METER**3/DAY, qg=METER**3/DAY,
                               qr=METER**3/DAY, trans=CENTI*POISE*METER**3/(DAY*BARSA))
    if unit == "field":
        return SimpleNamespace(len=FT, p=PSIA, ql=STB/DAY, qg=1000*FT**3/DAY, qr=STB/DAY,
                               trans=CENTI*POISE*STB/(DAY*PSIA))
    raise ValueError(f"Unit {unit} not supported")


__all__ = ["create_well_sol", "createWellSol"]


if __name__ == "__main__":
    from unittest import mock

    def _well(**over):
        base = dict(name="P", cijk=np.array([[1, 1, 1]]), cwi=np.array([2.0]),
                    depth=1000.0, cdepth=np.array([1010.0]), cdir=np.array([3]),
                    stat=True, cstat=np.array([1]), qWs=-1.0, qOs=-10.0, qGs=-100.0,
                    qr=-12.0, bhp=200.0, cqr=np.array([-12.0]), press=np.array([210.0]),
                    cqs=np.array([[-10.0, -1.0, -100.0]]), type=1, cntr=1)
        base.update(over)
        return SimpleNamespace(**base)

    def _create(well):
        info = SimpleNamespace(unit=1)
        with mock.patch.dict(globals(), get_restart_well_info=lambda r, k: ([well], info)):
            return create_well_sol({}, 1, lambda ijk: np.array([0]))

    def test_a_producer_on_oil_rate_control():
        ws = _create(_well())[0]
        assert ws.type == "orat" and ws.sign == -1
        np.testing.assert_allclose(ws.val, -10/DAY)
        np.testing.assert_allclose(ws.cqs, [[-1/DAY, -10/DAY, -100/DAY]])
        np.testing.assert_allclose([ws.bhp, ws.dZ[0]], [200*BARSA, 10.0])
        print("ok test_a_producer_on_oil_rate_control")

    def test_a_water_injector_is_rate_controlled():
        ws = _create(_well(type=3, cntr=2, qWs=50.0, qOs=0.0, qGs=0.0))[0]
        assert ws.type == "rate" and ws.sign == 1
        np.testing.assert_array_equal(ws.compi, [1, 0, 0])
        print("ok test_a_water_injector_is_rate_controlled")

    test_a_producer_on_oil_rate_control()
    test_a_water_injector_is_rate_controlled()
    print("\nAll tests passed!")
