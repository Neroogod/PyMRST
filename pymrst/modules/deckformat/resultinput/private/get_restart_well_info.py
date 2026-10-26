"""MRST ``deckformat/resultinput/private/getRestartWellInfo.m``.

The wells of one restart step, from the INTEHEAD sizes and the ZWEL, IWEL,
SWEL, XWEL, ICON, SCON and XCON arrays: name, head IJK, connection count,
group, type, control and status; reference depth; surface, liquid and
reservoir rates (production positive in the file, negated here) and bhp;
per connection IJK, status, direction, well index, depth, diameter, kh,
surface and reservoir rates and pressure. Positions are MRST's (1-based
item numbers in the ECLIPSE arrays). `rep_step` is 1-based.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

INTEHEAD_ITEMS = {"unit": 3, "nx": 9, "ny": 10, "nz": 11, "nactive": 12, "iphs": 15,
                  "nwell": 17, "ncwma": 18, "nwgmax": 20, "ngmaxz": 21, "niwel": 25,
                  "nswel": 26, "nxwel": 27, "nzwel": 28, "nicon": 33, "nscon": 34,
                  "nxcon": 35, "nigrpz": 37, "iday": 65, "imon": 66, "iyear": 67,
                  "iprog": 95}


def get_restart_well_info(rstrt: Dict[str, List[Any]], rep_step: int
                          ) -> Tuple[List[SimpleNamespace], SimpleNamespace]:
    """MRST `[w, ih] = getRestartWellInfo(rstrt, repStep)`."""
    k = rep_step - 1
    ih = get_intehead(rstrt["INTEHEAD"][k])
    if ih.nwell <= 0:
        return [], ih
    w = [SimpleNamespace() for _ in range(ih.nwell)]
    _zwel(w, rstrt["ZWEL"][k], ih)
    _iwel(w, np.asarray(rstrt["IWEL"][k]), ih)
    if "SWEL" in rstrt:
        _swel(w, np.asarray(rstrt["SWEL"][k]), ih)
    _xwel(w, np.asarray(rstrt["XWEL"][k]), ih)
    _icon(w, np.asarray(rstrt["ICON"][k]), ih)
    if "SCON" in rstrt:
        _scon(w, np.asarray(rstrt["SCON"][k]), ih)
    _xcon(w, np.asarray(rstrt["XCON"][k]), ih)
    return w, ih


getRestartWellInfo = get_restart_well_info


def get_intehead(intehead: Any) -> SimpleNamespace:
    """MRST's `getINTEHEAD`: the named INTEHEAD items."""
    ih = np.asarray(intehead).ravel()
    return SimpleNamespace(**{n: int(ih[i - 1]) for n, i in INTEHEAD_ITEMS.items()})


def _item(array: np.ndarray, item: Any, offset: int) -> Any:
    return array[np.asarray(item) - 1 + offset]


def _zwel(w, zwel, ih):
    for k, wk in enumerate(w):
        wk.name = str(zwel[k*ih.nzwel]).strip()


def _iwel(w, iwel, ih):
    for k, wk in enumerate(w):
        d = k*ih.niwel
        wk.ijk = _item(iwel, [1, 2, 3], d).astype(int)
        wk.ncon = int(_item(iwel, 5, d))
        wk.ginx = int(_item(iwel, 6, d))
        wk.type = int(_item(iwel, 7, d))
        wk.cntr = int(_item(iwel, 9, d))
        wk.stat = bool(_item(iwel, 11, d) > 0)


def _swel(w, swel, ih):
    for k, wk in enumerate(w):
        wk.depth = float(_item(swel, 10, k*ih.nswel))


def _xwel(w, xwel, ih):
    for k, wk in enumerate(w):
        d = k*ih.nxwel
        wk.qOs = -float(_item(xwel, 1, d))
        wk.qWs = -float(_item(xwel, 2, d))
        wk.qGs = -float(_item(xwel, 3, d))
        wk.lrat = -float(_item(xwel, 4, d))
        wk.qr = -float(_item(xwel, 5, d))
        wk.bhp = float(_item(xwel, 7, d))


def _icon(w, icon, ih):
    for k, wk in enumerate(w):
        d = k*ih.nicon*ih.ncwma
        dc = np.arange(wk.ncon)*ih.nicon
        wk.cijk = icon[dc[:, None] + np.arange(2, 5)[None, :] - 1 + d].astype(int)
        wk.cstat = icon[6 + dc + d - 1]
        wk.cdir = icon[14 + dc + d - 1]


def _scon(w, scon, ih):
    for k, wk in enumerate(w):
        d = k*ih.nscon*ih.ncwma
        dc = np.arange(wk.ncon)*ih.nscon
        wk.cwi = scon[1 + dc + d - 1]
        wk.cdepth = scon[2 + dc + d - 1]
        wk.cdiam = scon[3 + dc + d - 1]
        wk.ckh = scon[4 + dc + d - 1]


def _xcon(w, xcon, ih):
    for k, wk in enumerate(w):
        d = k*ih.nxcon*ih.ncwma
        dc = np.arange(wk.ncon)*ih.nxcon
        wk.cqs = -xcon[dc[:, None] + np.arange(1, 4)[None, :] - 1 + d]
        wk.cqr = -xcon[50 + dc + d - 1]
        wk.press = xcon[35 + dc + d - 1]


__all__ = ["get_intehead", "get_restart_well_info", "getRestartWellInfo"]


if __name__ == "__main__":
    def test_one_well_with_two_connections():
        ih = np.zeros(100, dtype=int)
        ih[[2, 16, 17, 24, 25, 26, 27, 32, 33, 34]] = [1, 1, 2, 11, 10, 7, 1, 15, 4, 50]
        iwel = np.zeros(11, dtype=int)
        iwel[[0, 1, 2, 4, 6, 8, 10]] = [3, 4, 1, 2, 1, 7, 1]
        icon = np.zeros(30, dtype=int)
        icon[[1, 2, 3, 5, 13]] = [3, 4, 1, 1, 3]
        icon[[16, 17, 18, 20, 28]] = [3, 4, 2, 1, 3]
        xwel = np.array([10.0, 2.0, 100.0, 12.0, 15.0, 0.0, 250.0])
        xcon = np.zeros(100)
        xcon[[0, 1, 2, 34, 49]] = [5.0, 1.0, 50.0, 260.0, 7.0]
        rstrt = {"INTEHEAD": [ih], "ZWEL": [["PROD1   "]], "IWEL": [iwel],
                 "XWEL": [xwel], "ICON": [icon], "XCON": [xcon]}
        w, info = get_restart_well_info(rstrt, 1)
        assert info.nwell == 1 and w[0].name == "PROD1"
        assert w[0].ncon == 2 and w[0].stat and w[0].cntr == 7
        np.testing.assert_array_equal(w[0].cijk, [[3, 4, 1], [3, 4, 2]])
        assert w[0].qOs == -10.0 and w[0].bhp == 250.0
        np.testing.assert_allclose(w[0].cqs[0], [-5.0, -1.0, -50.0])
        assert w[0].press[0] == 260.0 and w[0].cqr[0] == -7.0
        print("ok test_one_well_with_two_connections")

    test_one_well_with_two_connections()
    print("\nAll tests passed!")
