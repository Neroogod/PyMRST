"""MRST ``deckformat/resultinput/private/getRestartAquiInfo.m``.

The analytic aquifers of one restart step (Fetkovich only): per aquifer the
pressure, water rate and remaining volume, and per connection the IJK,
flux and connection number. Returns `(aq, unitNo)`, `aq` empty when the
restart has none or they are not Fetkovich (with MRST's warning).
`step` is 1-based.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np


def get_restart_aqui_info(rstrt: Dict[str, List[Any]], step: int
                          ) -> Tuple[List[SimpleNamespace], int]:
    """MRST `[aq, unitNo] = getRestartAquiInfo(rstrt, step)`."""
    k = step - 1
    ih = np.asarray(rstrt["INTEHEAD"][k]).ravel()
    unit_no = int(ih[2])
    naq = int(ih[40])
    if not (naq > 0 and "XAAQ" in rstrt and ("ACAQ_1" in rstrt or "ACAQ" in rstrt)):
        return [], unit_no
    niaaq = int(ih[42])
    iaaq = np.asarray(rstrt["IAAQ"][k]).ravel()
    tpix = (np.array([[10], [11]]) + np.arange(naq)*niaaq).ravel(order="F") - 1
    if not np.all(iaaq[tpix] == 0):
        warnings.warn("Only aquifers of Fetkovich-type are supported")
        return [], unit_no
    nsaaq, nxaaq = int(ih[43]), int(ih[44])
    saaq = np.asarray(rstrt["SAAQ"][k]).ravel()
    xaaq = np.asarray(rstrt["XAAQ"][k]).ravel()
    if xaaq.size != naq*nxaaq:
        warnings.warn("Unexpected format of aquifer data in restart file, skipping ...")
        return [], unit_no
    aq = [SimpleNamespace(cijk=None, pressure=xaaq[1 + i*nxaaq], qW=xaaq[i*nxaaq],
                          flux=None, vol=saaq[1 + i*nsaaq] - xaaq[2 + i*nxaaq], num=None)
          for i in range(naq)]
    nconn = iaaq[np.arange(naq)*niaaq]
    nicaq, nacaq = int(ih[45]), int(ih[47])
    sfx = (lambda i: f"_{i}") if naq > 1 else (lambda i: "")
    for i, a in enumerate(aq):
        icaq = np.asarray(rstrt["ICAQ" + sfx(i + 1)][k]).ravel()
        dc = np.arange(nconn[i])*nicaq
        a.cijk = icaq[dc[:, None] + np.arange(3)[None, :]].astype(int)
        acaq = np.asarray(rstrt["ACAQ" + sfx(i + 1)][k]).ravel()
        a.flux = acaq[np.arange(nconn[i])*nacaq]
        a.num = rstrt["ACAQNUM" + sfx(i + 1)][k]
    return aq, unit_no


getRestartAquiInfo = get_restart_aqui_info

__all__ = ["get_restart_aqui_info", "getRestartAquiInfo"]


if __name__ == "__main__":
    def test_no_aquifers_gives_an_empty_list():
        ih = np.zeros(100, dtype=int)
        ih[2] = 1
        aq, unit = get_restart_aqui_info({"INTEHEAD": [ih]}, 1)
        assert aq == [] and unit == 1
        print("ok test_no_aquifers_gives_an_empty_list")

    def test_one_fetkovich_aquifer():
        ih = np.zeros(100, dtype=int)
        ih[[2, 40, 42, 43, 44, 45, 47]] = [1, 1, 12, 5, 4, 3, 2]
        iaaq = np.zeros(12, dtype=int)
        iaaq[0] = 2
        rstrt = {"INTEHEAD": [ih], "IAAQ": [iaaq], "SAAQ": [np.array([0, 500.0, 0, 0, 0])],
                 "XAAQ": [np.array([3.0, 250.0, 100.0, 0.0])],
                 "ICAQ": [np.array([1, 2, 3, 4, 5, 6])], "ACAQ": [np.array([0.5, 0, 0.7, 0])],
                 "ACAQNUM": [np.array([1, 2])]}
        aq, _ = get_restart_aqui_info(rstrt, 1)
        assert aq[0].pressure == 250.0 and aq[0].vol == 400.0
        np.testing.assert_array_equal(aq[0].cijk, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(aq[0].flux, [0.5, 0.7])
        print("ok test_one_fetkovich_aquifer")

    test_no_aquifers_gives_an_empty_list()
    test_one_fetkovich_aquifer()
    print("\nAll tests passed!")
