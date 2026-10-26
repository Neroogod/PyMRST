"""MRST ``deckformat/resultinput/private/createAquiSol.m``.

One restart step's aquifers (`getRestartAquiInfo`) in SI units, with their
connection cells (through `ijk2act`, to 0-based active cells) in place of
the IJKs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from pymrst.modules.deckformat.resultinput.private.get_restart_aqui_info import (
    get_restart_aqui_info,
)
from pymrst.utils.units.get_unit_system import get_unit_system


def create_aqui_sol(rstrt: Dict[str, List[Any]], rep_step: int,
                    ijk2act: Callable) -> List[Any]:
    """MRST `aq = createAquiSol(rstrt, repStep, ijk2act)`."""
    aq, unit_no = get_restart_aqui_info(rstrt, rep_step)
    if aq:
        u = get_unit_system(("metric", "field", "lab")[unit_no - 1])
        for a in aq:
            a.cells = ijk2act(a.cijk)
            a.pressure = a.pressure*u.press
            a.qW = a.qW*(u.resvolume/u.time)
            a.flux = a.flux*(u.resvolume/u.time)
            a.vol = a.vol*u.resvolume
            del a.cijk
    return aq


createAquiSol = create_aqui_sol

__all__ = ["create_aqui_sol", "createAquiSol"]


if __name__ == "__main__":
    import numpy as np

    def test_no_aquifers_stays_empty():
        ih = np.zeros(100, dtype=int)
        ih[2] = 1
        assert create_aqui_sol({"INTEHEAD": [ih]}, 1, lambda ijk: ijk) == []
        print("ok test_no_aquifers_stays_empty")

    test_no_aquifers_stays_empty()
    print("\nAll tests passed!")
