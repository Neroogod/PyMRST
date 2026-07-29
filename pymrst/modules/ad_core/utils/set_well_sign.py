"""Ensure schedule wells have scalar injector/producer signs."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence


def set_well_sign(wells: Sequence, *, copy_wells: bool = True):
    """Port MRST ``setWellSign``.

    Only missing/empty signs are inferred. BHP/pressure controls receive zero;
    rate controls use the sign of their target.
    """
    out = deepcopy(list(wells)) if copy_wells else list(wells)
    for well in out:
        sign = getattr(well, "sign", None)
        if sign is not None and getattr(sign, "size", 1) != 0:
            continue
        control = str(well.type).lower()
        well.sign = 0 if control in {"pressure", "bhp"} else (1 if float(well.val) >= 0 else -1)
    return out


setWellSign = set_well_sign

__all__ = ["set_well_sign", "setWellSign"]
