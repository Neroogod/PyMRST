"""2D fracture preprocessing (MRST ``processFracture2D.m`` layout)."""

from typing import Iterable, List

import numpy as np

from pymrst.modules.hfm.edfm2d import FractureLine


def process_fracture_2d(fractures: Iterable[np.ndarray | FractureLine]) -> List[FractureLine]:
    """Normalize endpoint arrays into :class:`FractureLine` objects."""
    out: List[FractureLine] = []
    for frac in fractures:
        if isinstance(frac, FractureLine):
            line = frac
        else:
            arr = np.asarray(frac, dtype=float).ravel()
            if arr.size != 4:
                raise ValueError("each 2D fracture must be [x1, y1, x2, y2]")
            line = FractureLine(arr)
        if line.length <= 0:
            raise ValueError("fracture endpoints must be distinct")
        out.append(line)
    return out


__all__ = ["FractureLine", "process_fracture_2d"]
