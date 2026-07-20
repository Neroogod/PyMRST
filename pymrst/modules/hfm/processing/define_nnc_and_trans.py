"""NNC/transmissibility extraction (MRST ``defineNNCandTrans.m`` layout)."""

from typing import Tuple

import numpy as np

from pymrst.modules.hfm.edfm2d import EDFMGrid, build_edfm


def define_nnc_and_trans(e: EDFMGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return EDFM NNC pairs, transmissibilities, CI values, and labels."""
    return e.conn.copy(), e.trans.copy(), e.ci.copy(), list(e.conn_type)


__all__ = ["EDFMGrid", "build_edfm", "define_nnc_and_trans"]
