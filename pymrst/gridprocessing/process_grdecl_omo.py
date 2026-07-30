"""Compatibility entry point for MRST ``processGRDECLOMO``.

MRST's OMO file is an alternative implementation of the same corner-point
processor, not a distinct grid format. PyMRST routes it through the validated
``process_grdecl`` kernel while preserving its public behavioral options.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from pymrst.gridprocessing.check_and_repair_zcorn import check_and_repair_zcorn
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.gridprocessing.process_pinch import process_pinch
from pymrst.gridprocessing.split_disconnected_grid import split_disconnected_grid


def process_grdecl_omo(
    grdecl,
    *,
    check_grid: bool = True,
    repair_zcorn: bool = False,
    split_disconnected: bool = False,
    tolerance: float = 0.0,
    verbose: bool = False,
    preserve_cp_nodes: bool = False,
):
    """Process a corner-point grid through the OMO-compatible API.

    ``verbose`` is accepted for call compatibility. ``preserve_cp_nodes`` is
    rejected because PyMRST's canonical ``Grid`` does not currently retain the
    eight pre-deduplication corner-node indices.
    """
    del verbose
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if preserve_cp_nodes:
        raise NotImplementedError(
            "processGRDECLOMO PreserveCpNodes is not represented by PyMRST Grid"
        )
    data = deepcopy(dict(grdecl))
    if tolerance > 0:
        zcorn = np.asarray(data["ZCORN"], dtype=float)
        data["ZCORN"] = tolerance * np.round(zcorn / tolerance)
    if repair_zcorn:
        data["ZCORN"] = check_and_repair_zcorn(
            data["ZCORN"],
            data["cartDims"],
            active=data.get("ACTNUM"),
        )

    grid = process_grdecl(data, tolerance=0.0, check_grid=check_grid)
    if "PINCH" in data and "ACTNUM" in data:
        nnc = process_pinch(data, grid)
        if nnc is not None:
            grid.nnc = nnc
    if split_disconnected:
        return split_disconnected_grid(grid)
    return grid


processGRDECLOMO = process_grdecl_omo

__all__ = ["process_grdecl_omo", "processGRDECLOMO"]
