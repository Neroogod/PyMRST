"""Inspect one cell's faces (MRST ``plotSingleCellInfo.m``)."""

from __future__ import annotations
import numpy as np
from pymrst.utils.gridtools.tests.plot_single_face_info import plot_single_face_info


def plot_single_cell_info(grid, cell: int, *, plot_face_info=None, ax=None, **kwargs):
    cell = int(cell)
    if not 0 <= cell < grid.cells.num:
        raise IndexError("cell index outside grid")
    start, stop = grid.cells.facePos[cell:cell + 2]
    faces = np.asarray(grid.cells.faces[start:stop], int)
    if plot_face_info is None:
        flags = np.ones(faces.size, bool)
    else:
        raw = np.asarray(plot_face_info)
        if raw.dtype == bool:
            if raw.size != faces.size:
                raise ValueError("logical plot_face_info must have one entry per face")
            flags = raw
        else:
            flags = np.zeros(faces.size, bool); flags[np.asarray(raw, int)] = True
    details = [plot_single_face_info(grid, face, from_cell=cell,
               plot_info=bool(flag), ax=ax, **kwargs)
               for face, flag in zip(faces, flags)]
    return {"cell": cell, "faces": faces, "plot_info": flags, "details": details}


plotSingleCellInfo = plot_single_cell_info
__all__ = ["plotSingleCellInfo", "plot_single_cell_info"]
