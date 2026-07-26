"""Arithmetic one-sided flux maps (MRST ``AvgTransNTPFA.m``)."""

import numpy as np
import scipy.sparse as sp


def avg_trans_ntpfa(g, os_flux):
    """Return two sparse maps from cell pressure to one-sided face flux."""
    internal = np.flatnonzero(np.all(g.faces.neighbors >= 0, axis=1))
    matrices = []
    for side in range(2):
        rows, cols, values = [], [], []
        for row, face in enumerate(internal):
            stencil = os_flux[face, side]
            for position, (cell, coefficient) in enumerate(stencil):
                cell = int(cell)
                if cell < 0:
                    continue
                rows.append(row)
                cols.append(cell)
                # AvgMPFA's maps represent pressure gradients, i.e. the
                # negative of each outward one-sided Darcy flux stencil.
                values.append(-float(coefficient) if position == 0
                              else float(coefficient))
        matrices.append(sp.csr_array(
            (values, (rows, cols)), shape=(internal.size, g.cells.num)
        ))
    return tuple(matrices)


AvgTransNTPFA = avg_trans_ntpfa

__all__ = ["avg_trans_ntpfa", "AvgTransNTPFA"]
