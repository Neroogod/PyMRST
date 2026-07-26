"""Arithmetic face permeability (MRST ``arithmeticAverageMPFA.m``)."""

import numpy as np


def arithmetic_average_mpfa(g, krw, bc, potential):
    """Return face relative permeability for cells and boundary conditions."""
    potential = np.asarray(potential, dtype=float).ravel()
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    internal = np.all(neighbors >= 0, axis=1)
    result = np.zeros(g.faces.num)
    c1, c2 = neighbors[internal].T
    result[internal] = 0.5*(krw(potential[c1]) + krw(potential[c2]))
    if bc is not None:
        for face, kind, value in zip(bc.face, bc.type, bc.value):
            face = int(face)
            if kind == "flux":
                result[face] = 1.0
            elif kind == "pressure":
                cell = int(max(neighbors[face]))
                result[face] = 0.5*(
                    float(krw(np.array([value]))[0])
                    + float(krw(np.array([potential[cell]]))[0])
                )
            else:
                raise ValueError(f"unsupported boundary type {kind!r}")
    return result


arithmeticAverageMPFA = arithmetic_average_mpfa

__all__ = ["arithmetic_average_mpfa", "arithmeticAverageMPFA"]
