"""Per-cell-face flux to per-face flux (MRST ``utils/cellFlux2faceFlux.m``).

The inverse of `face_flux_to_cell_flux`: collapses the two cell-face values
of an interior face back to one, taking the first cell's sign convention.
Round-tripping through both is the identity only when the input was already
consistent between the two sides.
"""

from typing import Tuple

import numpy as np


from pymrst.utils._cell_face_signs import (
    _cell_no_faces,
    _outward_sign,
)


def cell_flux_to_face_flux(g, cell_flux: np.ndarray) -> np.ndarray:
    """Average half-face fluxes back onto faces (MRST `cellFlux2faceFlux`)."""
    cell_no, cell_faces = _cell_no_faces(g)
    sgn = _outward_sign(g, cell_no, cell_faces)
    cflux = np.asarray(cell_flux, dtype=float)
    scalar = cflux.ndim == 1
    counts = np.bincount(cell_faces, minlength=g.faces.num)
    if scalar:
        ff = np.zeros(g.faces.num)
        np.add.at(ff, cell_faces, sgn * cflux)
        return ff / counts
    ff = np.zeros((g.faces.num, cflux.shape[1]))
    for k in range(cflux.shape[1]):
        np.add.at(ff[:, k], cell_faces, sgn * cflux[:, k])
    return ff / counts[:, None]


cellFlux2faceFlux = cell_flux_to_face_flux
