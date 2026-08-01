"""Per-face flux to per-cell-face flux (MRST ``utils/faceFlux2cellFlux.m``).

Expands a value stored once per face into one value per *cell-face*, so
each cell sees the flux across its own boundary with the sign it needs. A
face shared by two cells appears twice, with opposite signs.
"""

from typing import Tuple

import numpy as np


from pymrst.utils._cell_face_signs import (
    _cell_no_faces,
    _outward_sign,
)


def face_flux_to_cell_flux(g, face_flux: np.ndarray) -> np.ndarray:
    """Half-face flux, positive out of each cell (MRST `faceFlux2cellFlux`)."""
    cell_no, cell_faces = _cell_no_faces(g)
    sgn = _outward_sign(g, cell_no, cell_faces)
    ff = np.asarray(face_flux, dtype=float)
    scalar = ff.ndim == 1
    ff = np.atleast_2d(ff.T).T if not scalar else ff
    cf = sgn * ff[cell_faces] if scalar else sgn[:, None] * ff[cell_faces]
    return cf


faceFlux2cellFlux = face_flux_to_cell_flux
