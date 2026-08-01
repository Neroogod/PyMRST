"""Per-face flux to a cell velocity vector (MRST ``utils/faceFlux2cellVelocity.m``).

Reconstructs a single velocity vector per cell from its face fluxes, by
least squares against the face normals. The result is a vector field for
plotting or streamline tracing, not a conservative flux - the cell velocity
cannot reproduce every face flux exactly on a general cell.
"""

from typing import Tuple

import numpy as np


from pymrst.utils._cell_face_signs import (
    _cell_no_faces,
    _outward_sign,
)
from pymrst.utils.face_flux_to_cell_flux import face_flux_to_cell_flux


def face_flux_to_cell_velocity(g, face_flux: np.ndarray) -> np.ndarray:
    """Constant per-cell velocity from face fluxes (MRST `faceFlux2cellVelocity`).

    Returns `(n_cells, dim)`. `v_c = (1/|c|) * sum over the cell's half-faces of
    q_hf * (face_centroid - cell_centroid)`.
    """
    cell_no, cell_faces = _cell_no_faces(g)
    fc = np.asarray(g.faces.centroids, dtype=float)
    cc = np.asarray(g.cells.centroids, dtype=float)
    c = fc[cell_faces] - cc[cell_no]                      # (nhf, dim)
    cf = face_flux_to_cell_flux(g, np.asarray(face_flux, dtype=float).ravel())
    dim = c.shape[1]
    v = np.zeros((g.cells.num, dim))
    for d in range(dim):
        np.add.at(v[:, d], cell_no, cf * c[:, d])
    return v / np.asarray(g.cells.volumes, dtype=float)[:, None]


faceFlux2cellVelocity = face_flux_to_cell_velocity
