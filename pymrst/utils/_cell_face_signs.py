"""Cell-face indexing and orientation.

Private, and not an MRST counterpart: MRST repeats this gather inline in each
of `faceFlux2cellFlux`, `cellFlux2faceFlux` and `faceFlux2cellVelocity`.
Three copies of the same index arithmetic is how one of them ends up fixed and
the others not.
"""

from typing import Tuple

import numpy as np


def _cell_no_faces(g) -> Tuple[np.ndarray, np.ndarray]:
    """Per half-face: the cell it belongs to and the face it is (MRST
    `getCellNoFaces`)."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    counts = np.diff(face_pos)
    cell_no = np.repeat(np.arange(g.cells.num, dtype=np.int64), counts)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    return cell_no, cell_faces


def _outward_sign(g, cell_no: np.ndarray, cell_faces: np.ndarray) -> np.ndarray:
    """`+1` where the cell is the face's first neighbour (flux out of the cell
    aligns with the face normal), `-1` otherwise."""
    n0 = np.asarray(g.faces.neighbors, dtype=np.int64)[cell_faces, 0]
    return np.where(n0 == cell_no, 1.0, -1.0)
