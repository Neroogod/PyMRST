"""MRST ``calculateQC.m`` 3D VEM node weights."""

from typing import Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.utils.mcolon import mcolon


def calculate_qc(g: Grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return VEM cell-node weights ``qc``, face-node weights ``qf``, volume weights.

    The output ordering matches MRST's ``calculateQC`` convention: ``qc`` and
    ``qcvol`` are ordered by ``g.cells.nodes``/``g.cells.nodePos``, while ``qf``
    is ordered by ``g.faces.nodes``.
    """
    node_pos = g.faces.nodePos
    nodes = g.faces.nodes
    nlc = np.diff(node_pos)
    face_no = np.repeat(np.arange(g.faces.num), nlc)

    xn = g.nodes.coords[nodes]
    xbb = np.column_stack([
        np.bincount(face_no, weights=xn[:, d], minlength=g.faces.num)
        for d in range(3)
    ]) / nlc[:, None]
    xb = g.faces.centroids
    n_unit = g.faces.normals / g.faces.areas[:, None]

    e = np.zeros_like(xn)
    first = node_pos[:-1]
    last = node_pos[1:] - 1
    inner1 = np.concatenate([np.arange(a + 1, b + 1)
                             for a, b in zip(first, last)]) \
        if g.faces.num else np.zeros(0, np.int64)
    inner2 = np.concatenate([np.arange(a, b)
                             for a, b in zip(first, last)]) \
        if g.faces.num else np.zeros(0, np.int64)
    e[inner2] = xn[inner1] - xn[inner2]
    e[last] = xn[first] - xn[last]

    ne = np.cross(e, n_unit[face_no])
    qf = np.zeros(nodes.size)
    dxbb = (xb - xbb)[face_no]
    qf[first] = np.sum((ne[last] + ne[first]) * dxbb[first], axis=1)
    qf[inner1] = np.sum((ne[inner2] + ne[inner1]) * dxbb[inner1], axis=1)
    qf = (g.faces.areas / nlc)[face_no] + 0.5 * qf

    assert g.cells.nodes is not None, "run create_augmented_grid first"
    cell_no_n = np.repeat(np.arange(g.cells.num), np.diff(g.cells.nodePos))
    lookup = sp.csr_array(
        (
            np.arange(1, g.cells.nodes.size + 1),
            (cell_no_n, g.cells.nodes),
        ),
        shape=(g.cells.num, g.nodes.num),
    )
    num_cn = g.cells.nodes.size
    qc = np.zeros((num_cn, 3))
    qcvol = np.zeros(num_cn)
    for side in range(2):
        cells = g.faces.neighbors[:, side][face_no]
        qcl = n_unit[face_no] * qf[:, None]
        if side == 1:
            qcl = -qcl
        keep = cells >= 0
        cellk = cells[keep]
        qclk = qcl[keep]
        rcord = g.faces.centroids[face_no][keep] - g.cells.centroids[cellk]
        cn = np.asarray(lookup[cellk, nodes[keep]]).ravel().astype(np.int64) - 1
        assert np.all(cn >= 0), "face node missing from cell node list"
        np.add.at(qcvol, cn, np.sum(rcord * qclk, axis=1) / 3.0)
        np.add.at(qc, cn, qclk)
    return qc, qf, qcvol

__all__ = ["calculate_qc"]
