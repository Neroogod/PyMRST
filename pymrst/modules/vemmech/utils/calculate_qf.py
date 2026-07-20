"""MRST ``calculateQF.m`` 2D VEM node weights."""

import numpy as np

from pymrst.gridprocessing.grid import Grid


def calculate_qf(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D VEM nodal force weights ordered by ``g.cells.nodes``."""
    assert g.griddim == 2
    assert g.cells.nodes is not None, "run create_augmented_grid first"

    qf = np.zeros((g.cells.nodes.size, 2))
    qf_vol = np.zeros(g.cells.nodes.size)
    for c in range(g.cells.num):
        sl = slice(g.cells.nodePos[c], g.cells.nodePos[c + 1])
        pts = g.nodes.coords[g.cells.nodes[sl], :2]
        n = pts.shape[0]
        center = g.cells.centroids[c, :2]
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            edge = p1 - p0
            normal = np.array([edge[1], -edge[0]])
            rel = 0.5 * (p0 + p1) - center
            tetvol = float(np.dot(normal, rel))
            qf[sl.start + i] += 0.5 * normal
            qf[sl.start + (i + 1) % n] += 0.5 * normal
            qf_vol[sl.start + i] += 0.25 * tetvol
            qf_vol[sl.start + (i + 1) % n] += 0.25 * tetvol
    return qf, qf_vol

__all__ = ["calculate_qf"]
