"""Harmonic averaging points, ported from MRST ``findHAP.m``."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from pymrst.params.rock.perm_tensor import perm_tensor


@dataclass
class HarmonicAveragingPoints:
    """Coordinates, two-cell interpolation weights, and hull diagnostics."""

    coords: np.ndarray
    weights: np.ndarray
    counter: np.ndarray
    fraction: float
    corrected: float = 0.0


def _inside_convex_hull(point, vertices, tolerance=1e-10):
    """Return whether ``point`` satisfies the hull's half-space equations."""
    if vertices.shape[0] <= vertices.shape[1]:
        return False
    try:
        hull = ConvexHull(vertices)
    except Exception:
        return False
    return bool(np.all(hull.equations[:, :-1] @ point
                       + hull.equations[:, -1] <= tolerance))


def find_hap(g, rock):
    """Compute NFVM harmonic averaging points for interior and boundary faces."""
    dim = g.griddim
    tensors = perm_tensor(rock, dim)[0].reshape(g.cells.num, dim, dim)
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    normals = np.asarray(g.faces.normals, dtype=float)
    face_centroids = np.asarray(g.faces.centroids, dtype=float)
    cell_centroids = np.asarray(g.cells.centroids, dtype=float)
    internal = np.all(neighbors >= 0, axis=1)

    coords = np.zeros((g.faces.num, dim))
    weights = np.zeros((g.faces.num, 2))

    faces = np.flatnonzero(internal)
    if faces.size:
        c1, c2 = neighbors[faces, 0], neighbors[faces, 1]
        n = normals[faces]
        kn1 = np.einsum("fij,fj->fi", tensors[c1], n)
        kn2 = np.einsum("fij,fj->fi", tensors[c2], n)
        x1, x2, xf = cell_centroids[c1], cell_centroids[c2], face_centroids[faces]
        x_a = x1 + (
            np.einsum("fi,fi->f", xf - x1, n)
            / np.einsum("fi,fi->f", kn1, n)
        )[:, None] * kn1
        x_b = x2 + (
            np.einsum("fi,fi->f", xf - x2, n)
            / np.einsum("fi,fi->f", kn2, n)
        )[:, None] * kn2
        w1 = np.linalg.norm(kn1, axis=1) / np.linalg.norm(x_a - x1, axis=1)
        w2 = np.linalg.norm(kn2, axis=1) / np.linalg.norm(x_b - x2, axis=1)
        coords[faces] = (
            x_a*w1[:, None] + x_b*w2[:, None]
        ) / (w1 + w2)[:, None]
        weights[faces, 0] = w1 / (w1 + w2)
        weights[faces, 1] = w2 / (w1 + w2)

    faces = np.flatnonzero(~internal)
    if faces.size:
        side = np.where(neighbors[faces, 0] >= 0, 0, 1)
        cells = neighbors[faces, side]
        n = normals[faces]
        kn = np.einsum("fij,fj->fi", tensors[cells], n)
        xc, xf = cell_centroids[cells], face_centroids[faces]
        coords[faces] = xc + (
            np.einsum("fi,fi->f", xf - xc, n)
            / np.einsum("fi,fi->f", kn, n)
        )[:, None] * kn
        weights[faces, side] = 1.0

    counter = np.zeros(g.cells.num, dtype=bool)
    for cell in range(g.cells.num):
        lo, hi = g.cells.facePos[cell:cell + 2]
        counter[cell] = _inside_convex_hull(
            cell_centroids[cell], coords[np.asarray(g.cells.faces[lo:hi])]
        )
    fraction = 1.0 - float(np.mean(counter))
    return HarmonicAveragingPoints(coords, weights, counter, fraction)


findHAP = find_hap

__all__ = ["HarmonicAveragingPoints", "find_hap", "findHAP"]
