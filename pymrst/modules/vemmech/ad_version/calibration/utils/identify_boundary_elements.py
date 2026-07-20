"""MRST ``identify_boundary_elements.m`` boundary classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundaryInfo:
    side_faces: list[np.ndarray]
    side_nodes: list[np.ndarray]


def identify_boundary_elements(g, tol: float = 1e-9) -> BoundaryInfo | None:
    """Identify xmin/xmax/ymin/ymax/zmin/zmax boundary faces and nodes."""
    coords = g.faces.centroids[:, :g.griddim]
    sides = []
    for ax in range(g.griddim):
        sides.append((ax, np.min(coords[:, ax])))
        sides.append((ax, np.max(coords[:, ax])))
    if g.griddim == 2:
        sides.append((None, None))
        sides.append((None, None))

    side_faces: list[np.ndarray] = []
    side_nodes: list[np.ndarray] = []
    boundary = np.any(g.faces.neighbors < 0, axis=1)
    for ax, val in sides[:6]:
        if ax is None:
            side_faces.append(np.zeros(0, dtype=np.int64))
            side_nodes.append(np.zeros(0, dtype=np.int64))
            continue
        keep = boundary & (np.abs(coords[:, ax] - val) <= tol)
        faces = np.flatnonzero(keep)
        if faces.size == 0:
            return None
        nodes = np.unique(
            np.concatenate(
                [
                    g.faces.nodes[g.faces.nodePos[f] : g.faces.nodePos[f + 1]]
                    for f in faces
                ]
            )
        )
        side_faces.append(faces)
        side_nodes.append(nodes)
    return BoundaryInfo(side_faces=side_faces, side_nodes=side_nodes)


__all__ = ["BoundaryInfo", "identify_boundary_elements"]
