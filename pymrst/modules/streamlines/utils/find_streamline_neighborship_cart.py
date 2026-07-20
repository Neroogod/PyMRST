"""MRST ``findStreamlineNeighborshipCart.m`` Cartesian streamline topology."""

import numpy as np

from pymrst.gridprocessing.grid import Grid


def cell_nodes(g: Grid) -> np.ndarray:
    """Return the tensor-product corner nodes of every logically Cartesian cell."""
    if g.cells.tags is None:
        raise ValueError("pollock needs a logically Cartesian grid "
                         "(cells.tags is None)")
    d = g.griddim
    out = np.zeros((g.cells.num, 2 ** d), dtype=np.int64)
    face_nodes = [[None] * (2 * d) for _ in range(g.cells.num)]
    for c in range(g.cells.num):
        lo, hi = g.cells.facePos[c], g.cells.facePos[c + 1]
        for f, t in zip(g.cells.faces[lo:hi], g.cells.tags[lo:hi]):
            face_nodes[c][int(t)] = set(
                int(n)
                for n in g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
            )
    for c in range(g.cells.num):
        for j in range(2 ** d):
            shared = None
            for i in range(d):
                tag = 2 * i + ((j >> i) & 1)
                ns = face_nodes[c][tag]
                if ns is None:
                    raise ValueError(f"cell {c} is missing face tag {tag}")
                shared = ns if shared is None else (shared & ns)
            if len(shared) != 1:
                raise ValueError(
                    f"cell {c}, corner {j}: expected 1 shared node, "
                    f"got {len(shared)}"
                )
            out[c, j] = shared.pop()
    return out


def streamline_neighbors(g: Grid) -> np.ndarray:
    """Return neighbouring cell across each tagged face, ``-1`` on boundaries."""
    if g.cells.tags is None:
        raise ValueError("pollock needs a logically Cartesian grid "
                         "(cells.tags is None)")
    d = g.griddim
    cell_no = np.repeat(np.arange(g.cells.num), np.diff(g.cells.facePos))
    nb = g.faces.neighbors[g.cells.faces]
    other = np.where(nb[:, 0] == cell_no, nb[:, 1], nb[:, 0])
    out = np.full((g.cells.num, 2 * d), -1, dtype=np.int64)
    out[cell_no, g.cells.tags] = other
    return out

__all__ = ["cell_nodes", "streamline_neighbors"]
