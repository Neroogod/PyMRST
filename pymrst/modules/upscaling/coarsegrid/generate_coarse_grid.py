"""Coarse-grid generation (MRST ``generateCoarseGrid.m`` layout)."""

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing.geometry import compute_geometry
from pymrst.gridprocessing.grid import Grid
from pymrst.gridprocessing.tensor_grid import tensor_grid


@dataclass
class CoarseGrid:
    """A coarse grid and its mapping back to the fine model."""

    g: Grid
    partition: np.ndarray
    n_blocks: int
    parent: Grid | None = None

    def blocks(self):
        order = np.argsort(self.partition, kind="stable")
        counts = np.bincount(self.partition, minlength=self.n_blocks)
        out, start = [], 0
        for c in counts:
            out.append(order[start:start + c])
            start += c
        return out


def generate_coarse_grid(g: Grid, partition: np.ndarray) -> CoarseGrid:
    """Build the coarse grid implied by a rectangular Cartesian partition."""
    assert g.cartDims is not None, "coarsening needs a Cartesian fine grid"
    partition = np.asarray(partition, dtype=np.int64).ravel()
    n_blocks = int(partition.max()) + 1

    dims = list(g.cartDims) + [1] * (3 - len(g.cartDims))
    ijk = np.array(np.unravel_index(np.arange(g.cells.num), dims, order="F")).T
    lines = []
    for d in range(len(g.cartDims)):
        nodes = np.unique(g.nodes.coords[:, d])
        edges = [nodes[0]]
        for idx in range(1, dims[d]):
            prev = partition[np.flatnonzero(ijk[:, d] == idx - 1)[0]]
            cur = partition[np.flatnonzero(ijk[:, d] == idx)[0]]
            if prev != cur:
                edges.append(nodes[idx])
        edges.append(nodes[-1])
        lines.append(np.asarray(edges))

    cg = compute_geometry(tensor_grid(*lines))
    assert cg.cells.num == n_blocks, (
        f"partition implies {n_blocks} blocks but the coarse grid has "
        f"{cg.cells.num}; only rectangular Cartesian partitions are supported"
    )
    # Keep the fine parent: MRST coarse-grid plotting expands a coarse face
    # into its contributing fine faces, which is impossible if generation
    # discards the source grid.
    cg.parent = g
    cg.partition = partition
    return CoarseGrid(
        g=cg, partition=partition, n_blocks=n_blocks, parent=g
    )


__all__ = ["CoarseGrid", "generate_coarse_grid"]
