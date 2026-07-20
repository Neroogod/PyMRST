"""Fracture node assembly (MRST ``assembleFracNodes2D.m`` layout)."""

from typing import List, Optional, Sequence

import numpy as np

from pymrst.modules.hfm.edfm2d import FractureLine, build_edfm


def assemble_frac_nodes_2d(
    lines: Sequence[FractureLine],
    cell_size: float = 0.1,
    n_cells: Optional[Sequence[int]] = None,
) -> List[np.ndarray]:
    """Split each fracture line into equally spaced node coordinates."""
    nodes: List[np.ndarray] = []
    for i, line in enumerate(lines):
        n_seg = int(n_cells[i]) if n_cells is not None else max(1, int(round(line.length / cell_size)))
        t = np.linspace(0.0, 1.0, n_seg + 1)
        nodes.append(line.a + np.outer(t, line.b - line.a))
    return nodes


__all__ = ["FractureLine", "assemble_frac_nodes_2d", "build_edfm"]
