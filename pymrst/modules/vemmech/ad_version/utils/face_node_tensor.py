"""MRST ``face_node_tensor.m`` face-node sparse tensor construction."""

import numpy as np

from pymrst.modules.vemmech.ad_version import SparseMultiArray, _coef_array, _coef_take


def face_node_tensor(g, fname: str, nname: str, values=None,
                     boundary_only: bool = False) -> SparseMultiArray:
    """Construct the face-node indicator tensor for a grid."""
    face_no = np.repeat(np.arange(g.faces.num) + 1, np.diff(g.faces.nodePos))
    nodes = g.faces.nodes + 1
    ixs = np.column_stack([face_no, nodes])
    vals = None if values is None else _coef_array(values)
    if boundary_only:
        boundary = np.any(g.faces.neighbors < 0, axis=1)
        keep = np.repeat(boundary, np.diff(g.faces.nodePos))
        ixs = ixs[keep]
        if vals is not None:
            vals = _coef_take(vals, keep)
    return SparseMultiArray(vals, ixs, [fname, nname])

__all__ = ["face_node_tensor"]
