"""MRST ``discreteDivergenceDiagonalJac`` public entry point."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def discrete_divergence_diagonal_jac(acc, jac, options, get_conservation=None):
    """Apply cell divergence to a Jacobian block and add accumulation.

    The compact diagonal block classes implement ``matmul_left``; sparse blocks
    use ordinary sparse multiplication. ``get_conservation`` is accepted for
    MRST signature compatibility—the Python blocks have one exact path.
    """
    get = options.get if isinstance(options, dict) else lambda key: getattr(options, key)
    neighbors = np.asarray(get("N"), dtype=np.int64)
    ncells = int(get("nc"))
    nfaces = neighbors.shape[0]
    matrix = sp.csr_array(
        (
            np.tile([1.0, -1.0], nfaces),
            (neighbors.ravel(), np.repeat(np.arange(nfaces), 2)),
        ),
        shape=(ncells, nfaces),
    )
    out = jac.matmul_left(matrix) if hasattr(jac, "matmul_left") else matrix @ jac
    if acc is not None and not (
        isinstance(acc, (list, tuple)) and len(acc) == 0
    ):
        out = out + acc
    return out


discreteDivergenceDiagonalJac = discrete_divergence_diagonal_jac

__all__ = [
    "discrete_divergence_diagonal_jac",
    "discreteDivergenceDiagonalJac",
]
