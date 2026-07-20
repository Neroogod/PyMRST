"""MRST ``kron_ADI.m`` Kronecker product for numeric/ADI vectors."""

import numpy as np

from pymrst.modules.vemmech.ad_version import _coef_concat, _coef_size, _coef_take
from pymrst.utils.adi import ADI


def kron_adi(u, v):
    """Eager counterpart of MRST ``kron_ADI`` for numeric or ADI vectors."""
    if not isinstance(u, ADI) and not isinstance(v, ADI):
        return np.kron(np.asarray(u, dtype=float).ravel(),
                       np.asarray(v, dtype=float).ravel())
    vals = [
        _coef_take(u, i) * _coef_take(v, j)
        for i in range(_coef_size(u))
        for j in range(_coef_size(v))
    ]
    return _coef_concat(vals)

__all__ = ["kron_adi"]
