"""Lim-Aziz shape factor (MRST ``LimAzizShapeFactor.m`` layout)."""

from dataclasses import dataclass

import numpy as np

from pymrst.params.rock import Rock
from pymrst.modules.dual_porosity.shape_factors import _anisotropic, _perms
from pymrst.modules.dual_porosity.transfer_models.shape_factor_models.shape_factor import (
    ShapeFactor,
)


@dataclass
class LimAzizShapeFactor(ShapeFactor):
    """Lim & Aziz shape factor, with optional MRST-compatible permeability removal."""

    mrst_compatible: bool = False

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        s = _anisotropic(rock_matrix, self.block_dimension, np.pi**2)
        if not self.mrst_compatible:
            return s
        nc = np.atleast_2d(rock_matrix.perm).shape[0]
        k = _perms(rock_matrix, nc)
        return s / np.cbrt(k.prod(axis=1))


__all__ = ["LimAzizShapeFactor"]
