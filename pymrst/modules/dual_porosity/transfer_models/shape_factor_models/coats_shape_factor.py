"""Coats shape factor (MRST ``CoatsShapeFactor.m`` layout)."""

from dataclasses import dataclass

import numpy as np

from pymrst.params.rock import Rock
from pymrst.modules.dual_porosity.shape_factors import _anisotropic
from pymrst.modules.dual_porosity.transfer_models.shape_factor_models.shape_factor import (
    ShapeFactor,
)


@dataclass
class CoatsShapeFactor(ShapeFactor):
    """Coats (1989): sigma = 8 * sum(k_i / l_i^2)."""

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        return _anisotropic(rock_matrix, self.block_dimension, 8.0)


__all__ = ["CoatsShapeFactor"]
