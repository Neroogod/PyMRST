"""Constant shape factor (MRST ``ConstantShapeFactor.m`` layout)."""

from dataclasses import dataclass

import numpy as np

from pymrst.params.rock import Rock
from pymrst.modules.dual_porosity.transfer_models.shape_factor_models.shape_factor import (
    ShapeFactor,
)


@dataclass
class ConstantShapeFactor(ShapeFactor):
    """A shape factor supplied directly by the caller."""

    value: float = 1.0

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        nc = np.atleast_2d(rock_matrix.perm).shape[0]
        return np.full(nc, float(self.value))


__all__ = ["ConstantShapeFactor"]
