"""Base dual-porosity shape factor (MRST ``ShapeFactor.m`` layout)."""

from dataclasses import dataclass
from typing import Union

import numpy as np

from pymrst.params.rock import Rock

Spacing = Union[float, np.ndarray]


@dataclass
class ShapeFactor:
    """Base class: sigma = 1."""

    block_dimension: Spacing = 1.0

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        return np.ones(np.atleast_2d(rock_matrix.perm).shape[0])


__all__ = ["ShapeFactor"]
