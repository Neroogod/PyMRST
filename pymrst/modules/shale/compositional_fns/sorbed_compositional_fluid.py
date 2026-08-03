"""Sorbed compositional fluid (MRST ``SorbedCompositionalFluid``)."""

import numpy as np

from pymrst.modules.compositional.fluids.table_compositional_mixture import (
    TableCompositionalMixture,
)


class SorbedCompositionalFluid(TableCompositionalMixture):
    """Table-backed mixture carrying per-component Langmuir parameters."""

    def __init__(self, names, isotherm):
        super().__init__(names)
        values = np.asarray(isotherm, dtype=float)
        if values.ndim != 2:
            raise ValueError("isotherm must be a two-dimensional array")
        if values.shape == (self.n, 2):
            values = values.T
        if values.shape != (2, self.n):
            raise ValueError(
                f"isotherm must have shape (2, {self.n}) or ({self.n}, 2)"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("isotherm entries must be finite and non-negative")
        self.isotherm = values

    @property
    def langmuir_pressure(self) -> np.ndarray:
        return self.isotherm[0]

    @property
    def langmuir_density(self) -> np.ndarray:
        return self.isotherm[1]


__all__ = ["SorbedCompositionalFluid"]
