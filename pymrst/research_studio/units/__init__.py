"""Opt-in dimensional checking (M2.12). Inert unless a caller supplies a Quantity."""

from pymrst.research_studio.units.dimensions import (
    AREA, COMPRESSIBILITY, DENSITY, DIMENSIONLESS, Dimension, DimensionError,
    LENGTH, MASS, PERMEABILITY, PRESSURE, Quantity, RATE, TIME, VISCOSITY,
    VOLUME, declared_dimensions, expects, magnitude,
)

__all__ = ["AREA", "COMPRESSIBILITY", "DENSITY", "DIMENSIONLESS", "Dimension",
           "DimensionError", "LENGTH", "MASS", "PERMEABILITY", "PRESSURE",
           "Quantity", "RATE", "TIME", "VISCOSITY", "VOLUME",
           "declared_dimensions", "expects", "magnitude"]
