"""Compatibility landing file for MRST `computeSequentialFluxes.m`.

PyMRST's validated oil-water sequential solver currently keeps the fractional
flow flux construction inside `SequentialOilWaterModel`.
"""

from pymrst.modules.sequential.seq_model import SequentialOilWaterModel

__all__ = ["SequentialOilWaterModel"]
