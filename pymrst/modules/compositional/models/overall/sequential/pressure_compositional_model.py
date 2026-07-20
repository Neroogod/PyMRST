"""MRST ``PressureCompositionalModel.m`` compatibility landing module."""

from pymrst.modules.compositional.model import CompositionalState, OverallCompositionModel

PressureCompositionalModel = OverallCompositionModel

__all__ = ["PressureCompositionalModel", "OverallCompositionModel", "CompositionalState"]
