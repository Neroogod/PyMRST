"""MRST ``TransportCompositionalModel.m`` compatibility landing module."""

from pymrst.modules.compositional.model import CompositionalState, OverallCompositionModel
from pymrst.modules.compositional.models.overall.sequential.transport_overall_composition_model import (
    TransportOverallCompositionMixin,
)

TransportCompositionalModel = OverallCompositionModel

__all__ = [
    "TransportCompositionalModel",
    "TransportOverallCompositionMixin",
    "OverallCompositionModel",
    "CompositionalState",
]
