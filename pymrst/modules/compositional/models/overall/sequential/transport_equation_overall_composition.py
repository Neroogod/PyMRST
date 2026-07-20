"""MRST ``transportEquationOverallComposition.m`` compatibility helper."""

from pymrst.modules.compositional.models.overall.sequential.transport_equation_compositional import (
    transportEquationCompositional,
    transport_equation_compositional,
)

transport_equation_overall_composition = transport_equation_compositional
transportEquationOverallComposition = transportEquationCompositional

__all__ = [
    "transport_equation_overall_composition",
    "transportEquationOverallComposition",
    "transport_equation_compositional",
    "transportEquationCompositional",
]
