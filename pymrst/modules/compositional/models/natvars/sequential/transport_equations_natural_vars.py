"""MRST ``transportEquationsNaturalVars.m`` compatibility helper."""

from pymrst.modules.compositional.models.natvars.sequential.transport_natural_variables_model import (
    transport_equations_natural_vars,
)

transportEquationsNaturalVars = transport_equations_natural_vars

__all__ = ["transport_equations_natural_vars", "transportEquationsNaturalVars"]
