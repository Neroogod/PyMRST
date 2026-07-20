"""MRST `modules/compositional/models/natvars` compatibility layout."""

from pymrst.modules.compositional.natural_variables import (
    GenericNaturalVariablesModel,
    NaturalVariablesCompositionalModel,
    NaturalVariablesState,
)
from pymrst.modules.compositional.models.natvars.reduced_linearized_system import (
    ReducedLinearizedSystem,
)
from pymrst.modules.compositional.models.natvars.equations_natural_variables import (
    NaturalVariablesProblem,
    equationsNaturalVariables,
    equations_natural_variables,
)
from pymrst.modules.compositional.models.natvars.sequential import (
    PressureNaturalVariablesModel,
    PressureReducedLinearSystem,
    TransportNaturalVariablesModel,
)

__all__ = [
    "GenericNaturalVariablesModel",
    "NaturalVariablesCompositionalModel",
    "NaturalVariablesState",
    "ReducedLinearizedSystem",
    "NaturalVariablesProblem",
    "equationsNaturalVariables",
    "equations_natural_variables",
    "PressureNaturalVariablesModel",
    "PressureReducedLinearSystem",
    "TransportNaturalVariablesModel",
]
