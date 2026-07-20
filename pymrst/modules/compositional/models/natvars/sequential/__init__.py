"""MRST ``models/natvars/sequential`` compatibility layout."""

from pymrst.modules.compositional.models.natvars.sequential.pressure_natural_variables_model import (
    PressureNaturalVariablesModel,
    equations_natural_variables_pressure,
)
from pymrst.modules.compositional.models.natvars.sequential.pressure_reduced_linear_system import (
    PressureReducedLinearSystem,
)
from pymrst.modules.compositional.models.natvars.sequential.transport_natural_variables_model import (
    TransportNaturalVariablesModel,
    transport_equations_natural_vars as _transport_equations_natural_vars,
)
from pymrst.modules.compositional.models.natvars.sequential.transport_equations_natural_vars import (
    transportEquationsNaturalVars,
)
from pymrst.modules.compositional.models.natvars.sequential.utils import (
    correctSequentialBHP,
    correct_sequential_bhp,
    getUpdatesFromPressure,
    get_updates_from_pressure,
)

transport_equations_natural_vars = _transport_equations_natural_vars

__all__ = [
    "PressureNaturalVariablesModel",
    "PressureReducedLinearSystem",
    "TransportNaturalVariablesModel",
    "equations_natural_variables_pressure",
    "transport_equations_natural_vars",
    "transportEquationsNaturalVars",
    "correctSequentialBHP",
    "correct_sequential_bhp",
    "getUpdatesFromPressure",
    "get_updates_from_pressure",
]
