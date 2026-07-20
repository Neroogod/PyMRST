"""MRST ``models/natvars/sequential/utils`` compatibility layout."""

from pymrst.modules.compositional.models.natvars.sequential.utils.correct_sequential_bhp import (
    correctSequentialBHP,
    correct_sequential_bhp,
)
from pymrst.modules.compositional.models.natvars.sequential.utils.get_updates_from_pressure import (
    getUpdatesFromPressure,
    get_updates_from_pressure,
)

__all__ = [
    "correctSequentialBHP",
    "correct_sequential_bhp",
    "getUpdatesFromPressure",
    "get_updates_from_pressure",
]
