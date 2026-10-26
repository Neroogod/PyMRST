"""Contacts, equilibration, initial state, and in-place-volume specialist."""

from . import DeterministicSpecialist


class InitializationSpecialist(DeterministicSpecialist):
    name = "initialization"
    tool_name = "analyze_initialization"


__all__ = ["InitializationSpecialist"]
