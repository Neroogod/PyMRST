"""Deterministic ensemble uncertainty specialist."""

from . import DeterministicSpecialist


class UncertaintySpecialist(DeterministicSpecialist):
    name = "uncertainty"
    tool_name = "analyze_ensemble"


__all__ = ["UncertaintySpecialist"]
