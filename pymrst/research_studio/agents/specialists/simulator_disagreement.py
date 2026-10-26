"""Deterministic cross-simulator disagreement specialist."""

from . import DeterministicSpecialist


class SimulatorDisagreementSpecialist(DeterministicSpecialist):
    name = "simulator_disagreement"
    tool_name = "compare_reference_runs"


__all__ = ["SimulatorDisagreementSpecialist"]
