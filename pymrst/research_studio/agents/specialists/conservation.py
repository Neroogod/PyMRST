"""Component conservation, material balance, and invariant specialist."""

from . import DeterministicSpecialist


class ConservationSpecialist(DeterministicSpecialist):
    name = "conservation"
    tool_name = "analyze_conservation"


class InvariantSpecialist(DeterministicSpecialist):
    name = "conservation"
    tool_name = "analyze_invariants"


__all__ = ["ConservationSpecialist", "InvariantSpecialist"]
