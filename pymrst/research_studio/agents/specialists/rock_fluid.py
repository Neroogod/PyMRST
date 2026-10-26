"""Rock, PVT, saturation-function, region, EOS, and flash specialist."""

from . import DeterministicSpecialist


class RockFluidSpecialist(DeterministicSpecialist):
    name = "rock_fluid"
    tool_name = "analyze_rock_fluid"


__all__ = ["RockFluidSpecialist"]
