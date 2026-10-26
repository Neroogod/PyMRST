"""Well, completion, segment, control, group, and schedule specialist."""

from . import DeterministicSpecialist


class WellsScheduleSpecialist(DeterministicSpecialist):
    name = "wells_schedule"
    tool_name = "analyze_wells_schedule"


__all__ = ["WellsScheduleSpecialist"]
