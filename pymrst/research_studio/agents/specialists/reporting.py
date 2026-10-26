"""Evidence-linked engineering-report composition specialist."""

from . import DeterministicSpecialist


class ReportingSpecialist(DeterministicSpecialist):
    name = "reporting"
    tool_name = "analyze_report_composition"


__all__ = ["ReportingSpecialist"]
