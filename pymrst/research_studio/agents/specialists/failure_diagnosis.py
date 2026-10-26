"""Deterministic interpreter for structured simulation failure diagnoses."""

from . import DeterministicSpecialist


class FailureDiagnosisSpecialist(DeterministicSpecialist):
    name = "failure_diagnosis"
    tool_name = "diagnose_run_failure"


__all__ = ["FailureDiagnosisSpecialist"]
