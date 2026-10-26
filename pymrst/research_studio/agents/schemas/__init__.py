"""Versioned persistent records exchanged by engineering agents."""

from .evidence import EvidenceKind, EvidenceReference
from .findings import (
    AGENT_FINDING_SCHEMA_VERSION,
    ClaimKind,
    EngineeringFinding,
    FindingSeverity,
)
from .recommendations import Recommendation, RecommendationStatus
from .run_request import RunRequest
from .run_result import RunFailure, RunResult, RunStatus
from .engineering_report import EngineeringReport, ReportSection

__all__ = [
    "AGENT_FINDING_SCHEMA_VERSION",
    "ClaimKind",
    "EngineeringFinding",
    "EvidenceKind",
    "EvidenceReference",
    "FindingSeverity",
    "EngineeringReport",
    "Recommendation",
    "RecommendationStatus",
    "ReportSection",
    "RunFailure",
    "RunRequest",
    "RunResult",
    "RunStatus",
]
