"""Specialist reservoir-engineering agents.

Specialists interpret deterministic tool outputs; they do not implement or
replace PyMRST's numerical physics.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from ..base import ToolResult
from ..schemas.evidence import EvidenceReference
from ..schemas.findings import (
    ClaimKind, EngineeringFinding, FindingSeverity,
)


class SpecialistInputError(ValueError):
    pass


class DeterministicSpecialist:
    """Convert one deterministic tool response into typed findings."""

    name = "specialist"
    tool_name = ""

    def _evidence(self, result: ToolResult) -> tuple[EvidenceReference, ...]:
        evidence_value = result.value.get("evidence")
        if evidence_value is None:
            raise SpecialistInputError("tool result has no evidence reference")
        evidence_values = (
            evidence_value if isinstance(evidence_value, tuple)
            else (evidence_value,)
        )
        return tuple(
            item if isinstance(item, EvidenceReference)
            else EvidenceReference.from_dict(item) for item in evidence_values
        )

    def make_finding(
        self,
        result: ToolResult,
        *,
        category: str,
        severity: FindingSeverity,
        statement: str,
        affected_objects=(),
        claim_kind: ClaimKind = ClaimKind.CALCULATED,
        confidence: float = 1.0,
        ordinal: int = 0,
        recommendation=None,
    ) -> EngineeringFinding:
        evidence = self._evidence(result)
        source_identity = "|".join(
            f"{item.uri}@{item.checksum or ''}" for item in evidence
        )
        identity = hashlib.sha256(
            f"{self.name}|{source_identity}|{ordinal}|{statement}".encode()
        ).hexdigest()[:20]
        return EngineeringFinding(
            finding_id=f"finding_{identity}", agent=self.name,
            category=category, severity=severity, claim_kind=claim_kind,
            statement=statement, evidence=evidence, confidence=confidence,
            affected_objects=tuple(affected_objects),
            recommendation=recommendation,
        )

    def analyze(self, result: ToolResult) -> tuple[EngineeringFinding, ...]:
        if not result.ok:
            raise SpecialistInputError("specialists require a successful tool result")
        actual = result.value.get("analysis_kind")
        if actual != self.tool_name:
            raise SpecialistInputError(
                f"{self.name} requires {self.tool_name!r} output, received {actual!r}"
            )
        output = []
        for ordinal, item in enumerate(result.value.get("findings", ())):
            severity = FindingSeverity(item["severity"])
            statement = str(item["statement"])
            output.append(self.make_finding(
                result, category=str(item["category"]), severity=severity,
                statement=statement,
                affected_objects=tuple(item.get("affected_objects", ())),
                ordinal=ordinal,
                recommendation=item.get("recommendation"),
            ))
        return tuple(output)


__all__ = ["DeterministicSpecialist", "SpecialistInputError"]
