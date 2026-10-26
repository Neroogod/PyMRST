"""Policies governing support, confidence, provenance, and report claims."""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.findings import ClaimKind, EngineeringFinding
from .execution import DecisionCode, PolicyDecision


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    minimum_observed_confidence: float = 1.0
    minimum_calculated_confidence: float = 0.95
    minimum_inferred_confidence: float = 0.5
    require_checksums_for_publication: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_observed_confidence", "minimum_calculated_confidence",
            "minimum_inferred_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
            object.__setattr__(self, name, value)

    def threshold(self, claim_kind: ClaimKind) -> float:
        return {
            ClaimKind.OBSERVED: self.minimum_observed_confidence,
            ClaimKind.CALCULATED: self.minimum_calculated_confidence,
            ClaimKind.INFERRED: self.minimum_inferred_confidence,
            ClaimKind.UNCERTAIN: 0.0,
        }[ClaimKind(claim_kind)]


def qualify_finding(
    finding: EngineeringFinding,
    policy: EvidencePolicy = EvidencePolicy(),
    *,
    for_publication: bool = False,
) -> PolicyDecision:
    if not finding.evidence:
        return PolicyDecision(False, DecisionCode.DENY_EVIDENCE,
                              "finding has no supporting evidence")
    threshold = policy.threshold(finding.claim_kind)
    if finding.confidence < threshold:
        return PolicyDecision(
            False, DecisionCode.DENY_EVIDENCE,
            f"confidence {finding.confidence:g} is below the "
            f"{finding.claim_kind.value} threshold {threshold:g}",
        )
    if for_publication and policy.require_checksums_for_publication:
        missing = [item.uri for item in finding.evidence if item.checksum is None]
        if missing:
            return PolicyDecision(
                False, DecisionCode.DENY_EVIDENCE,
                f"publication evidence lacks checksums: {missing}",
            )
    return PolicyDecision(True, DecisionCode.ALLOW, "finding satisfies evidence policy")


def authorize_publication(
    findings: tuple[EngineeringFinding, ...],
    *,
    qualified_run: bool,
    policy: EvidencePolicy = EvidencePolicy(),
) -> PolicyDecision:
    if not qualified_run:
        return PolicyDecision(
            False, DecisionCode.DENY_QUALIFICATION,
            "a completed run must pass requested qualification before publication",
        )
    for finding in findings:
        decision = qualify_finding(finding, policy, for_publication=True)
        if not decision.allowed:
            return decision
    return PolicyDecision(True, DecisionCode.ALLOW,
                          "qualified run and all findings are publication-ready")


__all__ = ["EvidencePolicy", "authorize_publication", "qualify_finding"]
