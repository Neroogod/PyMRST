"""Acceptance tests for deterministic cross-specialist synthesis."""

from pymrst.research_studio.agents.schemas.evidence import EvidenceReference
from pymrst.research_studio.agents.schemas.findings import EngineeringFinding
from pymrst.research_studio.agents.synthesis import FindingSynthesis, synthesize_findings


def finding(identifier, agent, *, statement="Pressure decline is physically consistent."):
    return EngineeringFinding(
        identifier, agent, "pressure.depletion", "info", "inferred", statement,
        (EvidenceReference(
            "state", f"run://reference/{identifier}", identifier,
            {"field": "pressure"}, "a" * 64,
        ),),
        0.85, ("field:pressure",),
    )


def test_identical_independent_claims_are_corroborated_not_conflicted():
    result = synthesize_findings((
        finding("left", "reservoir_performance"),
        finding("right", "material_balance"),
    ))
    assert result.conflicts == ()
    assert len(result.corroborations) == 1
    assert result.corroborations[0].finding_ids == ("left", "right")
    assert len(result.uncertainties) == 2
    assert result.summary["evidence_reference_count"] == 2
    assert FindingSynthesis.from_dict(result.to_dict()) == result


def test_divergent_claims_are_conflicts_and_never_corroborations():
    result = synthesize_findings((
        finding("left", "reservoir_performance"),
        finding(
            "right", "material_balance",
            statement="Pressure decline violates material balance.",
        ),
    ))
    assert len(result.conflicts) == 1
    assert result.corroborations == ()
    assert result.summary["unresolved_conflict_count"] == 1

