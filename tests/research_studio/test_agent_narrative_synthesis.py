"""Acceptance tests for evidence-bound provider narrative synthesis."""

import pytest

from pymrst.research_studio.agents.narrative import (
    DeterministicNarrativeProvider, NarrativeProposal,
    NarrativeProposalSection, NarrativeSynthesizer,
)
from pymrst.research_studio.agents.schemas.evidence import EvidenceReference
from pymrst.research_studio.agents.schemas.findings import EngineeringFinding


def finding(identifier="finding_pressure", *, statement="Pressure differs by 1.573 Pa.",
            claim_kind="calculated", confidence=1.0):
    return EngineeringFinding(
        identifier, "comparison", "pressure.parity", "info", claim_kind,
        statement,
        (EvidenceReference(
            "validation", f"validation://comparison/{identifier}", "Pressure comparison",
            {"field": "pressure"}, "a" * 64,
        ),),
        confidence, ("field:pressure",),
    )


class ProposalProvider:
    def __init__(self, proposal):
        self.proposal = proposal

    def complete(self, messages, tools, response_type):
        assert tools == ()
        assert set(messages[-1]["content"]) == {"question", "accepted_findings"}
        assert "evidence" not in str(messages[-1]["content"])
        assert response_type is NarrativeProposal
        return self.proposal


def synthesize(provider, findings):
    return NarrativeSynthesizer().synthesize(
        investigation_id="investigation_1", question="Do the results agree?",
        findings=findings, provider=provider, provider_name="test-provider",
        provider_configuration_hash="b" * 64,
    )


def test_deterministic_narrative_binds_authoritative_evidence_and_roundtrips():
    result = synthesize(DeterministicNarrativeProvider(), (finding(),))
    assert result.status == "draft"
    assert result.sections[0].finding_ids == ("finding_pressure",)
    assert result.sections[0].evidence[0].uri.startswith("validation://")
    assert type(result).from_dict(result.to_dict()) == result


def test_unknown_or_omitted_findings_are_rejected():
    unknown = NarrativeProposal(
        "Answer", "Accepted evidence supports the answer.",
        (NarrativeProposalSection("result", "Result", "Pressure agrees.", ("unknown",)),),
    )
    with pytest.raises(ValueError, match="unknown findings"):
        synthesize(ProposalProvider(unknown), (finding(),))

    omitted = NarrativeProposal(
        "Answer", "Accepted evidence supports the answer.",
        (NarrativeProposalSection("result", "Result", "Pressure agrees.",
                                  ("finding_pressure",)),),
    )
    with pytest.raises(ValueError, match="omitted accepted findings"):
        synthesize(ProposalProvider(omitted), (finding(), finding("finding_second")))


def test_provider_cannot_invent_numbers_or_links():
    invented = NarrativeProposal(
        "Answer", "The difference is 99 Pa.",
        (NarrativeProposalSection("result", "Result", "Pressure differs by 1.573 Pa.",
                                  ("finding_pressure",)),),
    )
    with pytest.raises(ValueError, match="unsupported numbers"):
        synthesize(ProposalProvider(invented), (finding(),))
    linked = NarrativeProposal(
        "Answer", "See https://example.invalid for details.",
        (NarrativeProposalSection("result", "Result", "Pressure differs by 1.573 Pa.",
                                  ("finding_pressure",)),),
    )
    with pytest.raises(ValueError, match="external links"):
        synthesize(ProposalProvider(linked), (finding(),))


def test_uncertainty_language_is_attached_by_pymrst_not_the_provider():
    uncertain = finding(claim_kind="inferred", confidence=0.7)
    result = synthesize(DeterministicNarrativeProvider(), (uncertain,))
    caveat = result.sections[0].uncertainty_caveats[0]
    assert "inferred" in caveat and "0.700" in caveat

