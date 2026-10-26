"""Evidence-bound narrative synthesis for engineering investigations.

Language models are presentation components here, never sources of truth.  A
provider may arrange accepted findings into prose, while PyMRST independently
checks its references and attaches the authoritative evidence objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from ._serialization import require_fields
from .review import assess_uncertainty
from .schemas.evidence import EvidenceReference
from .schemas.findings import EngineeringFinding


NARRATIVE_PROPOSAL_SCHEMA_VERSION = 1
EVIDENCE_LINKED_NARRATIVE_SCHEMA_VERSION = 1
_NUMBER = re.compile(r"(?<![\w])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _clean(value: Any, name: str, *, maximum: int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return text


@dataclass(frozen=True, slots=True)
class NarrativeProposalSection:
    section_id: str
    heading: str
    narrative: str
    finding_ids: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "section_id", _clean(self.section_id, "section_id", maximum=80))
        object.__setattr__(self, "heading", _clean(self.heading, "heading", maximum=160))
        object.__setattr__(self, "narrative", _clean(self.narrative, "narrative", maximum=8_000))
        identifiers = tuple(str(item).strip() for item in self.finding_ids)
        if not identifiers or any(not item for item in identifiers):
            raise ValueError("narrative sections require finding ids")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("narrative section finding ids must be unique")
        object.__setattr__(self, "finding_ids", identifiers)

    def to_dict(self):
        return {
            "section_id": self.section_id, "heading": self.heading,
            "narrative": self.narrative, "finding_ids": list(self.finding_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = require_fields(value, required={
            "section_id", "heading", "narrative", "finding_ids",
        }, name="NarrativeProposalSection")
        data["finding_ids"] = tuple(data["finding_ids"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class NarrativeProposal:
    title: str
    executive_summary: str
    sections: tuple[NarrativeProposalSection, ...]
    schema_version: int = NARRATIVE_PROPOSAL_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != NARRATIVE_PROPOSAL_SCHEMA_VERSION:
            raise ValueError("unsupported NarrativeProposal schema version")
        object.__setattr__(self, "title", _clean(self.title, "title", maximum=200))
        object.__setattr__(self, "executive_summary", _clean(
            self.executive_summary, "executive_summary", maximum=4_000,
        ))
        sections = tuple(
            item if isinstance(item, NarrativeProposalSection)
            else NarrativeProposalSection.from_dict(item)
            for item in self.sections
        )
        identifiers = [item.section_id for item in sections]
        if not sections or len(sections) > 24:
            raise ValueError("narrative proposals require between 1 and 24 sections")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("narrative proposal section ids must be unique")
        object.__setattr__(self, "sections", sections)

    def to_dict(self):
        return {
            "schema_version": self.schema_version, "title": self.title,
            "executive_summary": self.executive_summary,
            "sections": [item.to_dict() for item in self.sections],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = require_fields(value, required={
            "schema_version", "title", "executive_summary", "sections",
        }, name="NarrativeProposal")
        data["sections"] = tuple(NarrativeProposalSection.from_dict(x) for x in data["sections"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EvidenceLinkedNarrativeSection:
    section_id: str
    heading: str
    narrative: str
    finding_ids: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    uncertainty_caveats: tuple[str, ...] = ()

    def to_dict(self):
        return {
            "section_id": self.section_id, "heading": self.heading,
            "narrative": self.narrative, "finding_ids": list(self.finding_ids),
            "evidence": [item.to_dict() for item in self.evidence],
            "uncertainty_caveats": list(self.uncertainty_caveats),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = require_fields(value, required={
            "section_id", "heading", "narrative", "finding_ids", "evidence",
            "uncertainty_caveats",
        }, name="EvidenceLinkedNarrativeSection")
        return cls(
            data["section_id"], data["heading"], data["narrative"],
            tuple(data["finding_ids"]),
            tuple(EvidenceReference.from_dict(x) for x in data["evidence"]),
            tuple(data["uncertainty_caveats"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceLinkedNarrative:
    narrative_id: str
    investigation_id: str
    provider: str
    provider_configuration_hash: str
    source_findings_hash: str
    title: str
    executive_summary: str
    sections: tuple[EvidenceLinkedNarrativeSection, ...]
    status: str = "draft"
    schema_version: int = EVIDENCE_LINKED_NARRATIVE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != EVIDENCE_LINKED_NARRATIVE_SCHEMA_VERSION:
            raise ValueError("unsupported EvidenceLinkedNarrative schema version")
        for name in ("narrative_id", "investigation_id", "provider",
                     "provider_configuration_hash", "source_findings_hash", "title",
                     "executive_summary"):
            object.__setattr__(self, name, _clean(getattr(self, name), name, maximum=8_000))
        if self.status not in {"draft", "approved", "rejected"}:
            raise ValueError("invalid narrative status")
        sections = tuple(self.sections)
        if not sections:
            raise ValueError("evidence-linked narrative requires sections")
        object.__setattr__(self, "sections", sections)

    def to_dict(self):
        return {
            "schema_version": self.schema_version, "narrative_id": self.narrative_id,
            "investigation_id": self.investigation_id, "provider": self.provider,
            "provider_configuration_hash": self.provider_configuration_hash,
            "source_findings_hash": self.source_findings_hash, "title": self.title,
            "executive_summary": self.executive_summary,
            "sections": [item.to_dict() for item in self.sections], "status": self.status,
        }

    @property
    def content_hash(self):
        # Review state is a separate immutable human decision, not authored
        # narrative content, so approval cannot change this identity.
        value = self.to_dict()
        value.pop("status")
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = require_fields(value, required={
            "schema_version", "narrative_id", "investigation_id", "provider",
            "provider_configuration_hash", "source_findings_hash", "title",
            "executive_summary", "sections", "status",
        }, name="EvidenceLinkedNarrative")
        data["sections"] = tuple(EvidenceLinkedNarrativeSection.from_dict(x) for x in data["sections"])
        return cls(**data)


def _deduplicate_evidence(findings: Iterable[EngineeringFinding]):
    output, seen = [], set()
    for finding in findings:
        for item in finding.evidence:
            key = json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                output.append(item)
    return tuple(output)


def _numeric_tokens(value: str):
    return {_normalize_number(match.group()) for match in _NUMBER.finditer(value)}


def _normalize_number(value: str):
    try:
        return format(float(value), ".15g")
    except ValueError:
        return value


def source_payload(question: str, findings: Iterable[EngineeringFinding]):
    """Return the only project-derived payload an external narrative provider sees."""
    return {
        "question": str(question).strip(),
        "accepted_findings": [
            {
                "finding_id": item.finding_id, "category": item.category,
                "severity": item.severity.value, "claim_kind": item.claim_kind.value,
                "statement": item.statement, "confidence": item.confidence,
                "affected_objects": list(item.affected_objects),
                "has_recommendation": item.recommendation is not None,
            }
            for item in findings
        ],
    }


def findings_hash(findings: Iterable[EngineeringFinding]) -> str:
    return hashlib.sha256(json.dumps(
        [item.to_dict() for item in findings],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


class DeterministicNarrativeProvider:
    """Offline baseline that reproduces finding statements without paraphrasing."""

    def complete(self, messages, tools, response_type):
        if response_type is not NarrativeProposal or tools:
            raise TypeError("deterministic narrative provider only produces narratives")
        payload = messages[-1]["content"]
        findings = tuple(payload["accepted_findings"])
        sections = tuple(NarrativeProposalSection(
            f"finding_{index + 1}", item["category"].replace("_", " ").title(),
            item["statement"], (item["finding_id"],),
        ) for index, item in enumerate(findings))
        return NarrativeProposal(
            "Evidence-bound investigation answer",
            "The answer below is limited to human-accepted, evidence-backed findings.",
            sections,
        )


class NarrativeSynthesizer:
    """Validate provider prose and bind it to authoritative finding evidence."""

    def synthesize(
        self, *, investigation_id: str, question: str,
        findings: Iterable[EngineeringFinding], provider, provider_name: str,
        provider_configuration_hash: str,
    ) -> EvidenceLinkedNarrative:
        findings = tuple(findings)
        if not findings:
            raise ValueError("narrative synthesis requires accepted findings")
        lookup = {item.finding_id: item for item in findings}
        if len(lookup) != len(findings):
            raise ValueError("narrative source finding ids must be unique")
        payload = source_payload(question, findings)
        proposal = provider.complete((
            {"role": "system", "content": (
                "Organize only the supplied accepted findings. Every section must cite "
                "finding_ids. Do not introduce facts, numbers, links, or recommendations."
            )},
            {"role": "user", "content": payload},
        ), (), NarrativeProposal)
        if not isinstance(proposal, NarrativeProposal):
            raise TypeError("narrative provider returned the wrong response type")
        referenced = {fid for section in proposal.sections for fid in section.finding_ids}
        unknown = referenced - set(lookup)
        missing = set(lookup) - referenced
        if unknown:
            raise ValueError(f"narrative referenced unknown findings: {sorted(unknown)}")
        if missing:
            raise ValueError(f"narrative omitted accepted findings: {sorted(missing)}")
        prose = " ".join((proposal.title, proposal.executive_summary,
                          *(item.heading for item in proposal.sections),
                          *(item.narrative for item in proposal.sections)))
        if "://" in prose:
            raise ValueError("provider narratives cannot introduce external links")
        # Numbers are allowed only when they appeared in the bounded source
        # payload (statement, confidence, category, or affected-object label).
        source_numbers = _numeric_tokens(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ))
        invented = _numeric_tokens(prose) - source_numbers
        if invented:
            raise ValueError(f"provider narrative introduced unsupported numbers: {sorted(invented)}")
        uncertainty = {item.finding_id: item for item in assess_uncertainty(findings)}
        sections = []
        for section in proposal.sections:
            linked = tuple(lookup[fid] for fid in section.finding_ids)
            caveats = tuple(uncertainty[fid].explanation for fid in section.finding_ids
                            if fid in uncertainty)
            sections.append(EvidenceLinkedNarrativeSection(
                section.section_id, section.heading, section.narrative,
                section.finding_ids, _deduplicate_evidence(linked), caveats,
            ))
        finding_hash = findings_hash(findings)
        identity = hashlib.sha256(json.dumps({
            "investigation_id": investigation_id, "provider": provider_name,
            "provider_configuration_hash": provider_configuration_hash,
            "source_findings_hash": finding_hash, "proposal": proposal.to_dict(),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
        return EvidenceLinkedNarrative(
            f"narrative_{identity}", investigation_id, provider_name,
            provider_configuration_hash, finding_hash, proposal.title,
            proposal.executive_summary, tuple(sections), "draft",
        )


__all__ = [
    "DeterministicNarrativeProvider", "EvidenceLinkedNarrative",
    "EvidenceLinkedNarrativeSection", "NarrativeProposal",
    "NarrativeProposalSection", "NarrativeSynthesizer", "source_payload",
    "findings_hash",
]
