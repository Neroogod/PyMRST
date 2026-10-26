"""Deterministic specialist execution graphs and review-controlled synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ._serialization import require_fields
from .base import ToolCall
from .context import InvestigationContext
from .memory import EntryKind, InvestigationStatus
from .orchestrator import (
    InvestigationOutcome, InvestigationPlan, SingleCaseInvestigator,
)
from .policies.evidence import authorize_publication
from .review import (
    FindingConflict, HumanReviewDecision, ReviewAction, ReviewTarget,
    UncertaintyAssessment, assess_uncertainty, detect_finding_conflicts,
)


@dataclass(frozen=True, slots=True)
class InvestigationNode:
    node_id: str
    call: ToolCall
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("investigation node id cannot be empty")
        dependencies = tuple(str(item).strip() for item in self.dependencies)
        if any(not item for item in dependencies) or len(set(dependencies)) != len(dependencies):
            raise ValueError("node dependencies must be unique and nonempty")
        if node_id in dependencies:
            raise ValueError("an investigation node cannot depend on itself")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "dependencies", dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id, "call": self.call.to_dict(),
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationNode":
        data = require_fields(
            value, required={"node_id", "call", "dependencies"},
            name="InvestigationNode",
        )
        data["call"] = ToolCall.from_dict(data["call"])
        data["dependencies"] = tuple(data["dependencies"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class InvestigationGraph:
    graph_id: str
    nodes: tuple[InvestigationNode, ...]

    def __post_init__(self) -> None:
        graph_id = str(self.graph_id).strip()
        nodes = tuple(self.nodes)
        if not graph_id or not nodes:
            raise ValueError("an investigation graph requires an id and nodes")
        identifiers = [node.node_id for node in nodes]
        call_ids = [node.call.call_id for node in nodes]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("investigation graph node ids must be unique")
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("investigation graph call ids must be unique")
        unknown = sorted({item for node in nodes for item in node.dependencies} - set(identifiers))
        if unknown:
            raise ValueError(f"investigation graph has unknown dependencies: {unknown}")
        object.__setattr__(self, "graph_id", graph_id)
        object.__setattr__(self, "nodes", nodes)
        self.topological_levels()  # Validate acyclicity at construction.

    def topological_levels(self) -> tuple[tuple[InvestigationNode, ...], ...]:
        by_id = {node.node_id: node for node in self.nodes}
        remaining = set(by_id)
        completed: set[str] = set()
        levels = []
        while remaining:
            level_ids = [
                node.node_id for node in self.nodes
                if node.node_id in remaining
                and set(node.dependencies).issubset(completed)
            ]
            if not level_ids:
                raise ValueError("investigation graph contains a dependency cycle")
            levels.append(tuple(by_id[item] for item in level_ids))
            remaining.difference_update(level_ids)
            completed.update(level_ids)
        return tuple(levels)

    def execution_plan(self) -> InvestigationPlan:
        return InvestigationPlan(tuple(
            node.call for level in self.topological_levels() for node in level
        ))

    def to_dict(self) -> dict[str, Any]:
        return {"graph_id": self.graph_id, "nodes": [node.to_dict() for node in self.nodes]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationGraph":
        data = require_fields(
            value, required={"graph_id", "nodes"}, name="InvestigationGraph",
        )
        data["nodes"] = tuple(InvestigationNode.from_dict(item) for item in data["nodes"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class GraphInvestigationOutcome:
    graph: InvestigationGraph
    investigation: InvestigationOutcome
    conflicts: tuple[FindingConflict, ...] = ()
    uncertainties: tuple[UncertaintyAssessment, ...] = ()
    reviews: tuple[HumanReviewDecision, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflicts", tuple(self.conflicts))
        object.__setattr__(self, "uncertainties", tuple(self.uncertainties))
        object.__setattr__(self, "reviews", tuple(self.reviews))

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "investigation": self.investigation.to_dict(),
            "conflicts": [item.to_dict() for item in self.conflicts],
            "uncertainties": [item.to_dict() for item in self.uncertainties],
            "reviews": [item.to_dict() for item in self.reviews],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphInvestigationOutcome":
        data = require_fields(value, required={
            "graph", "investigation", "conflicts", "uncertainties", "reviews",
        }, name="GraphInvestigationOutcome")
        data["graph"] = InvestigationGraph.from_dict(data["graph"])
        data["investigation"] = InvestigationOutcome.from_dict(data["investigation"])
        data["conflicts"] = tuple(FindingConflict.from_dict(x) for x in data["conflicts"])
        data["uncertainties"] = tuple(
            UncertaintyAssessment.from_dict(x) for x in data["uncertainties"]
        )
        data["reviews"] = tuple(HumanReviewDecision.from_dict(x) for x in data["reviews"])
        return cls(**data)


class SpecialistGraphInvestigator:
    """Execute a graph and synthesize findings under explicit human review."""

    def __init__(self, pipeline: SingleCaseInvestigator) -> None:
        self.pipeline = pipeline

    @staticmethod
    def _validate_reviews(
        reviews: tuple[HumanReviewDecision, ...],
        finding_ids: set[str], conflict_ids: set[str], investigation_id: str,
    ) -> None:
        decision_ids = [item.decision_id for item in reviews]
        targets = [(item.target, item.target_id) for item in reviews]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("human review decision ids must be unique")
        if len(set(targets)) != len(targets):
            raise ValueError("only one human decision is allowed per review target")
        for item in reviews:
            valid = {
                ReviewTarget.FINDING: finding_ids,
                ReviewTarget.CONFLICT: conflict_ids,
                ReviewTarget.REPORT: {investigation_id},
            }[item.target]
            if item.target_id not in valid:
                raise ValueError(f"review targets unknown {item.target.value}: {item.target_id}")

    def investigate(
        self,
        context: InvestigationContext,
        graph: InvestigationGraph,
        *,
        title: str,
        reviews: Iterable[HumanReviewDecision] = (),
        approved_call_ids: Iterable[str] = (),
        qualification_requested: bool = False,
        assumptions: Iterable[str] = (),
        limitations: Iterable[str] = (),
    ) -> GraphInvestigationOutcome:
        base = self.pipeline.investigate(
            context, graph.execution_plan(), title=title,
            approved_call_ids=approved_call_ids, build_report=False,
        )
        record = base.record.append(EntryKind.EXECUTION_GRAPH, graph.to_dict())
        if record.status is not InvestigationStatus.COMPLETED:
            failed = InvestigationOutcome(
                record, base.findings, None, base.failure_reason,
            )
            return GraphInvestigationOutcome(graph, failed)

        reviews = tuple(reviews)
        provisional = detect_finding_conflicts(base.findings)
        self._validate_reviews(
            reviews, {item.finding_id for item in base.findings},
            {item.conflict_id for item in provisional}, context.investigation_id,
        )
        conflicts = detect_finding_conflicts(base.findings, reviews)
        uncertainties = assess_uncertainty(base.findings)
        for conflict in conflicts:
            record = record.append(EntryKind.CONFLICT, conflict.to_dict())
        for uncertainty in uncertainties:
            record = record.append(EntryKind.UNCERTAINTY, uncertainty.to_dict())
        for review in reviews:
            record = record.record_human_decision(review)

        unresolved = tuple(item for item in conflicts if not item.resolved)
        negative_findings = tuple(
            item for item in reviews
            if item.target is ReviewTarget.FINDING
            and item.action in {ReviewAction.REJECT, ReviewAction.REQUEST_CHANGES}
        )
        report_reviews = tuple(
            item for item in reviews if item.target is ReviewTarget.REPORT
        )
        qualified = False
        reason = None
        status = InvestigationStatus.COMPLETED
        if negative_findings:
            status = InvestigationStatus.UNQUALIFIED
            reason = "human review rejected or requested changes to one or more findings"
        elif unresolved:
            status = InvestigationStatus.WAITING_APPROVAL
            reason = f"{len(unresolved)} potential finding conflict(s) require human resolution"
        elif qualification_requested:
            if not report_reviews:
                status = InvestigationStatus.WAITING_APPROVAL
                reason = "qualified reporting requires explicit human report approval"
            elif report_reviews[0].action is not ReviewAction.APPROVE:
                status = InvestigationStatus.UNQUALIFIED
                reason = f"human report review action is {report_reviews[0].action.value}"
            else:
                decision = authorize_publication(
                    base.findings, qualified_run=True,
                    policy=self.pipeline.evidence_policy,
                )
                record = record.record_policy_decision("__report__", decision)
                if decision.allowed:
                    qualified = True
                else:
                    status = InvestigationStatus.UNQUALIFIED
                    reason = decision.reason

        report_limitations = list(limitations)
        report_limitations.extend(
            f"Unresolved conflict {item.conflict_id}: {item.reason}" for item in unresolved
        )
        report_limitations.extend(item.explanation for item in uncertainties)
        report = self.pipeline.report_builder.build(
            context, base.findings, title=title, qualified=qualified,
            assumptions=assumptions, limitations=report_limitations,
            review_notes=(
                f"{item.reviewer} recorded `{item.action.value}` for "
                f"{item.target.value} `{item.target_id}` in decision "
                f"`{item.decision_id}`: {item.rationale}"
                for item in reviews
            ),
        )
        record = record.record_report(report).with_status(status)
        investigation = InvestigationOutcome(
            record, base.findings, report, reason,
        )
        return GraphInvestigationOutcome(
            graph, investigation, conflicts, uncertainties, reviews,
        )


__all__ = [
    "GraphInvestigationOutcome", "InvestigationGraph", "InvestigationNode",
    "SpecialistGraphInvestigator",
]
