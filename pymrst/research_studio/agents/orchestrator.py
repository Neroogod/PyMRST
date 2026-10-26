"""Deterministic, permission-gated orchestration for engineering tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

from ._serialization import require_fields
from .base import AgentTool, ToolCall, ToolResult
from .context import InvestigationContext
from .memory import EntryKind, InvestigationRecord, InvestigationStatus
from .policies.evidence import EvidencePolicy, authorize_publication, qualify_finding
from .policies.execution import DecisionCode, authorize_tool
from .schemas.engineering_report import EngineeringReport
from .schemas.findings import EngineeringFinding
from .specialists import DeterministicSpecialist
from .tools.report_tools import EngineeringReportBuilder


class ToolRegistrationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InvestigationPlan:
    calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        calls = tuple(self.calls)
        identifiers = [call.call_id for call in calls]
        if not calls:
            raise ValueError("an investigation plan requires at least one tool call")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("investigation tool-call ids must be unique")
        object.__setattr__(self, "calls", calls)


class DeterministicOrchestrator:
    """Execute an explicit plan without an LLM or hidden retry behavior.

    Tool exceptions are isolated into failed ``ToolResult`` records. A denied
    or approval-gated call is recorded but never invoked. This is deliberately
    conservative: automatic planning can be layered above this boundary later.
    """

    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def register(self, tool: AgentTool) -> None:
        descriptor = tool.descriptor
        if descriptor.name in self._tools:
            raise ToolRegistrationError(f"duplicate agent tool: {descriptor.name}")
        if not callable(getattr(tool, "invoke", None)):
            raise ToolRegistrationError(f"agent tool {descriptor.name} has no invoke method")
        self._tools[descriptor.name] = tool

    def execute(
        self,
        context: InvestigationContext,
        plan: InvestigationPlan,
        *,
        approved_call_ids: Iterable[str] = (),
    ) -> InvestigationRecord:
        approved = frozenset(str(value) for value in approved_call_ids)
        unknown_approvals = approved - {call.call_id for call in plan.calls}
        if unknown_approvals:
            raise ValueError(f"approval references unknown calls: {sorted(unknown_approvals)}")
        record = InvestigationRecord(context)
        for call in plan.calls:
            record = record.record_call(call)
            tool = self._tools.get(call.tool)
            if tool is None:
                result = ToolResult(call.call_id, False, error=f"unknown tool: {call.tool}")
                record = record.record_result(result)
                return record.with_status(InvestigationStatus.FAILED)
            try:
                custom_authorize = getattr(tool, "authorize", None)
                decision = (
                    custom_authorize(
                        call, context.permission, approved=call.call_id in approved,
                    )
                    if callable(custom_authorize)
                    else authorize_tool(
                        tool.descriptor, context.permission,
                        approved=call.call_id in approved,
                    )
                )
                if not hasattr(decision, "allowed") or not hasattr(decision, "code"):
                    raise TypeError("tool authorization hooks must return PolicyDecision")
            except Exception as error:
                result = ToolResult(
                    call.call_id, False,
                    value={"exception_type": type(error).__name__, "stage": "authorization"},
                    error=str(error) or type(error).__name__,
                )
                return record.record_result(result).with_status(InvestigationStatus.FAILED)
            if not decision.allowed:
                result = ToolResult(
                    call.call_id, False,
                    value={"policy_code": decision.code.value},
                    error=decision.reason,
                )
                record = record.record_result(result)
                status = (
                    InvestigationStatus.WAITING_APPROVAL
                    if decision.code is DecisionCode.REQUIRE_APPROVAL
                    else InvestigationStatus.FAILED
                )
                return record.with_status(status)
            try:
                result = tool.invoke(call)
                if not isinstance(result, ToolResult):
                    raise TypeError("agent tools must return ToolResult")
                if result.call_id != call.call_id:
                    raise ValueError("tool result call_id does not match request")
            except Exception as error:
                result = ToolResult(
                    call.call_id, False,
                    value={"exception_type": type(error).__name__},
                    error=str(error) or type(error).__name__,
                )
            record = record.record_result(result)
            if not result.ok:
                return record.with_status(InvestigationStatus.FAILED)
        return record.with_status(InvestigationStatus.COMPLETED)


@dataclass(frozen=True, slots=True)
class InvestigationOutcome:
    """Complete, serializable result of one evidence-controlled investigation."""

    record: InvestigationRecord
    findings: tuple[EngineeringFinding, ...] = ()
    report: EngineeringReport | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        findings = tuple(self.findings)
        identifiers = [item.finding_id for item in findings]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("investigation outcome finding ids must be unique")
        object.__setattr__(self, "findings", findings)
        reason = None if self.failure_reason is None else str(self.failure_reason).strip()
        if self.record.status in {
            InvestigationStatus.FAILED, InvestigationStatus.UNQUALIFIED,
        } and not reason:
            raise ValueError("failed or unqualified outcomes require a reason")
        if self.report is not None:
            if self.report.investigation_id != self.record.context.investigation_id:
                raise ValueError("report and investigation identifiers do not match")
            if tuple(self.report.findings) != findings:
                raise ValueError("report and outcome findings do not match")
        object.__setattr__(self, "failure_reason", reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "findings": [item.to_dict() for item in self.findings],
            "report": None if self.report is None else self.report.to_dict(),
            "failure_reason": self.failure_reason,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationOutcome":
        data = require_fields(
            value, required={"record", "findings", "report", "failure_reason"},
            name="InvestigationOutcome",
        )
        data["record"] = InvestigationRecord.from_dict(data["record"])
        data["findings"] = tuple(
            EngineeringFinding.from_dict(item) for item in data["findings"]
        )
        if data["report"] is not None:
            data["report"] = EngineeringReport.from_dict(data["report"])
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "InvestigationOutcome":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("InvestigationOutcome JSON root must be an object")
        return cls.from_dict(parsed)


class SingleCaseInvestigator:
    """Run tools once, interpret their outputs, and build an auditable report.

    Tools listed in ``uninterpreted_tools`` may perform bounded catalog or
    rendering work without producing findings. Every other successful tool
    call must have exactly one specialist. This makes missing analysis visible
    instead of silently producing an incomplete engineering report.
    """

    def __init__(
        self,
        orchestrator: DeterministicOrchestrator,
        specialists: Iterable[DeterministicSpecialist] = (),
        *,
        evidence_policy: EvidencePolicy = EvidencePolicy(),
        report_builder: EngineeringReportBuilder | None = None,
        uninterpreted_tools: Iterable[str] = (),
    ) -> None:
        self.orchestrator = orchestrator
        self.evidence_policy = evidence_policy
        self.report_builder = report_builder or EngineeringReportBuilder()
        self.uninterpreted_tools = frozenset(str(item) for item in uninterpreted_tools)
        self._specialists: dict[str, DeterministicSpecialist] = {}
        for specialist in specialists:
            tool_name = str(specialist.tool_name).strip()
            if not tool_name:
                raise ValueError("specialists must declare a tool_name")
            if tool_name in self._specialists:
                raise ValueError(f"duplicate specialist for tool: {tool_name}")
            self._specialists[tool_name] = specialist

    @staticmethod
    def _results(record: InvestigationRecord) -> dict[str, ToolResult]:
        return {
            str(entry.payload["call_id"]): ToolResult.from_dict(entry.payload)
            for entry in record.entries if entry.kind is EntryKind.TOOL_RESULT
        }

    @staticmethod
    def _fail(
        record: InvestigationRecord,
        findings: tuple[EngineeringFinding, ...],
        reason: str,
        *,
        status: InvestigationStatus = InvestigationStatus.FAILED,
    ) -> InvestigationOutcome:
        record = record.append(EntryKind.NOTE, {"message": reason})
        return InvestigationOutcome(record.with_status(status), findings, None, reason)

    def investigate(
        self,
        context: InvestigationContext,
        plan: InvestigationPlan,
        *,
        title: str,
        approved_call_ids: Iterable[str] = (),
        qualification_passed: bool = False,
        assumptions: Iterable[str] = (),
        limitations: Iterable[str] = (),
        build_report: bool = True,
    ) -> InvestigationOutcome:
        if qualification_passed and not build_report:
            raise ValueError("qualification requires report generation")
        record = self.orchestrator.execute(
            context, plan, approved_call_ids=approved_call_ids,
        )
        if record.status is not InvestigationStatus.COMPLETED:
            reason = next(
                (str(entry.payload.get("error")) for entry in reversed(record.entries)
                 if entry.kind is EntryKind.TOOL_RESULT and entry.payload.get("error")),
                f"tool execution ended with status {record.status.value}",
            )
            return InvestigationOutcome(record, (), None, reason)

        results = self._results(record)
        findings: list[EngineeringFinding] = []
        finding_ids: set[str] = set()
        for call in plan.calls:
            result = results.get(call.call_id)
            if result is None:
                return self._fail(record, tuple(findings),
                                  f"missing recorded result for call {call.call_id}")
            specialist = self._specialists.get(call.tool)
            if specialist is None:
                if call.tool in self.uninterpreted_tools:
                    continue
                return self._fail(
                    record, tuple(findings),
                    f"no specialist registered for analytical tool {call.tool}",
                )
            try:
                produced = tuple(specialist.analyze(result))
            except Exception as error:
                return self._fail(
                    record, tuple(findings),
                    f"specialist {specialist.name} failed: "
                    f"{type(error).__name__}: {error}",
                )
            for finding in produced:
                if finding.finding_id in finding_ids:
                    return self._fail(
                        record, tuple(findings),
                        f"duplicate finding id produced: {finding.finding_id}",
                    )
                finding_ids.add(finding.finding_id)
                findings.append(finding)
                record = record.record_finding(finding)
                decision = qualify_finding(finding, self.evidence_policy)
                record = record.record_policy_decision(finding.finding_id, decision)
                if not decision.allowed:
                    return self._fail(
                        record, tuple(findings), decision.reason,
                        status=InvestigationStatus.UNQUALIFIED,
                    )

        findings_tuple = tuple(findings)
        if not build_report:
            return InvestigationOutcome(
                record.with_status(InvestigationStatus.COMPLETED), findings_tuple,
            )
        if qualification_passed:
            publication = authorize_publication(
                findings_tuple, qualified_run=True, policy=self.evidence_policy,
            )
            record = record.record_policy_decision("__report__", publication)
            if not publication.allowed:
                return self._fail(
                    record, findings_tuple, publication.reason,
                    status=InvestigationStatus.UNQUALIFIED,
                )
        report = self.report_builder.build(
            context, findings_tuple, title=title,
            qualified=qualification_passed,
            assumptions=assumptions, limitations=limitations,
        )
        record = record.record_report(report)
        return InvestigationOutcome(
            record.with_status(InvestigationStatus.COMPLETED),
            findings_tuple, report,
        )


__all__ = [
    "DeterministicOrchestrator", "InvestigationOutcome", "InvestigationPlan",
    "SingleCaseInvestigator", "ToolRegistrationError",
]
