"""Approval-gated simulation execution, checkpoint lineage, and retry proposals."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
from typing import Any, Mapping, Protocol

from ._serialization import frozen_json_mapping, json_value, require_fields
from .base import ToolCall, ToolDescriptor, ToolResult
from .permissions import AgentPermission, PermissionGrant
from .policies.execution import (
    DecisionCode, PolicyDecision, ResourceEstimate, authorize_run, authorize_tool,
)
from .schemas.run_request import RunRequest
from .schemas.run_result import RunResult, RunStatus


class SimulationExecutionAdapter(Protocol):
    """Application-owned adapter; implementations may use local, remote, or HPC workers."""

    def submit(self, request: RunRequest) -> RunResult: ...
    def inspect(self, run_id: str) -> RunResult: ...
    def cancel(self, run_id: str) -> RunResult: ...


@dataclass(frozen=True, slots=True)
class RegisteredRunRequest:
    request: RunRequest
    estimate: ResourceEstimate


class RunRequestCatalog:
    """Append-only registry ensuring executed requests were reviewed in full."""

    def __init__(self) -> None:
        self._requests: dict[str, RegisteredRunRequest] = {}

    def register(
        self, request: RunRequest, estimate: ResourceEstimate,
    ) -> RegisteredRunRequest:
        if request.request_id in self._requests:
            raise KeyError(f"run request already registered: {request.request_id}")
        registered = RegisteredRunRequest(request, estimate)
        self._requests[request.request_id] = registered
        return registered

    def get(self, request_id: str) -> RegisteredRunRequest:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"unknown run request: {request_id}") from error


class RunResultCatalog:
    """Append-only status history with legal lifecycle transitions."""

    _terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    _allowed = {
        RunStatus.QUEUED: {
            RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED,
            RunStatus.FAILED, RunStatus.CANCELLED,
        },
        RunStatus.RUNNING: {
            RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED,
            RunStatus.CANCELLED,
        },
    }

    def __init__(self) -> None:
        self._history: dict[str, list[RunResult]] = {}

    def record(self, result: RunResult) -> RunResult:
        run_id = str(result.run_id)
        history = self._history.setdefault(run_id, [])
        if history:
            previous = history[-1]
            if previous.request_id != result.request_id:
                raise ValueError("run status update changed request lineage")
            if previous.status in self._terminal:
                if previous == result:
                    return result
                raise ValueError("terminal run status cannot transition")
            if result.status not in self._allowed[previous.status]:
                raise ValueError(
                    f"illegal run status transition {previous.status.value} -> "
                    f"{result.status.value}"
                )
        history.append(result)
        return result

    def latest(self, run_id: str) -> RunResult:
        try:
            return self._history[str(run_id)][-1]
        except (KeyError, IndexError) as error:
            raise KeyError(f"unknown run result: {run_id}") from error

    def history(self, run_id: str) -> tuple[RunResult, ...]:
        try:
            return tuple(self._history[str(run_id)])
        except KeyError as error:
            raise KeyError(f"unknown run result: {run_id}") from error


class CheckpointLineageCatalog:
    def __init__(self) -> None:
        self._items: dict[str, CheckpointLineage] = {}

    def register(self, lineage: "CheckpointLineage") -> "CheckpointLineage":
        if lineage.checkpoint_id in self._items:
            raise KeyError(f"checkpoint lineage already registered: {lineage.checkpoint_id}")
        self._items[lineage.checkpoint_id] = lineage
        return lineage

    def for_run(self, run_id: str) -> tuple["CheckpointLineage", ...]:
        return tuple(sorted(
            (item for item in self._items.values() if item.source_run_id == str(run_id)),
            key=lambda item: (item.completed_report_steps, item.checkpoint_id),
        ))


class SubmitSimulationTool:
    descriptor = ToolDescriptor(
        "submit_simulation", "Submit one registered and bounded simulation",
        AgentPermission.RUN_SIMULATION, starts_workload=True,
    )

    def __init__(
        self, catalog: RunRequestCatalog, adapter: SimulationExecutionAdapter,
        results: RunResultCatalog | None = None,
    ):
        self.catalog = catalog
        self.adapter = adapter
        self.results = results

    def authorize(
        self, call: ToolCall, grant: PermissionGrant, *, approved: bool,
    ) -> PolicyDecision:
        request_id = str(call.arguments.get("request_id", ""))
        try:
            registered = self.catalog.get(request_id)
        except KeyError as error:
            return PolicyDecision(False, DecisionCode.DENY_TOOL, str(error))
        return authorize_run(
            registered.request, grant, registered.estimate, approved=approved,
            descriptor=self.descriptor,
        )

    def invoke(self, call: ToolCall) -> ToolResult:
        registered = self.catalog.get(str(call.arguments["request_id"]))
        result = self.adapter.submit(registered.request)
        if not isinstance(result, RunResult):
            raise TypeError("simulation adapters must return RunResult")
        if result.request_id != registered.request.request_id:
            raise ValueError("adapter result request_id does not match submission")
        if str(result.run_id) != str(registered.request.run_id):
            raise ValueError("adapter result run_id does not match submission")
        if self.results is not None:
            self.results.record(result)
        return ToolResult(call.call_id, True, {"run_result": result.to_dict()})


class InspectSimulationTool:
    descriptor = ToolDescriptor(
        "inspect_simulation", "Inspect run status without changing it",
        AgentPermission.INSPECT,
    )

    def __init__(
        self, adapter: SimulationExecutionAdapter,
        results: RunResultCatalog | None = None,
    ):
        self.adapter = adapter
        self.results = results

    def invoke(self, call: ToolCall) -> ToolResult:
        result = self.adapter.inspect(str(call.arguments["run_id"]))
        if not isinstance(result, RunResult):
            raise TypeError("simulation adapters must return RunResult")
        if self.results is not None:
            self.results.record(result)
        return ToolResult(call.call_id, True, {"run_result": result.to_dict()})


class CancelSimulationTool:
    descriptor = ToolDescriptor(
        "cancel_simulation", "Request cancellation of a simulation worker",
        AgentPermission.RUN_SIMULATION, requires_approval=True,
    )

    def __init__(
        self, adapter: SimulationExecutionAdapter,
        results: RunResultCatalog | None = None,
    ):
        self.adapter = adapter
        self.results = results

    def invoke(self, call: ToolCall) -> ToolResult:
        run_id = str(call.arguments["run_id"])
        result = self.adapter.cancel(run_id)
        if not isinstance(result, RunResult) or str(result.run_id) != run_id:
            raise ValueError("simulation adapter returned an invalid cancellation result")
        if self.results is not None:
            self.results.record(result)
        return ToolResult(call.call_id, True, {"run_result": result.to_dict()})


@dataclass(frozen=True, slots=True)
class CheckpointLineage:
    checkpoint_id: str
    source_request_id: str
    source_run_id: str
    case_content_hash: str
    completed_report_steps: int
    schedule_fingerprint: str
    model_fingerprint: str
    solver_fingerprint: str
    code_fingerprint: str
    failure_bundle_uri: str | None = None

    def __post_init__(self) -> None:
        for name in ("checkpoint_id", "source_request_id", "source_run_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        for name in (
            "case_content_hash", "schedule_fingerprint", "model_fingerprint",
            "solver_fingerprint", "code_fingerprint",
        ):
            value = str(getattr(self, name)).lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a SHA-256 digest")
            object.__setattr__(self, name, value)
        steps = int(self.completed_report_steps)
        if steps < 0:
            raise ValueError("completed report steps cannot be negative")
        object.__setattr__(self, "completed_report_steps", steps)
        if self.failure_bundle_uri is not None and "://" not in self.failure_bundle_uri:
            raise ValueError("failure bundle must use an application URI")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "source_request_id": self.source_request_id,
            "source_run_id": self.source_run_id,
            "case_content_hash": self.case_content_hash,
            "completed_report_steps": self.completed_report_steps,
            "schedule_fingerprint": self.schedule_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "solver_fingerprint": self.solver_fingerprint,
            "code_fingerprint": self.code_fingerprint,
            "failure_bundle_uri": self.failure_bundle_uri,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointLineage":
        return cls(**require_fields(value, required={
            "checkpoint_id", "source_request_id", "source_run_id",
            "case_content_hash", "completed_report_steps", "schedule_fingerprint",
            "model_fingerprint", "solver_fingerprint", "code_fingerprint",
            "failure_bundle_uri",
        }, name="CheckpointLineage"))


class FailureClass(StrEnum):
    NONFINITE = "nonfinite"
    LINEAR_SOLVER = "linear_solver"
    NONLINEAR_CONVERGENCE = "nonlinear_convergence"
    RESOURCE_LIMIT = "resource_limit"
    INPUT_OR_CONFIGURATION = "input_or_configuration"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailureDiagnosis:
    failure_class: FailureClass
    summary: str
    retry_candidate: bool
    diagnostic_actions: tuple[str, ...]
    evidence_uri: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_class", FailureClass(self.failure_class))
        summary = str(self.summary).strip()
        actions = tuple(str(item).strip() for item in self.diagnostic_actions)
        if not summary or any(not item for item in actions):
            raise ValueError("failure diagnoses require a summary and valid actions")
        if not actions:
            raise ValueError("failure diagnoses require at least one diagnostic action")
        if self.evidence_uri is not None and "://" not in self.evidence_uri:
            raise ValueError("diagnosis evidence must use an application URI")
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "diagnostic_actions", actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class.value, "summary": self.summary,
            "retry_candidate": self.retry_candidate,
            "diagnostic_actions": list(self.diagnostic_actions),
            "evidence_uri": self.evidence_uri,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureDiagnosis":
        data = require_fields(value, required={
            "failure_class", "summary", "retry_candidate",
            "diagnostic_actions", "evidence_uri",
        }, name="FailureDiagnosis")
        data["diagnostic_actions"] = tuple(data["diagnostic_actions"])
        return cls(**data)


def diagnose_run_failure(result: RunResult) -> FailureDiagnosis:
    if result.status is RunStatus.CANCELLED:
        return FailureDiagnosis(
            FailureClass.CANCELLED, "The simulation was cancelled.", False,
            ("inspect_cancellation_request",),
        )
    if result.status is not RunStatus.FAILED or result.failure is None:
        raise ValueError("failure diagnosis requires a failed or cancelled run")
    text = f"{result.failure.error_type} {result.failure.message}".lower()
    uri = result.failure.failure_bundle_uri
    if any(token in text for token in ("nan", "inf", "nonfinite", "non-finite")):
        return FailureDiagnosis(
            FailureClass.NONFINITE, "A non-finite numerical value terminated the run.",
            False, ("replay_failure_bundle", "inspect_first_nonfinite_equation"), uri,
        )
    if any(token in text for token in ("linear solver", "singular", "gmres", "factorization")):
        return FailureDiagnosis(
            FailureClass.LINEAR_SOLVER, "The linear solution stage failed.",
            result.failure.retryable,
            ("inspect_jacobian_rank_and_scaling", "compare_linear_solver_backend"), uri,
        )
    if any(token in text for token in ("newton", "converg", "max iteration", "timestep")):
        return FailureDiagnosis(
            FailureClass.NONLINEAR_CONVERGENCE,
            "The nonlinear solve did not reach its convergence criteria.",
            result.failure.retryable,
            ("inspect_cnv_mb_history", "inspect_control_switches", "replay_last_checkpoint"),
            uri,
        )
    if any(token in text for token in ("memory", "timeout", "resource")):
        return FailureDiagnosis(
            FailureClass.RESOURCE_LIMIT, "The run exceeded an execution resource.",
            False, ("inspect_resource_metrics", "request_revised_execution_budget"), uri,
        )
    if any(token in text for token in ("keyword", "deck", "input", "configuration")):
        return FailureDiagnosis(
            FailureClass.INPUT_OR_CONFIGURATION,
            "The run failed while validating its input or configuration.",
            False, ("run_deck_and_configuration_validation",), uri,
        )
    return FailureDiagnosis(
        FailureClass.UNKNOWN, "The failure is not covered by a deterministic classifier.",
        False, ("inspect_failure_bundle", "request_engineering_review"), uri,
    )


class DiagnoseRunFailureTool:
    descriptor = ToolDescriptor(
        "diagnose_run_failure", "Classify one recorded failed simulation",
        AgentPermission.ANALYZE,
    )

    def __init__(
        self, results: RunResultCatalog,
        checkpoints: CheckpointLineageCatalog | None = None,
    ) -> None:
        self.results = results
        self.checkpoints = checkpoints

    def invoke(self, call: ToolCall) -> ToolResult:
        run_id = str(call.arguments["run_id"])
        result = self.results.latest(run_id)
        diagnosis = diagnose_run_failure(result)
        serialized = result.to_dict()
        checksum = hashlib.sha256(json.dumps(
            serialized, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        checkpoint_values = (
            () if self.checkpoints is None else self.checkpoints.for_run(run_id)
        )
        return ToolResult(call.call_id, True, {
            "analysis_kind": "diagnose_run_failure",
            "diagnosis": diagnosis.to_dict(),
            "run_result": serialized,
            "checkpoint_lineage": tuple(item.to_dict() for item in checkpoint_values),
            "evidence": {
                "kind": "run", "uri": f"run://{run_id}/failure",
                "label": f"Failure record for {run_id}",
                "locator": {"run_id": run_id, "status": result.status.value},
                "checksum": checksum,
            },
            "findings": ({
                "category": f"numerical.failure.{diagnosis.failure_class.value}",
                "severity": "error",
                "statement": diagnosis.summary,
                "affected_objects": (f"run:{run_id}",),
            },),
        })


@dataclass(frozen=True, slots=True)
class RetryLimits:
    max_attempts: int = 3
    max_iterations: int = 100
    max_timestep_cuts: int = 20
    allowed_solver_keys: tuple[str, ...] = (
        "max_iterations", "max_timestep_cuts", "initial_timestep_factor",
        "use_line_search",
    )

    def __post_init__(self) -> None:
        for name in ("max_attempts", "max_iterations", "max_timestep_cuts"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        keys = tuple(str(item).strip() for item in self.allowed_solver_keys)
        if any(not item for item in keys) or len(set(keys)) != len(keys):
            raise ValueError("retry solver keys must be unique and nonempty")
        object.__setattr__(self, "allowed_solver_keys", keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts, "max_iterations": self.max_iterations,
            "max_timestep_cuts": self.max_timestep_cuts,
            "allowed_solver_keys": list(self.allowed_solver_keys),
        }


@dataclass(frozen=True, slots=True)
class RetryProposal:
    proposal_id: str
    source_request_id: str
    source_run_id: str
    attempt: int
    rationale: str
    diagnosis: FailureDiagnosis
    solver_overrides: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("proposal_id", "source_request_id", "source_run_id", "rationale"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        attempt = int(self.attempt)
        if attempt <= 0:
            raise ValueError("retry attempt must be positive")
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(
            self, "solver_overrides",
            frozen_json_mapping(self.solver_overrides, "$.solver_overrides"),
        )
        if self.checkpoint_id is not None:
            value = str(self.checkpoint_id).strip()
            if not value:
                raise ValueError("checkpoint_id cannot be empty")
            object.__setattr__(self, "checkpoint_id", value)

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "proposal_id": self.proposal_id,
            "source_request_id": self.source_request_id,
            "source_run_id": self.source_run_id, "attempt": self.attempt,
            "rationale": self.rationale, "diagnosis": self.diagnosis.to_dict(),
            "solver_overrides": self.solver_overrides,
            "checkpoint_id": self.checkpoint_id,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetryProposal":
        data = require_fields(value, required={
            "proposal_id", "source_request_id", "source_run_id", "attempt",
            "rationale", "diagnosis", "solver_overrides", "checkpoint_id",
        }, name="RetryProposal")
        data["diagnosis"] = FailureDiagnosis.from_dict(data["diagnosis"])
        return cls(**data)


def authorize_retry(
    proposal: RetryProposal,
    grant: PermissionGrant,
    *,
    approved: bool,
    limits: RetryLimits = RetryLimits(),
) -> PolicyDecision:
    descriptor = ToolDescriptor(
        "materialize_retry", "Materialize a reviewed retry request",
        AgentPermission.RUN_SIMULATION, requires_approval=True,
    )
    decision = authorize_tool(descriptor, grant, approved=approved)
    if not decision.allowed:
        return decision
    if not proposal.diagnosis.retry_candidate:
        return PolicyDecision(
            False, DecisionCode.DENY_RETRY_POLICY,
            "the deterministic failure diagnosis does not permit automatic retry",
        )
    if proposal.attempt > limits.max_attempts:
        return PolicyDecision(
            False, DecisionCode.DENY_RETRY_POLICY,
            f"retry attempt {proposal.attempt} exceeds limit {limits.max_attempts}",
        )
    unknown = set(proposal.solver_overrides) - set(limits.allowed_solver_keys)
    if unknown:
        return PolicyDecision(
            False, DecisionCode.DENY_RETRY_POLICY,
            f"retry contains unsupported solver overrides: {sorted(unknown)}",
        )
    values = proposal.solver_overrides
    if "max_iterations" in values and not 1 <= int(values["max_iterations"]) <= limits.max_iterations:
        return PolicyDecision(False, DecisionCode.DENY_RETRY_POLICY,
                              "retry max_iterations is outside its safe bound")
    if "max_timestep_cuts" in values and not 0 <= int(values["max_timestep_cuts"]) <= limits.max_timestep_cuts:
        return PolicyDecision(False, DecisionCode.DENY_RETRY_POLICY,
                              "retry max_timestep_cuts is outside its safe bound")
    if "initial_timestep_factor" in values and not 0 < float(values["initial_timestep_factor"]) <= 1:
        return PolicyDecision(False, DecisionCode.DENY_RETRY_POLICY,
                              "initial_timestep_factor must lie in (0, 1]")
    if "use_line_search" in values and not isinstance(values["use_line_search"], bool):
        return PolicyDecision(False, DecisionCode.DENY_RETRY_POLICY,
                              "use_line_search retry override must be boolean")
    return PolicyDecision(True, DecisionCode.ALLOW, "retry proposal is approved and bounded")


def materialize_retry_request(
    source: RunRequest,
    proposal: RetryProposal,
    *,
    new_request_id: str,
    new_run_id: str,
    grant: PermissionGrant,
    approved: bool,
    limits: RetryLimits = RetryLimits(),
) -> RunRequest:
    if proposal.source_request_id != source.request_id or proposal.source_run_id != str(source.run_id):
        raise ValueError("retry proposal lineage does not match source request")
    decision = authorize_retry(proposal, grant, approved=approved, limits=limits)
    if not decision.allowed:
        raise PermissionError(decision.reason)
    solver = dict(source.solver)
    solver.update(dict(proposal.solver_overrides))
    return replace(
        source, request_id=new_request_id, run_id=new_run_id, solver=solver,
        checkpoint_id=proposal.checkpoint_id,
    )


def materialize_resume_request(
    source: RunRequest,
    lineage: CheckpointLineage,
    *,
    new_request_id: str,
    new_run_id: str,
) -> RunRequest:
    if (
        lineage.source_request_id != source.request_id
        or lineage.source_run_id != str(source.run_id)
        or lineage.case_content_hash != source.case_content_hash
    ):
        raise ValueError("checkpoint lineage does not match source run request")
    return replace(
        source, request_id=new_request_id, run_id=new_run_id,
        checkpoint_id=lineage.checkpoint_id,
    )


__all__ = [
    "CancelSimulationTool", "CheckpointLineage", "CheckpointLineageCatalog",
    "DiagnoseRunFailureTool", "FailureClass", "FailureDiagnosis",
    "InspectSimulationTool", "RegisteredRunRequest", "RetryLimits",
    "RetryProposal", "RunRequestCatalog", "RunResultCatalog",
    "SimulationExecutionAdapter", "SubmitSimulationTool", "authorize_retry",
    "diagnose_run_failure", "materialize_resume_request",
    "materialize_retry_request",
]
