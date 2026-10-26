"""Policies governing local, ensemble, remote, and HPC execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..base import ToolDescriptor
from ..permissions import AgentPermission, PermissionGrant
from ..schemas.run_request import RunRequest


class DecisionCode(StrEnum):
    ALLOW = "allow"
    DENY_PERMISSION = "deny_permission"
    DENY_TOOL = "deny_tool"
    REQUIRE_APPROVAL = "require_approval"
    DENY_RESOURCE_LIMIT = "deny_resource_limit"
    DENY_MUTATION = "deny_mutation"
    DENY_EVIDENCE = "deny_evidence"
    DENY_QUALIFICATION = "deny_qualification"
    DENY_RETRY_POLICY = "deny_retry_policy"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    code: DecisionCode
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", DecisionCode(self.code))
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("policy decisions require a reason")
        object.__setattr__(self, "reason", reason)
        if self.allowed != (self.code is DecisionCode.ALLOW):
            raise ValueError("allowed policy decisions must use the ALLOW code")

    def to_dict(self):
        return {
            "allowed": self.allowed, "code": self.code.value, "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value):
        fields = {"allowed", "code", "reason"}
        if set(value) != fields:
            raise ValueError(f"invalid PolicyDecision fields: {sorted(set(value) ^ fields)}")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    cells: int
    report_steps: int
    realizations: int = 1
    workers: int = 1
    runtime_seconds: float = 0.0
    memory_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("cells", "report_steps", "realizations", "workers", "memory_bytes"):
            value = int(getattr(self, name))
            if value < 0 or (name in {"realizations", "workers"} and value == 0):
                raise ValueError(f"invalid resource estimate {name}")
            object.__setattr__(self, name, value)
        runtime = float(self.runtime_seconds)
        if runtime < 0:
            raise ValueError("runtime estimate cannot be negative")
        object.__setattr__(self, "runtime_seconds", runtime)

    def to_dict(self):
        return {
            "cells": self.cells, "report_steps": self.report_steps,
            "realizations": self.realizations, "workers": self.workers,
            "runtime_seconds": self.runtime_seconds,
            "memory_bytes": self.memory_bytes,
        }

    @classmethod
    def from_dict(cls, value):
        fields = {
            "cells", "report_steps", "realizations", "workers",
            "runtime_seconds", "memory_bytes",
        }
        if set(value) != fields:
            raise ValueError("invalid ResourceEstimate fields")
        return cls(**value)


def authorize_tool(
    descriptor: ToolDescriptor,
    grant: PermissionGrant,
    *,
    approved: bool = False,
) -> PolicyDecision:
    if descriptor.permission > grant.maximum:
        return PolicyDecision(
            False, DecisionCode.DENY_PERMISSION,
            f"{descriptor.name} requires {descriptor.permission.name.lower()} permission",
        )
    if not grant.permits(descriptor.permission, descriptor.name):
        return PolicyDecision(
            False, DecisionCode.DENY_TOOL,
            f"{descriptor.name} is not in the grant's tool allowlist",
        )
    if (descriptor.starts_workload or descriptor.requires_approval) and not approved:
        return PolicyDecision(
            False, DecisionCode.REQUIRE_APPROVAL,
            f"{descriptor.name} requires explicit approval",
        )
    return PolicyDecision(True, DecisionCode.ALLOW, "tool invocation is authorized")


def authorize_run(
    request: RunRequest,
    grant: PermissionGrant,
    estimate: ResourceEstimate,
    *,
    approved: bool,
    descriptor: ToolDescriptor | None = None,
) -> PolicyDecision:
    descriptor = descriptor or ToolDescriptor(
        "run_case", "Execute one bounded PyMRST case",
        AgentPermission.RUN_SIMULATION, starts_workload=True,
    )
    if not descriptor.starts_workload:
        raise ValueError("run authorization requires a workload descriptor")
    decision = authorize_tool(descriptor, grant, approved=approved)
    if not decision.allowed:
        return decision
    return authorize_resource_estimate(
        descriptor, grant, request.budget, estimate, approved=True,
        skip_tool_authorization=True,
    )


def authorize_resource_estimate(
    descriptor: ToolDescriptor,
    grant: PermissionGrant,
    requested,
    estimate: ResourceEstimate,
    *,
    approved: bool,
    skip_tool_authorization: bool = False,
) -> PolicyDecision:
    if not skip_tool_authorization:
        decision = authorize_tool(descriptor, grant, approved=approved)
        if not decision.allowed:
            return decision
    limits = grant.budget
    pairs = (
        ("cells", estimate.cells, limits.max_cells, requested.max_cells),
        ("report steps", estimate.report_steps, limits.max_report_steps,
         requested.max_report_steps),
        ("realizations", estimate.realizations, limits.max_realizations,
         requested.max_realizations),
        ("workers", estimate.workers, limits.max_workers, requested.max_workers),
        ("runtime seconds", estimate.runtime_seconds, limits.max_runtime_seconds,
         requested.max_runtime_seconds),
        ("memory bytes", estimate.memory_bytes, limits.max_memory_bytes,
         requested.max_memory_bytes),
    )
    for label, actual, grant_limit, request_limit in pairs:
        if actual > min(grant_limit, request_limit):
            return PolicyDecision(
                False, DecisionCode.DENY_RESOURCE_LIMIT,
                f"estimated {label} {actual} exceeds effective limit "
                f"{min(grant_limit, request_limit)}",
            )
    return PolicyDecision(True, DecisionCode.ALLOW, "run is approved and within budget")


__all__ = [
    "DecisionCode", "PolicyDecision", "ResourceEstimate", "authorize_run",
    "authorize_resource_estimate", "authorize_tool",
]
