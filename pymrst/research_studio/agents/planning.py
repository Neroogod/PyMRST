"""Provider-independent, permission-checked investigation planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .base import ReasoningProvider, ToolCall, ToolDescriptor
from ._serialization import frozen_json_mapping
from .context import InvestigationContext
from .orchestrator import InvestigationPlan


@dataclass(frozen=True, slots=True)
class PlanningRecord:
    provider: str
    context_hash: str
    plan: InvestigationPlan
    content_hash: str
    provider_metadata: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(
            self, "provider_metadata",
            frozen_json_mapping(self.provider_metadata, "$.provider_metadata"),
        )

    def to_dict(self):
        return {
            "provider": self.provider, "context_hash": self.context_hash,
            "plan": {"calls": [item.to_dict() for item in self.plan.calls]},
            "content_hash": self.content_hash,
            "provider_metadata": dict(self.provider_metadata),
        }


class BoundedInvestigationPlanner:
    """Ask a provider for a plan, then independently enforce its authority."""

    def __init__(
        self, provider: ReasoningProvider, *, provider_name="provider",
        maximum_calls=32, provider_metadata: Mapping[str, Any] | None = None,
    ):
        self.provider, self.provider_name = provider, str(provider_name).strip()
        self.maximum_calls = int(maximum_calls)
        self.provider_metadata = dict(provider_metadata or {})
        if not self.provider_name or self.maximum_calls <= 0:
            raise ValueError("planner provider name and call limit must be valid")

    @staticmethod
    def _tool_schema(descriptor: ToolDescriptor):
        return {
            "name": descriptor.name, "description": descriptor.description,
            "permission": descriptor.permission.name.lower(),
            "mutates_case": descriptor.mutates_case,
            "starts_workload": descriptor.starts_workload,
            "requires_approval": descriptor.requires_approval,
        }

    def plan(self, context: InvestigationContext, descriptors: Sequence[ToolDescriptor]):
        descriptors = tuple(descriptors)
        by_name = {item.name: item for item in descriptors}
        if len(by_name) != len(descriptors):
            raise ValueError("planning tool descriptors must have unique names")
        context_value = context.to_dict()
        context_hash = hashlib.sha256(json.dumps(
            context_value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        messages = ({
            "role": "system",
            "content": "Propose only typed, bounded engineering tool calls. Do not invent results.",
        }, {
            "role": "user", "content": {
                "investigation_id": context.investigation_id,
                "question": context.question, "case_id": str(context.case_id),
                "case_content_hash": context.case_content_hash,
                "run_ids": [str(item) for item in context.run_ids],
                "selection": context.selection.to_dict(),
                "maximum_calls": self.maximum_calls,
            },
        })
        proposed = self.provider.complete(
            messages, tuple(self._tool_schema(item) for item in descriptors),
            InvestigationPlan,
        )
        if not isinstance(proposed, InvestigationPlan):
            raise TypeError("reasoning provider must return InvestigationPlan")
        if len(proposed.calls) > self.maximum_calls:
            raise ValueError("provider plan exceeds maximum tool-call count")
        run_ids = {str(item) for item in context.run_ids}
        for call in proposed.calls:
            descriptor = by_name.get(call.tool)
            if descriptor is None:
                raise ValueError(f"provider planned an unavailable tool: {call.tool}")
            if not context.permission.permits(descriptor.permission, descriptor.name):
                raise PermissionError(f"provider planned an unauthorized tool: {call.tool}")
            if descriptor.mutates_case or descriptor.starts_workload:
                raise PermissionError("analysis planning cannot include mutations or workloads")
            for key in ("run_id", "reference_run_id", "candidate_run_id"):
                if key in call.arguments and str(call.arguments[key]) not in run_ids:
                    raise ValueError(f"provider planned an out-of-context {key}")
            if "case_id" in call.arguments and str(call.arguments["case_id"]) != str(context.case_id):
                raise ValueError("provider planned an out-of-context case_id")
        encoded = json.dumps(
            {"provider": self.provider_name, "provider_metadata": self.provider_metadata,
             "context_hash": context_hash,
             "calls": [item.to_dict() for item in proposed.calls]},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return PlanningRecord(
            self.provider_name, context_hash, proposed,
            hashlib.sha256(encoded.encode()).hexdigest(), self.provider_metadata,
        )


class DeterministicPlanningProvider:
    """Offline planner used when no language-model provider is configured."""

    def complete(self, messages, tools, response_type):
        if response_type is not InvestigationPlan:
            raise TypeError("deterministic planner only produces InvestigationPlan")
        context = dict(messages[-1]["content"])
        available = {item["name"] for item in tools}
        case_id, run_ids = context["case_id"], tuple(context["run_ids"])
        calls = []
        case_tools = (
            "analyze_deck_review", "analyze_grid_quality", "analyze_rock_fluid",
            "analyze_initialization", "analyze_wells_schedule",
        )
        for name in case_tools:
            if name in available:
                calls.append(ToolCall(f"plan_{len(calls):03d}", name, {"case_id": case_id}))
        for run_id in run_ids:
            for name in (
                "analyze_numerical_diagnostics", "analyze_reservoir_performance",
                "analyze_conservation", "analyze_invariants",
            ):
                if name in available:
                    calls.append(ToolCall(
                        f"plan_{len(calls):03d}", name, {"run_id": run_id},
                    ))
        if len(run_ids) >= 2 and "compare_reference_runs" in available:
            for candidate in run_ids[1:]:
                calls.append(ToolCall(
                    f"plan_{len(calls):03d}", "compare_reference_runs",
                    {"reference_run_id": run_ids[0], "candidate_run_id": candidate},
                ))
        if not calls:
            raise ValueError("no compatible deterministic planning tools are available")
        return InvestigationPlan(tuple(calls))


__all__ = [
    "BoundedInvestigationPlanner", "DeterministicPlanningProvider", "PlanningRecord",
]
