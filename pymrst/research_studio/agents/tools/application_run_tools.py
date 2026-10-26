"""Provider-independent diagnostics for immutable application run-event snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


@dataclass(frozen=True, slots=True)
class RunEventSnapshot:
    run_id: str
    status: str
    events: tuple[Mapping[str, Any], ...]
    checksum: str

    def __post_init__(self):
        if not self.run_id or not self.status or len(self.checksum) != 64:
            raise ValueError("run-event snapshot identity is incomplete")
        object.__setattr__(self, "events", tuple(dict(item) for item in self.events))


class RunEventCatalog:
    def __init__(self):
        self._items: dict[str, RunEventSnapshot] = {}

    def register(self, snapshot: RunEventSnapshot) -> RunEventSnapshot:
        existing = self._items.get(snapshot.run_id)
        if existing is not None and existing != snapshot:
            raise ValueError("run evidence changed within one investigation")
        self._items[snapshot.run_id] = snapshot
        return snapshot

    def get(self, run_id: str) -> RunEventSnapshot:
        try:
            return self._items[str(run_id)]
        except KeyError as error:
            raise KeyError(f"unknown run-event snapshot: {run_id}") from error


class AnalyzeApplicationRunTool:
    descriptor = ToolDescriptor(
        "analyze_numerical_diagnostics",
        "Analyze a checksum-bound application run-event snapshot",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: RunEventCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        snapshot = self.catalog.get(str(call.arguments["run_id"]))
        events = snapshot.events
        convergence = [item for item in events if item["event_type"] == "convergence"]
        iteration_events = [item for item in convergence
                            if item["payload"].get("newton_iterations") is not None]
        if not iteration_events:
            iteration_events = [item for item in events if item["event_type"] == "progress"
                                and item["payload"].get("newton_iterations") is not None]
        iterations = [int(item["payload"]["newton_iterations"])
                      for item in iteration_events]
        cuts = [int(item["payload"].get("timestep_cuts", 0)) for item in convergence]
        findings = []
        status = snapshot.status
        if status in {"failed", "interrupted", "cancelled"}:
            failure = next((item["payload"] for item in reversed(events)
                            if item["event_type"] in {"failure", "orchestration_failure"}), {})
            findings.append({
                "severity": "error" if status == "failed" else "warning",
                "category": "numerical.run_status",
                "statement": f"Run {snapshot.run_id} ended with status {status}: "
                             f"{failure.get('message', 'No structured failure message was recorded.')}",
                "affected_objects": [f"run:{snapshot.run_id}"],
                "recommendation": {
                    "recommendation_id": f"recommendation_diagnose_{snapshot.run_id}",
                    "statement": "Review the final convergence event and checkpoint before proposing a retry.",
                    "rationale": "The terminal status is supported by retained worker evidence.",
                    "requires_confirmation": True, "status": "proposed",
                },
            })
        elif status == "completed":
            findings.append({
                "severity": "info", "category": "numerical.run_status",
                "statement": f"Run {snapshot.run_id} completed with "
                             f"{len(convergence)} recorded convergence event(s).",
                "affected_objects": [f"run:{snapshot.run_id}"],
            })
        else:
            findings.append({
                "severity": "warning", "category": "numerical.run_status",
                "statement": f"Run {snapshot.run_id} is {status}; diagnostics are provisional.",
                "affected_objects": [f"run:{snapshot.run_id}"],
            })
        iteration_summary = None
        if iterations:
            maximum, total = max(iterations), sum(iterations)
            severity = "error" if maximum >= 15 else "warning" if maximum >= 8 else "info"
            item = {
                "severity": severity, "category": "numerical.nonlinear_iterations",
                "statement": f"The recorded nonlinear history contains {total} Newton "
                             f"iteration(s); the maximum at one report step is {maximum}.",
                "affected_objects": [f"run:{snapshot.run_id}"],
            }
            if severity != "info":
                item["recommendation"] = {
                    "recommendation_id": f"recommendation_timestep_{snapshot.run_id}",
                    "statement": "Run a bounded timestep-sensitivity branch around the difficult report step.",
                    "rationale": "A localized experiment separates physical difficulty from solver configuration.",
                    "requires_confirmation": True, "status": "proposed",
                }
            findings.append(item)
            iteration_summary = {
                "total": float(total), "maximum": float(maximum),
                "mean": float(total / len(iterations)),
            }
        if cuts and sum(cuts):
            findings.append({
                "severity": "warning", "category": "numerical.timestep_cuts",
                "statement": f"The solver recorded {sum(cuts)} timestep cut(s) across "
                             f"{sum(value > 0 for value in cuts)} report step(s).",
                "affected_objects": [f"run:{snapshot.run_id}"],
                "recommendation": {
                    "recommendation_id": f"recommendation_cut_review_{snapshot.run_id}",
                    "statement": "Inspect control changes and residual components at each cut report step.",
                    "rationale": "Timestep cuts localize schedule or constitutive transitions.",
                    "requires_confirmation": True, "status": "proposed",
                },
            })
        return ToolResult(call.call_id, True, {
            "analysis_kind": "analyze_numerical_diagnostics",
            "report_steps": len(convergence), "iterations": iteration_summary,
            "findings": findings,
            "evidence": {
                "kind": "convergence", "uri": f"run://{snapshot.run_id}/events",
                "label": f"Immutable event history for {snapshot.run_id}",
                "locator": {"event_count": len(events), "run_status": status},
                "checksum": snapshot.checksum,
            },
        })


__all__ = ["AnalyzeApplicationRunTool", "RunEventCatalog", "RunEventSnapshot"]
