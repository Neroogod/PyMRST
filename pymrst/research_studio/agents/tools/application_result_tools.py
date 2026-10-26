"""Deterministic KPI and conservation tools for application result summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


@dataclass(frozen=True, slots=True)
class ApplicationResultSnapshot:
    run_id: str
    checksum: str
    states: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]

    def __post_init__(self):
        if not self.run_id or len(self.checksum) != 64:
            raise ValueError("application result snapshot identity is incomplete")
        object.__setattr__(self, "states", tuple(dict(item) for item in self.states))
        object.__setattr__(self, "events", tuple(dict(item) for item in self.events))


class ApplicationResultCatalog:
    def __init__(self): self._items = {}
    def register(self, item):
        existing = self._items.get(item.run_id)
        if existing is not None and existing != item:
            raise ValueError("result evidence changed within one investigation")
        self._items[item.run_id] = item
        return item
    def get(self, run_id):
        try: return self._items[str(run_id)]
        except KeyError as error: raise KeyError(f"unknown application result: {run_id}") from error


def _evidence(item, subject):
    return {
        "kind": "state", "uri": f"run://{item.run_id}/results/{subject}",
        "label": f"{subject.replace('_', ' ').title()} for {item.run_id}",
        "locator": {"state_count": len(item.states)}, "checksum": item.checksum,
    }


class AnalyzeApplicationPerformanceTool:
    descriptor = ToolDescriptor(
        "analyze_reservoir_performance", "Analyze saved application state summaries",
        AgentPermission.ANALYZE,
    )
    def __init__(self, catalog): self.catalog = catalog
    def invoke(self, call):
        item = self.catalog.get(call.arguments["run_id"])
        findings, fields = [], {}
        if not item.states:
            findings.append({
                "severity": "warning", "category": "performance.missing_states",
                "statement": "No indexed saved states are available for performance analysis.",
                "affected_objects": [f"run:{item.run_id}"],
            })
        else:
            first = {value["name"]: value for value in item.states[0]["fields"]}
            last = {value["name"]: value for value in item.states[-1]["fields"]}
            for name in sorted(set(first) & set(last)):
                left, right = first[name], last[name]
                if left.get("mean") is not None and right.get("mean") is not None:
                    fields[name] = {
                        "initial_mean": left["mean"], "final_mean": right["mean"],
                        "final_minimum": right.get("minimum"),
                        "final_maximum": right.get("maximum"),
                        "mean_change": right["mean"] - left["mean"],
                    }
                if left.get("nonfinite_count", 0) or right.get("nonfinite_count", 0):
                    findings.append({
                        "severity": "critical", "category": "performance.nonfinite",
                        "statement": f"Saved field {name} contains non-finite values.",
                        "affected_objects": [f"field:{name}"],
                    })
        return ToolResult(call.call_id, True, {
            "analysis_kind": self.descriptor.name, "result_id": item.run_id,
            "report_steps": len(item.states), "fields": fields,
            "wells": {}, "production": None, "regions": {},
            "findings": findings, "evidence": _evidence(item, "reservoir_performance"),
        })


class AnalyzeApplicationConservationTool:
    descriptor = ToolDescriptor(
        "analyze_conservation", "Inspect recorded conservation and material-balance evidence",
        AgentPermission.ANALYZE,
    )
    def __init__(self, catalog): self.catalog = catalog
    def invoke(self, call):
        item = self.catalog.get(call.arguments["run_id"])
        values = []
        for event in item.events:
            if event["event_type"] != "convergence": continue
            residuals = event["payload"].get("residuals", {})
            for name, value in residuals.items():
                key = str(name).lower()
                if "mb" in key or "material" in key or "conservation" in key:
                    try: values.append((str(name), abs(float(value))))
                    except (TypeError, ValueError): pass
        if values:
            name, maximum = max(values, key=lambda pair: pair[1])
            findings = [{
                "severity": "info", "category": "conservation.recorded_residual",
                "statement": f"The maximum recorded material-balance residual is "
                             f"{maximum:.6g} ({name}).",
                "affected_objects": [f"run:{item.run_id}"],
            }]
        else:
            findings = [{
                "severity": "warning", "category": "conservation.missing_evidence",
                "statement": "The retained run events contain no explicitly named material-balance residual.",
                "affected_objects": [f"run:{item.run_id}"],
            }]
        return ToolResult(call.call_id, True, {
            "analysis_kind": self.descriptor.name, "result_id": item.run_id,
            "findings": findings, "evidence": _evidence(item, "conservation"),
        })


__all__ = [
    "AnalyzeApplicationConservationTool", "AnalyzeApplicationPerformanceTool",
    "ApplicationResultCatalog", "ApplicationResultSnapshot",
]
