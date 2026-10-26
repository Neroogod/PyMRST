"""Deterministic specialist tools for immutable application model snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


@dataclass(frozen=True, slots=True)
class ApplicationCaseSnapshot:
    case_id: str
    checksum: str
    model: Mapping[str, Any]
    grid: Mapping[str, Any] | None = None

    def __post_init__(self):
        if not self.case_id or len(self.checksum) != 64:
            raise ValueError("application case snapshot identity is incomplete")
        object.__setattr__(self, "model", dict(self.model))
        object.__setattr__(self, "grid", None if self.grid is None else dict(self.grid))


class ApplicationCaseCatalog:
    def __init__(self):
        self._items: dict[str, ApplicationCaseSnapshot] = {}

    def register(self, item: ApplicationCaseSnapshot):
        existing = self._items.get(item.case_id)
        if existing is not None and existing != item:
            raise ValueError("application case evidence changed within one investigation")
        self._items[item.case_id] = item
        return item

    def get(self, case_id: str):
        try:
            return self._items[str(case_id)]
        except KeyError as error:
            raise KeyError(f"unknown application case snapshot: {case_id}") from error


def _finding(severity, category, statement, affected=(), recommendation=None):
    value = {
        "severity": severity, "category": category, "statement": statement,
        "affected_objects": list(affected),
    }
    if recommendation is not None:
        value["recommendation"] = recommendation
    return value


def _recommend(identifier, statement, rationale):
    return {
        "recommendation_id": identifier, "statement": statement,
        "rationale": rationale, "requires_confirmation": True,
        "status": "proposed",
    }


class _ApplicationCaseTool:
    analysis_kind = ""
    description = ""

    def __init__(self, catalog: ApplicationCaseCatalog):
        self.catalog = catalog

    @property
    def descriptor(self):
        return ToolDescriptor(self.analysis_kind, self.description, AgentPermission.ANALYZE)

    def findings(self, snapshot: ApplicationCaseSnapshot):
        raise NotImplementedError

    def invoke(self, call: ToolCall) -> ToolResult:
        snapshot = self.catalog.get(str(call.arguments["case_id"]))
        findings = list(self.findings(snapshot))
        return ToolResult(call.call_id, True, {
            "analysis_kind": self.analysis_kind, "case_id": snapshot.case_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "findings": findings,
            "evidence": {
                "kind": "validation", "uri": f"case://{snapshot.case_id}/{self.analysis_kind}",
                "label": f"{self.description} for {snapshot.case_id}",
                "locator": {
                    "model_id": snapshot.model.get("model_id"),
                    "grid_id": None if snapshot.grid is None else snapshot.grid.get("snapshot_id"),
                },
                "checksum": snapshot.checksum,
            },
        })


class AnalyzeDeckInspectionTool(_ApplicationCaseTool):
    analysis_kind = "analyze_deck_review"
    description = "Deck structure and keyword inspection"

    def findings(self, snapshot):
        model = snapshot.model
        output = []
        for issue in model.get("issues", ()):
            location = issue.get("location") or {}
            affected = () if not location else (
                f"deck:{location.get('path', 'unknown')}:{location.get('line', 0)}",
            )
            recommendation = None
            if issue.get("severity") in {"error", "warning"}:
                recommendation = _recommend(
                    f"recommendation_deck_{issue.get('code', 'issue')}",
                    "Create a reviewable input patch for this deck finding.",
                    "The source location and immutable model inspection identify the affected record.",
                )
            output.append(_finding(
                issue.get("severity", "warning"),
                f"deck.{issue.get('category', 'inspection')}",
                str(issue.get("message", "Deck inspection finding")), affected,
                recommendation,
            ))
        sections = set(model.get("sections", ()))
        required = {"RUNSPEC", "GRID", "PROPS", "SOLUTION", "SCHEDULE"}
        missing = sorted(required - sections)
        if missing:
            output.append(_finding(
                "warning", "deck.sections",
                f"The inspected deck index does not contain sections: {', '.join(missing)}.",
                (f"model:{model.get('model_id', snapshot.case_id)}",),
            ))
        if not output:
            output.append(_finding(
                "info", "deck.inspection",
                f"Deck inspection completed with {len(model.get('occurrences', ()))} indexed keyword occurrence(s).",
                (f"model:{model.get('model_id', snapshot.case_id)}",),
            ))
        return output


class AnalyzeGridSnapshotTool(_ApplicationCaseTool):
    analysis_kind = "analyze_grid_quality"
    description = "Grid topology and property-summary inspection"

    def findings(self, snapshot):
        grid = snapshot.grid
        if grid is None:
            return [_finding("warning", "grid.missing", "No immutable grid snapshot is registered for this project.")]
        output = []
        for item in grid.get("findings", ()):
            output.append(_finding(
                item.get("severity", "warning"), f"grid.{item.get('code', 'quality')}",
                str(item.get("message", "Grid quality finding")),
            ))
        arrays = {item["name"]: item for item in grid.get("arrays", ())}
        volumes = arrays.get("CELL_VOLUME")
        if volumes is None:
            output.append(_finding("warning", "grid.volume", "Cell-volume data is unavailable in the grid snapshot."))
        else:
            if volumes.get("nonfinite_count", 0):
                output.append(_finding("error", "grid.volume", "Cell volumes contain non-finite values."))
            if volumes.get("minimum") is not None and volumes["minimum"] <= 0:
                output.append(_finding("error", "grid.volume", "The grid contains non-positive cell volumes."))
        if not output:
            output.append(_finding(
                "info", "grid.summary",
                f"Grid snapshot contains {grid.get('cell_count', 0)} cells, "
                f"{grid.get('face_count', 0)} faces, and {grid.get('node_count', 0)} nodes.",
                (f"grid:{grid.get('snapshot_id')}",),
            ))
        return output


class AnalyzeRockFluidSnapshotTool(_ApplicationCaseTool):
    analysis_kind = "analyze_rock_fluid"
    description = "Rock, region, and fluid-property inspection"

    def findings(self, snapshot):
        grid, output = snapshot.grid, []
        arrays = {} if grid is None else {item["name"]: item for item in grid.get("arrays", ())}
        porosity = next((arrays[name] for name in ("PORO", "POROSITY") if name in arrays), None)
        if porosity is not None:
            if porosity.get("nonfinite_count", 0) or porosity.get("minimum", 0) < 0 or porosity.get("maximum", 1) > 1:
                output.append(_finding("error", "rock.porosity", "Porosity is non-finite or lies outside [0, 1]."))
        permeability = [item for name, item in arrays.items() if name.startswith("PERM")]
        if permeability and any(item.get("minimum") is not None and item["minimum"] < 0 for item in permeability):
            output.append(_finding("error", "rock.permeability", "A permeability component contains negative values."))
        for issue in snapshot.model.get("issues", ()):
            category = str(issue.get("category", "")).lower()
            if any(token in category for token in ("rock", "fluid", "pvt", "region", "saturation")):
                output.append(_finding(issue.get("severity", "warning"), f"rock_fluid.{category}", str(issue.get("message"))))
        if not output:
            output.append(_finding("info", "rock_fluid.summary", f"Rock/fluid summary inspected {len(arrays)} registered cell array(s)."))
        return output


class AnalyzeInitializationSnapshotTool(_ApplicationCaseTool):
    analysis_kind = "analyze_initialization"
    description = "Initial pressure and saturation-summary inspection"

    def findings(self, snapshot):
        grid, output = snapshot.grid, []
        arrays = {} if grid is None else {item["name"]: item for item in grid.get("arrays", ())}
        pressure = next((arrays[name] for name in ("PRESSURE", "INITIAL_PRESSURE") if name in arrays), None)
        if pressure is not None and (pressure.get("nonfinite_count", 0) or pressure.get("minimum", 1) <= 0):
            output.append(_finding("error", "initialization.pressure", "Initial pressure is non-finite or non-positive."))
        saturation_names = ("SW", "SWAT", "SO", "SOIL", "SG", "SGAS")
        saturations = [arrays[name] for name in saturation_names if name in arrays]
        for item in saturations:
            if item.get("nonfinite_count", 0) or item.get("minimum", 0) < 0 or item.get("maximum", 1) > 1:
                output.append(_finding("error", "initialization.saturation", f"{item['name']} lies outside [0, 1] or is non-finite."))
        if not pressure and not saturations:
            output.append(_finding("warning", "initialization.missing", "No initial pressure or saturation arrays are registered in the grid snapshot."))
        elif not output:
            output.append(_finding("info", "initialization.summary", "Registered initialization summaries satisfy bounded pressure and saturation checks."))
        return output


class AnalyzeWellsScheduleSnapshotTool(_ApplicationCaseTool):
    analysis_kind = "analyze_wells_schedule"
    description = "Well, completion, control, and schedule inspection"

    def findings(self, snapshot):
        output = []
        occurrences = snapshot.model.get("occurrences", ())
        keywords = {item.get("keyword") for item in occurrences}
        well_keywords = keywords & {"WELSPECS", "COMPDAT", "WCONPROD", "WCONINJE", "WCONINJH"}
        for issue in snapshot.model.get("issues", ()):
            category = str(issue.get("category", "")).lower()
            if any(token in category for token in ("well", "schedule", "control", "completion")):
                output.append(_finding(issue.get("severity", "warning"), f"wells_schedule.{category}", str(issue.get("message"))))
        if "SCHEDULE" in set(snapshot.model.get("sections", ())) and not well_keywords:
            output.append(_finding("warning", "wells_schedule.empty", "The schedule section has no indexed well/control keywords."))
        if not output:
            output.append(_finding("info", "wells_schedule.summary", f"Well/schedule inspection found {len(well_keywords)} relevant keyword type(s)."))
        return output


__all__ = [
    "AnalyzeDeckInspectionTool", "AnalyzeGridSnapshotTool",
    "AnalyzeInitializationSnapshotTool", "AnalyzeRockFluidSnapshotTool",
    "AnalyzeWellsScheduleSnapshotTool", "ApplicationCaseCatalog",
    "ApplicationCaseSnapshot",
]
