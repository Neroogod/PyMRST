"""Read-only case inspection and immutable lineage tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
import warnings

import numpy as np

from pymrst.research_studio.case.definition import ReservoirCase, read_reservoir_case

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


class CaseAccessError(ValueError):
    pass


def _within(path: Path, roots: tuple[Path, ...]) -> Path:
    resolved = path.resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise CaseAccessError(f"case path lies outside configured roots: {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class RegisteredCase:
    case_id: str
    case: ReservoirCase
    source_path: str = ""


@dataclass(frozen=True, slots=True)
class WellSnapshot:
    name: str
    cells: tuple[int, ...]
    control_type: str
    target: float
    status: bool
    completion_status: tuple[bool, ...]
    sign: int


@dataclass(frozen=True, slots=True)
class ScheduleSnapshot:
    step_val: tuple[float, ...]
    step_control: tuple[int, ...]
    control_wells: tuple[tuple[WellSnapshot, ...], ...]


@dataclass(frozen=True, slots=True)
class AnalysisCase:
    case_id: str
    content_hash: str
    analysis_hash: str
    grid: Any
    rock: Any
    fluid_tables: Mapping[str, tuple[np.ndarray, ...]]
    regions: Mapping[str, np.ndarray]
    initial_state: Mapping[str, np.ndarray]
    wells: tuple[WellSnapshot, ...]
    schedule: ScheduleSnapshot | None


def _readonly_array(value, *, dtype=float) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _well_snapshot(well: Any, index: int) -> WellSnapshot:
    get = (lambda name, default=None: well.get(name, default)) \
        if isinstance(well, Mapping) else (lambda name, default=None: getattr(well, name, default))
    cells = tuple(int(item) for item in np.asarray(get("cells", ()), dtype=np.int64).ravel())
    completion = get("cstatus")
    if completion is None:
        completion_status = (True,) * len(cells)
    else:
        completion_status = tuple(bool(item) for item in np.asarray(completion).ravel())
    if len(completion_status) != len(cells):
        raise ValueError("well completion status must align with completion cells")
    target = float(get("val", 0.0))
    if not np.isfinite(target):
        raise ValueError("well target must be finite")
    return WellSnapshot(
        str(get("name") or f"well_{index}"), cells, str(get("type", "unknown")),
        target, bool(get("status", True)), completion_status, int(get("sign", 0)),
    )


def _schedule_snapshot(schedule: Any) -> ScheduleSnapshot | None:
    if schedule is None:
        return None
    if hasattr(schedule, "step_val"):
        values = np.asarray(schedule.step_val, dtype=float).ravel()
        controls = np.asarray(schedule.step_control, dtype=np.int64).ravel()
        owners = tuple(getattr(item, "W", None) or () for item in schedule.controls)
    elif isinstance(schedule, Mapping) and "step" in schedule:
        step = schedule["step"]
        values = np.asarray(step["val"], dtype=float).ravel()
        # Deck schedules are 1-based while runtime Schedule is 0-based.
        raw = np.asarray(step["control"], dtype=np.int64).ravel()
        controls = raw - 1 if raw.size and raw.min() >= 1 else raw
        owners = tuple(item.get("W", ()) if isinstance(item, Mapping) else ()
                       for item in schedule.get("control", ()))
    else:
        raise TypeError("schedule must be a runtime Schedule or deck schedule mapping")
    wells = tuple(
        tuple(_well_snapshot(well, index) for index, well in enumerate(group))
        for group in owners
    )
    return ScheduleSnapshot(
        tuple(float(item) for item in values), tuple(int(item) for item in controls), wells,
    )


def _analysis_digest(
    case_id: str,
    grid: Any,
    rock: Any,
    fluid_tables: Mapping[str, tuple[np.ndarray, ...]],
    regions: Mapping[str, np.ndarray],
    initial_state: Mapping[str, np.ndarray],
    wells: tuple[WellSnapshot, ...],
    schedule: ScheduleSnapshot | None,
) -> str:
    digest = hashlib.sha256()

    def text(label: str, value: Any) -> None:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")

    def array(label: str, value: Any) -> None:
        text("array", label)
        if value is None:
            digest.update(b"none\0")
            return
        item = np.ascontiguousarray(np.asarray(value))
        text("dtype", item.dtype.str)
        digest.update(np.asarray(item.shape, dtype=np.int64).tobytes())
        digest.update(item.tobytes())

    text("case_id", case_id)
    text("cells_num", int(grid.cells.num))
    text("faces_num", int(grid.faces.num))
    array("grid.faces.neighbors", getattr(grid.faces, "neighbors", None))
    array("grid.cells.volumes", getattr(grid.cells, "volumes", None))
    array("grid.cells.centroids", getattr(grid.cells, "centroids", None))
    array("rock.poro", getattr(rock, "poro", None) if rock is not None else None)
    array("rock.perm", getattr(rock, "perm", None) if rock is not None else None)
    for name, tables in sorted(fluid_tables.items()):
        text("fluid_table", name)
        for index, table in enumerate(tables):
            array(f"{name}[{index}]", table)
    for name, value in sorted(regions.items()):
        array(f"region.{name}", value)
    for name, value in sorted(initial_state.items()):
        array(f"state.{name}", value)
    text("wells", json.dumps([
        {
            "name": well.name, "cells": well.cells, "type": well.control_type,
            "target": well.target, "status": well.status,
            "completion_status": well.completion_status, "sign": well.sign,
        }
        for well in wells
    ], sort_keys=True))
    if schedule is None:
        text("schedule", "none")
    else:
        array("schedule.step_val", schedule.step_val)
        array("schedule.step_control", schedule.step_control)
        text("schedule.control_wells", json.dumps([
            [well.name for well in control] for control in schedule.control_wells
        ]))
    return digest.hexdigest()


class CaseAnalysisCatalog:
    """Explicit in-memory boundary for deterministic read-only case checks."""

    def __init__(self) -> None:
        self._cases: dict[str, AnalysisCase] = {}

    def register(
        self,
        case_id: str,
        *,
        grid: Any,
        rock: Any = None,
        fluid_tables: Mapping[str, Any] | None = None,
        regions: Mapping[str, Any] | None = None,
        initial_state: Mapping[str, Any] | None = None,
        wells: Iterable[Any] = (),
        schedule: Any = None,
        content_hash: str = "",
    ) -> AnalysisCase:
        case_id = str(case_id).strip()
        if not case_id or case_id in self._cases:
            raise KeyError(f"analysis case id is empty or already registered: {case_id}")
        if not hasattr(grid, "cells") or not hasattr(grid, "faces"):
            raise TypeError("analysis grid must expose cells and faces")
        count = int(grid.cells.num)
        if count <= 0:
            raise ValueError("analysis grid must contain cells")
        table_copy = {}
        for name, value in (fluid_tables or {}).items():
            entries = value if isinstance(value, (list, tuple)) else (value,)
            table_copy[str(name).upper()] = tuple(_readonly_array(item) for item in entries)
        region_copy = {
            str(name).upper(): _readonly_array(value, dtype=np.int64).reshape(-1)
            for name, value in (regions or {}).items()
        }
        state_copy = {
            str(name): _readonly_array(value)
            for name, value in (initial_state or {}).items()
        }
        for owner, values in (("region", region_copy), ("state", state_copy)):
            bad = [
                name for name, value in values.items()
                if (value.ndim == 0 or value.shape[0] != count)
                or (owner == "region" and value.size != count)
            ]
            if bad:
                raise ValueError(f"{owner} arrays must align with {count} cells: {bad}")
        well_copy = tuple(_well_snapshot(well, index) for index, well in enumerate(wells))
        schedule_copy = _schedule_snapshot(schedule)
        analysis_hash = _analysis_digest(
            case_id, grid, rock, table_copy, region_copy, state_copy, well_copy,
            schedule_copy,
        )
        digest = str(content_hash).lower() if content_hash else analysis_hash
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("analysis content_hash must be a lowercase SHA-256 digest")
        value = AnalysisCase(
            case_id, digest, analysis_hash, grid, rock, table_copy, region_copy, state_copy,
            well_copy, schedule_copy,
        )
        self._cases[case_id] = value
        return value

    def get(self, case_id: str) -> AnalysisCase:
        try:
            value = self._cases[case_id]
        except KeyError as error:
            raise KeyError(f"unknown analysis case: {case_id}") from error
        current = _analysis_digest(
            value.case_id, value.grid, value.rock, value.fluid_tables,
            value.regions, value.initial_state, value.wells, value.schedule,
        )
        if current != value.analysis_hash:
            raise CaseAccessError(
                f"registered analysis case changed after registration: {case_id}"
            )
        return value


class CaseCatalog:
    """Catalog definitions only; inspection never calls ``case.construct``."""

    def __init__(self, allowed_roots: Iterable[str | Path] = ()) -> None:
        if isinstance(allowed_roots, (str, Path)):
            allowed_roots = (allowed_roots,)
        self.roots = tuple(Path(item).resolve() for item in allowed_roots)
        self._cases: dict[str, RegisteredCase] = {}

    def register(self, case_id: str, case: ReservoirCase, *, source_path=""):
        case_id = str(case_id).strip()
        if not case_id or case_id in self._cases:
            raise KeyError(f"case id is empty or already registered: {case_id}")
        if not isinstance(case, ReservoirCase):
            raise TypeError("case catalog accepts ReservoirCase definitions")
        source = ""
        if source_path:
            if not self.roots:
                raise CaseAccessError("source paths require configured roots")
            source = str(_within(Path(source_path), self.roots))
        item = RegisteredCase(case_id, case, source)
        self._cases[case_id] = item
        return item

    def register_file(self, case_id: str, path: str | Path):
        if not self.roots:
            raise CaseAccessError("case-file registration requires configured roots")
        source = _within(Path(path), self.roots)
        if not source.is_file():
            raise FileNotFoundError(source)
        return self.register(case_id, read_reservoir_case(source), source_path=source)

    def get(self, case_id: str) -> RegisteredCase:
        try:
            return self._cases[case_id]
        except KeyError as error:
            raise KeyError(f"unknown case: {case_id}") from error

    def summary(self, case_id: str, *, verify_artifacts=False):
        item = self.get(case_id)
        case = item.case
        artifacts = []
        for artifact in case.artifacts:
            status = "not_verified"
            error = ""
            if verify_artifacts:
                try:
                    artifact.verify()
                    status = "verified"
                except (FileNotFoundError, ValueError) as exc:
                    status, error = "invalid", str(exc)
            artifacts.append({
                "role": artifact.role, "path": artifact.path,
                "sha256": artifact.sha256, "status": status, "error": error,
            })
        sections = {}
        for name in (
            "grid", "properties", "physics", "discretization", "facilities",
            "schedule", "solver",
        ):
            value = getattr(case, name)
            sections[name] = {
                "keys": sorted(value), "entry_count": len(value),
                "configured": bool(value),
            }
        return {
            "case_id": item.case_id, "name": case.name,
            "content_hash": case.content_hash, "builder": case.builder.owner,
            "source_path": item.source_path, "sections": sections,
            "artifacts": artifacts, "metadata": dict(case.metadata),
            "evidence": {
                "kind": "input", "uri": f"case://{item.case_id}",
                "label": f"Case definition {case.name}",
                "locator": {"source_path": item.source_path},
                "checksum": case.content_hash,
            },
        }


class InspectCaseTool:
    descriptor = ToolDescriptor(
        "inspect_case", "Inspect a registered case definition without constructing it",
        AgentPermission.INSPECT,
    )

    def __init__(self, catalog: CaseCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        value = self.catalog.summary(
            str(call.arguments["case_id"]),
            verify_artifacts=bool(call.arguments.get("verify_artifacts", False)),
        )
        return ToolResult(call.call_id, True, value)


def _evidence(case: AnalysisCase, subject: str):
    return {
        "kind": "validation", "uri": f"case://{case.case_id}/{subject}",
        "label": f"{subject.replace('_', ' ').title()} for {case.case_id}",
        "locator": {
            "subject": subject, "case_content_hash": case.content_hash,
        },
        "checksum": case.analysis_hash,
    }


def _check(severity, category, statement, *, cells=(), wells=(), details=None):
    affected = [f"cell:{int(item)}" for item in cells]
    affected += [f"well:{item}" for item in wells]
    return {
        "severity": severity, "category": category, "statement": statement,
        "affected_objects": affected, "details": details or {},
    }


class GridQualityTool:
    descriptor = ToolDescriptor(
        "analyze_grid_quality", "Analyze registered grid, rock, and well geometry",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: CaseAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        from pymrst.research_studio.inspection import validate

        case = self.catalog.get(str(call.arguments["case_id"]))
        wells = tuple(
            SimpleNamespace(name=w.name, cells=np.asarray(w.cells, dtype=np.int64))
            for w in case.wells
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            report = validate(case.grid, case.rock, wells)
        findings = []
        for item in report.findings:
            findings.append(_check(
                item.severity.value, f"grid.{item.check}", item.message,
                cells=item.cells,
                details={"faces": list(item.faces)},
            ))
        for item in captured:
            findings.append(_check(
                "warning", "grid.numerical_warning", str(item.message),
                details={"warning_type": item.category.__name__},
            ))
        value = {
            "analysis_kind": self.descriptor.name,
            "case_id": case.case_id, "ok": report.ok,
            "cell_count": int(case.grid.cells.num),
            "face_count": int(case.grid.faces.num),
            "findings": findings, "evidence": _evidence(case, "grid_quality"),
        }
        return ToolResult(call.call_id, True, value)


class RockFluidQualityTool:
    descriptor = ToolDescriptor(
        "analyze_rock_fluid", "Analyze rock arrays, regions, and tabular fluid data",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: CaseAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        case = self.catalog.get(str(call.arguments["case_id"]))
        findings = []
        for name, values in sorted(case.regions.items()):
            nonpositive = np.flatnonzero(values <= 0)
            if nonpositive.size:
                findings.append(_check(
                    "error", "regions", f"{name} has non-positive region identifiers",
                    cells=nonpositive,
                ))
            unique = np.unique(values[values > 0])
            if unique.size and not np.array_equal(unique, np.arange(1, unique.max() + 1)):
                findings.append(_check(
                    "warning", "regions", f"{name} region identifiers contain gaps",
                    details={"identifiers": unique.tolist()},
                ))
        table_summary = {}
        for name, tables in sorted(case.fluid_tables.items()):
            table_summary[name] = []
            for region, table in enumerate(tables, start=1):
                array = np.asarray(table, dtype=float)
                table_summary[name].append({"region": region, "shape": list(array.shape)})
                if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
                    findings.append(_check(
                        "error", "fluid_table",
                        f"{name} region {region} must be a two-dimensional table "
                        "with at least two rows and columns",
                    ))
                    continue
                if np.any(~np.isfinite(array)):
                    findings.append(_check(
                        "error", "fluid_table", f"{name} region {region} contains non-finite values",
                    ))
                saturation_table = name in {
                    "SWOF", "SWFN", "SGOF", "SGFN", "SOF2", "SOF3",
                }
                delta = np.diff(array[:, 0])
                monotone = np.all(delta > 0) if saturation_table else (
                    np.all(delta > 0) or np.all(delta < 0)
                )
                if not monotone:
                    findings.append(_check(
                        "error", "fluid_table",
                        f"{name} region {region} first column is not strictly monotone",
                    ))
                if saturation_table:
                    saturation = array[:, 0]
                    if np.any((saturation < 0) | (saturation > 1)):
                        findings.append(_check(
                            "error", "saturation_function",
                            f"{name} region {region} saturation lies outside [0, 1]",
                        ))
                    relperm = array[:, 1:-1] if array.shape[1] > 2 else array[:, 1:]
                    if np.any((relperm < 0) | (relperm > 1 + 1e-12)):
                        findings.append(_check(
                            "error", "saturation_function",
                            f"{name} region {region} relative permeability lies outside [0, 1]",
                        ))
        value = {
            "analysis_kind": self.descriptor.name,
            "case_id": case.case_id, "ok": not any(
                item["severity"] in {"error", "critical"} for item in findings
            ),
            "region_counts": {
                name: int(np.unique(values[values > 0]).size)
                for name, values in sorted(case.regions.items())
            },
            "tables": table_summary, "findings": findings,
            "evidence": _evidence(case, "rock_fluid"),
        }
        return ToolResult(call.call_id, True, value)


class InitializationQualityTool:
    descriptor = ToolDescriptor(
        "analyze_initialization", "Analyze pressure and phase saturation initial state",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: CaseAnalysisCatalog, *, saturation_tolerance=1e-8):
        self.catalog = catalog
        self.saturation_tolerance = float(saturation_tolerance)
        if self.saturation_tolerance <= 0:
            raise ValueError("saturation tolerance must be positive")

    def invoke(self, call: ToolCall) -> ToolResult:
        case = self.catalog.get(str(call.arguments["case_id"]))
        state, findings = case.initial_state, []
        pressure = state.get("pressure")
        if pressure is None:
            findings.append(_check("error", "initialization", "initial pressure is missing"))
        else:
            pressure = np.asarray(pressure, dtype=float).reshape(-1)
            bad = np.flatnonzero(~np.isfinite(pressure) | (pressure <= 0))
            if bad.size:
                findings.append(_check(
                    "error", "initialization", "initial pressure is non-finite or non-positive",
                    cells=bad,
                ))
        if "s" in state:
            saturation = np.asarray(state["s"], dtype=float)
            if saturation.ndim != 2:
                raise ValueError("initial-state s must be a cell-by-phase matrix")
            saturations = [saturation[:, column] for column in range(saturation.shape[1])]
        else:
            saturations = [
                np.asarray(state[name], dtype=float).reshape(-1)
                for name in ("sW", "sO", "sG") if name in state
            ]
        if not saturations:
            findings.append(_check("warning", "initialization", "no phase saturations were registered"))
        else:
            matrix = np.column_stack(saturations)
            outside = np.unique(np.argwhere(
                ~np.isfinite(matrix) | (matrix < -self.saturation_tolerance)
                | (matrix > 1 + self.saturation_tolerance)
            )[:, 0]) if matrix.size else np.array([], dtype=int)
            if outside.size:
                findings.append(_check(
                    "error", "initialization", "initial saturation lies outside [0, 1]",
                    cells=outside,
                ))
            closure = np.abs(np.sum(matrix, axis=1) - 1.0)
            bad = np.flatnonzero(closure > self.saturation_tolerance)
            if bad.size:
                findings.append(_check(
                    "error", "initialization",
                    f"phase saturations do not close within {self.saturation_tolerance:g}",
                    cells=bad, details={"maximum_closure_error": float(closure.max())},
                ))
        value = {
            "analysis_kind": self.descriptor.name, "case_id": case.case_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "state_fields": sorted(state), "findings": findings,
            "evidence": _evidence(case, "initialization"),
        }
        return ToolResult(call.call_id, True, value)


class WellsScheduleQualityTool:
    descriptor = ToolDescriptor(
        "analyze_wells_schedule", "Analyze wells, completions, controls, and timesteps",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: CaseAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        case = self.catalog.get(str(call.arguments["case_id"]))
        findings, names = [], [well.name for well in case.wells]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            findings.append(_check(
                "error", "wells", f"duplicate well names: {duplicates}", wells=duplicates,
            ))
        for well in case.wells:
            outside = [cell for cell in well.cells if not 0 <= cell < case.grid.cells.num]
            if outside:
                findings.append(_check(
                    "error", "wells", f"{well.name} has completions outside the grid",
                    cells=outside, wells=(well.name,),
                ))
            if not well.cells:
                findings.append(_check(
                    "warning", "wells", f"{well.name} has no completions", wells=(well.name,),
                ))
            if well.status and well.cells and not any(well.completion_status):
                findings.append(_check(
                    "warning", "wells",
                    f"{well.name} is open but all completions are closed", wells=(well.name,),
                ))
            if well.sign not in (-1, 0, 1):
                findings.append(_check(
                    "error", "wells", f"{well.name} has invalid sign {well.sign}",
                    wells=(well.name,),
                ))
        schedule = case.schedule
        schedule_summary = None
        if schedule is None:
            findings.append(_check("warning", "schedule", "no schedule was registered"))
        else:
            dt = np.asarray(schedule.step_val)
            control = np.asarray(schedule.step_control)
            if dt.size == 0:
                findings.append(_check("error", "schedule", "schedule has no timesteps"))
            bad_dt = np.flatnonzero(~np.isfinite(dt) | (dt <= 0))
            if bad_dt.size:
                findings.append(_check(
                    "error", "schedule", "schedule has non-finite or non-positive timesteps",
                    details={"report_steps": bad_dt.tolist()},
                ))
            invalid = np.flatnonzero(
                (control < 0) | (control >= len(schedule.control_wells))
            ) if control.size else np.array([], dtype=int)
            if invalid.size:
                findings.append(_check(
                    "error", "schedule", "report steps reference nonexistent controls",
                    details={"report_steps": invalid.tolist()},
                ))
            schedule_summary = {
                "step_count": int(dt.size), "control_count": len(schedule.control_wells),
                "total_time": float(dt.sum()) if np.all(np.isfinite(dt)) else None,
                "minimum_timestep": float(dt.min()) if dt.size else None,
                "maximum_timestep": float(dt.max()) if dt.size else None,
            }
        value = {
            "analysis_kind": self.descriptor.name, "case_id": case.case_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "well_count": len(case.wells), "schedule": schedule_summary,
            "findings": findings, "evidence": _evidence(case, "wells_schedule"),
        }
        return ToolResult(call.call_id, True, value)


__all__ = [
    "AnalysisCase", "CaseAccessError", "CaseAnalysisCatalog", "CaseCatalog",
    "GridQualityTool", "InitializationQualityTool", "InspectCaseTool",
    "RegisteredCase", "RockFluidQualityTool", "ScheduleSnapshot",
    "WellSnapshot", "WellsScheduleQualityTool",
]
