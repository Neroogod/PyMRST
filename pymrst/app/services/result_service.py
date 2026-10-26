"""Lazy, checksum-bound result browsing, comparison, animation, and export."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from ..database.models import AppDatabase
from ..schemas._common import json_value, utc_now
from ..schemas.result import ResultExport, ResultField, ResultState, RunComparison
from .project_service import ProjectService, _file_hash
from .run_service import RunService


def _field(value: dict[str, Any]) -> ResultField:
    return ResultField(
        value["name"], tuple(value["shape"]), value["dtype"], value["minimum"],
        value["maximum"], value["mean"], value["nonfinite_count"],
    )


def _state(row) -> ResultState:
    return ResultState(
        row["state_id"], row["run_id"], row["report_step"],
        row["simulated_time_seconds"], row["relative_path"], row["content_hash"],
        tuple(_field(item) for item in json.loads(row["fields_json"])), row["created_at"],
    )


class ResultService:
    def __init__(
        self, database: AppDatabase, projects: ProjectService, runs: RunService, *,
        maximum_fields: int = 512, maximum_slice_values: int = 250_000,
        maximum_comparison_values: int = 250_000_000,
    ) -> None:
        self.database, self.projects, self.runs = database, projects, runs
        self.maximum_fields = int(maximum_fields)
        self.maximum_slice_values = int(maximum_slice_values)
        self.maximum_comparison_values = int(maximum_comparison_values)

    def _run_root(self, run_id: str) -> Path:
        run = self.runs.get(run_id); project = self.projects.get(run.project_id)
        root = (Path(project.root_path) / run.run_relative_path).resolve()
        if not root.is_dir():
            raise RuntimeError(f"run directory is missing: {run_id}")
        return root

    @staticmethod
    def _summarize(name: str, value: np.ndarray) -> ResultField:
        numeric = value.astype(float, copy=False).reshape(-1)
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            minimum, maximum, mean = float(finite.min()), float(finite.max()), float(finite.mean())
        else:
            minimum = maximum = mean = None
        return ResultField(
            name, tuple(value.shape), value.dtype.str, minimum, maximum, mean,
            int(numeric.size - finite.size),
        )

    @staticmethod
    def _transport_values(value: np.ndarray):
        if value.ndim == 0:
            item = value.item()
            return None if isinstance(item, (float, np.floating)) and not np.isfinite(item) else item
        return [ResultService._transport_values(item) for item in value]

    def index_run(self, run_id: str) -> tuple[ResultState, ...]:
        root = self._run_root(run_id)
        events = self.runs.events(run_id, limit=10_000)
        state_events = [item for item in events if item.event_type == "state_saved"]
        times = {
            int(item.payload["report_step"]): float(item.payload.get("simulated_time_seconds", 0))
            for item in events
            if item.event_type == "progress" and "report_step" in item.payload
        }
        discovered: list[tuple[int, str]] = []
        if state_events:
            discovered = [(int(item.payload["report_step"]), str(item.payload["relative_path"]))
                          for item in state_events]
        else:
            for path in sorted((root / "states").glob("report_*.npz")) if (root / "states").is_dir() else ():
                match = re.fullmatch(r"report_(\d+)\.npz", path.name)
                if match:
                    discovered.append((int(match.group(1)), path.relative_to(root).as_posix()))
        for report_step, relative in discovered:
            target = (root / relative).resolve()
            if root not in target.parents or target.suffix != ".npz" or not target.is_file() or target.is_symlink():
                raise ValueError("state artifact is unsafe or unsupported")
            digest = _file_hash(target)
            fields = []
            with np.load(target, allow_pickle=False, mmap_mode="r") as archive:
                if len(archive.files) > self.maximum_fields:
                    raise ValueError("state contains too many fields")
                for name in sorted(archive.files):
                    value = archive[name]
                    if value.dtype.kind not in "biufc":
                        raise TypeError(f"state field {name} is not numeric")
                    fields.append(self._summarize(name, value))
            state_id = "state_" + hashlib.sha256(f"{run_id}:{report_step}:{digest}".encode()).hexdigest()[:24]
            created = utc_now()
            with self.database.transaction() as connection:
                connection.execute(
                    """INSERT INTO app_result_states VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id,report_step) DO UPDATE SET
                    state_id=excluded.state_id, simulated_time_seconds=excluded.simulated_time_seconds,
                    relative_path=excluded.relative_path, content_hash=excluded.content_hash,
                    fields_json=excluded.fields_json""",
                    (state_id, run_id, report_step, times.get(report_step, 0.0), relative,
                     digest, json.dumps([item.to_dict() for item in fields]), created),
                )
        return self.states(run_id, auto_index=False)

    def states(self, run_id: str, *, auto_index=True) -> tuple[ResultState, ...]:
        self.runs.get(run_id)
        rows = self.database.fetch_all(
            "SELECT * FROM app_result_states WHERE run_id=? ORDER BY report_step", (run_id,),
        )
        if not rows and auto_index:
            return self.index_run(run_id)
        return tuple(_state(row) for row in rows)

    def state(self, state_id: str) -> ResultState:
        row = self.database.fetch_one("SELECT * FROM app_result_states WHERE state_id=?", (state_id,))
        if row is None:
            raise KeyError(f"unknown result state: {state_id}")
        return _state(row)

    def _state_path(self, state: ResultState) -> Path:
        root = self._run_root(state.run_id); path = (root / state.relative_path).resolve()
        if root not in path.parents or _file_hash(path) != state.content_hash:
            raise RuntimeError("result state failed immutable lineage verification")
        return path

    def field_values(
        self, state_id: str, field: str, *, offset: int = 0, count: int = 25_000,
        stride: int = 1,
    ) -> dict[str, Any]:
        state = self.state(state_id); field = str(field)
        metadata = next((item for item in state.fields if item.name == field), None)
        if metadata is None:
            raise KeyError(f"unknown state field: {field}")
        offset, count, stride = int(offset), int(count), int(stride)
        if offset < 0 or not 1 <= count <= self.maximum_slice_values or stride < 1:
            raise ValueError("result field slice exceeds configured bounds")
        with np.load(self._state_path(state), allow_pickle=False, mmap_mode="r") as archive:
            value = archive[field]
            if value.ndim == 0:
                selected, total = value.reshape(1), 1
            else:
                selected, total = value[
                    offset:min(value.shape[0], offset + count*stride):stride
                ], value.shape[0]
            return {
                "state_id": state_id, "field": field, "offset": offset,
                "count": int(selected.shape[0]), "total": int(total), "stride": stride,
                "shape": list(selected.shape),
                "values": self._transport_values(np.asarray(selected)),
            }

    def dashboard(self, run_id: str) -> dict[str, Any]:
        states = self.states(run_id)
        if not states:
            return {"run_id": run_id, "times": [], "field": {}, "wells": {}}
        field_names = {item.name for item in states[0].fields}
        if any({item.name for item in state.fields} != field_names for state in states):
            raise ValueError("state field schema changes across report steps")
        reservoir = [name for name in ("pressure", "sW", "sO", "sG", "rs") if name in field_names]
        wells = [name for name in ("bhp", "qWs", "qOs", "qGs") if name in field_names]
        field_result = {name: [] for name in reservoir}; well_result = {name: [] for name in wells}
        for state in states:
            with np.load(self._state_path(state), allow_pickle=False, mmap_mode="r") as archive:
                for name in reservoir:
                    value = np.asarray(archive[name], dtype=float)
                    finite = value[np.isfinite(value)]
                    field_result[name].append({
                        "minimum": float(finite.min()) if finite.size else None,
                        "mean": float(finite.mean()) if finite.size else None,
                        "maximum": float(finite.max()) if finite.size else None,
                    })
                for name in wells:
                    well_result[name].append(self._transport_values(
                        np.asarray(archive[name], dtype=float).reshape(-1)
                    ))
        well_count = max((len(row) for values in well_result.values() for row in values), default=0)
        initialized = next((item for item in self.runs.events(run_id, limit=10_000)
                            if item.event_type == "simulation_initialized"), None)
        declared_names = list(initialized.payload.get("well_names", ())) if initialized else []
        well_names = declared_names if len(declared_names) == well_count else [
            f"WELL_{index}" for index in range(well_count)
        ]
        return json_value({
            "run_id": run_id, "report_steps": [item.report_step for item in states],
            "times": [item.simulated_time_seconds for item in states],
            "field": field_result, "wells": well_result,
            "well_names": well_names,
        })

    def analysis_fields(
        self, run_id: str, *, fields=(), report_steps=(), maximum_values=5_000_000,
    ) -> dict[str, np.ndarray]:
        """Return checksum-verified full arrays under an explicit value ceiling."""
        states = self.states(run_id)
        by_step = {item.report_step: item for item in states}
        selected_steps = tuple(int(item) for item in report_steps) or (
            (() if not states else (states[0].report_step, states[-1].report_step))
        )
        selected_steps = tuple(dict.fromkeys(selected_steps))
        if any(step not in by_step for step in selected_steps):
            raise KeyError("analysis report step is not indexed")
        available = ({item.name for item in states[0].fields} if states else set())
        selected_fields = tuple(str(item) for item in fields) or tuple(sorted(available))
        if any(name not in available for name in selected_fields):
            raise KeyError("analysis field is not indexed")
        limit, count, output = int(maximum_values), 0, {}
        if limit <= 0:
            raise ValueError("analysis value limit must be positive")
        for step in selected_steps:
            state = by_step[step]
            with np.load(self._state_path(state), allow_pickle=False, mmap_mode="r") as archive:
                for name in selected_fields:
                    value = np.asarray(archive[name])
                    count += value.size
                    if count > limit:
                        raise ValueError(
                            f"full-array analysis contains {count} values; limit is {limit}"
                        )
                    copied = np.array(value, copy=True); copied.setflags(write=False)
                    output[f"{name}@{step}"] = copied
        return output

    def compare(self, reference_run_id: str, candidate_run_id: str) -> RunComparison:
        reference_run, candidate_run = self.runs.get(reference_run_id), self.runs.get(candidate_run_id)
        if reference_run.project_id != candidate_run.project_id:
            raise ValueError("run comparison requires runs in the same project")
        reference = {item.report_step: item for item in self.states(reference_run_id)}
        candidate = {item.report_step: item for item in self.states(candidate_run_id)}
        steps = sorted(set(reference) & set(candidate))
        if not steps:
            raise ValueError("runs have no common saved report steps")
        names = sorted(
            {item.name for item in reference[steps[0]].fields}
            & {item.name for item in candidate[steps[0]].fields}
        )
        metrics: dict[str, Any] = {}
        total = 0
        for name in names:
            maximum_absolute = maximum_relative = squared = 0.0; count = 0; worst_step = steps[0]
            for step in steps:
                with np.load(self._state_path(reference[step]), allow_pickle=False) as left, \
                     np.load(self._state_path(candidate[step]), allow_pickle=False) as right:
                    if name not in left.files or name not in right.files or left[name].shape != right[name].shape:
                        raise ValueError(f"field {name} is unavailable or shape-incompatible at step {step}")
                    a, b = np.asarray(left[name], dtype=float), np.asarray(right[name], dtype=float)
                    total += a.size
                    if total > self.maximum_comparison_values:
                        raise ValueError("comparison exceeds configured value limit")
                    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
                        raise ValueError(f"field {name} contains non-finite comparison values")
                    absolute = np.abs(b - a); local = float(absolute.max(initial=0))
                    relative = absolute / np.maximum(np.abs(a), 1e-30)
                    if local > maximum_absolute:
                        maximum_absolute, worst_step = local, step
                    maximum_relative = max(maximum_relative, float(relative.max(initial=0)))
                    squared += float(np.dot((b-a).reshape(-1), (b-a).reshape(-1))); count += a.size
            metrics[name] = {
                "maximum_absolute_error": maximum_absolute,
                "maximum_relative_error": maximum_relative,
                "root_mean_square_error": float(np.sqrt(squared / max(count, 1))),
                "worst_report_step": worst_step, "compared_values": count,
            }
        summary = {"common_report_steps": steps, "fields": metrics}
        canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(
            (reference_run_id + "\0" + candidate_run_id + "\0" + canonical).encode()
        ).hexdigest()
        comparison_id = "comparison_" + digest[:24]; created = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_result_comparisons VALUES(?,?,?,?,?,?,?)",
                (comparison_id, reference_run.project_id, reference_run_id,
                 candidate_run_id, digest, json.dumps(summary), created),
            )
        row = self.database.fetch_one(
            "SELECT * FROM app_result_comparisons WHERE comparison_id=?", (comparison_id,),
        )
        return RunComparison(
            row["comparison_id"], row["project_id"], row["reference_run_id"],
            row["candidate_run_id"], row["content_hash"],
            json.loads(row["summary_json"]), row["created_at"],
        )

    def difference(
        self, reference_run_id: str, candidate_run_id: str, report_step: int,
        field: str, *, offset=0, count=25_000,
    ) -> dict[str, Any]:
        left = next((item for item in self.states(reference_run_id) if item.report_step == int(report_step)), None)
        right = next((item for item in self.states(candidate_run_id) if item.report_step == int(report_step)), None)
        if left is None or right is None:
            raise KeyError("report step is not saved by both runs")
        offset, count = int(offset), int(count)
        if offset < 0 or not 1 <= count <= self.maximum_slice_values:
            raise ValueError("difference slice exceeds configured bounds")
        with np.load(self._state_path(left), allow_pickle=False, mmap_mode="r") as a, \
             np.load(self._state_path(right), allow_pickle=False, mmap_mode="r") as b:
            if field not in a.files or field not in b.files or a[field].shape != b[field].shape:
                raise ValueError("difference field is unavailable or shape-incompatible")
            stop = min(a[field].shape[0], offset + count)
            difference = np.asarray(b[field][offset:stop], dtype=float) - np.asarray(a[field][offset:stop], dtype=float)
            return json_value({
                "reference_run_id": reference_run_id, "candidate_run_id": candidate_run_id,
                "report_step": int(report_step), "field": field, "offset": offset,
                "count": int(stop-offset), "total": int(a[field].shape[0]),
                "shape": list(difference.shape), "values": difference.tolist(),
            })

    def animation_manifest(self, run_id: str, field: str) -> dict[str, Any]:
        frames = []
        for state in self.states(run_id):
            if not any(item.name == field for item in state.fields):
                raise KeyError(f"field {field} is not available in every saved state")
            frames.append({
                "state_id": state.state_id, "report_step": state.report_step,
                "simulated_time_seconds": state.simulated_time_seconds,
                "field_url": f"/api/v1/result-states/{state.state_id}/fields/{field}",
            })
        return {"schema_version": 1, "run_id": run_id, "field": field, "frames": frames}

    def export_field(self, state_id: str, field: str, *, kind="csv") -> ResultExport:
        if kind not in {"csv", "json"}:
            raise ValueError("supported result exports are csv and json")
        state = self.state(state_id); values = self.field_values(
            state_id, field, count=self.maximum_slice_values,
        )
        if values["total"] > self.maximum_slice_values:
            raise ValueError("field export exceeds configured single-file value limit")
        run = self.runs.get(state.run_id); root = self._run_root(state.run_id)
        export_id = "export_" + hashlib.sha256(
            f"{state_id}:{field}:{kind}:{state.content_hash}".encode()
        ).hexdigest()[:24]
        safe_field = re.sub(r"[^A-Za-z0-9_]+", "_", field)
        relative = Path("exports") / f"{export_id}_{safe_field}.{kind}"
        target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=".export-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            if kind == "json":
                with os.fdopen(handle, "w") as stream:
                    json.dump(values, stream, indent=2, sort_keys=True); stream.write("\n")
            else:
                with os.fdopen(handle, "w", newline="") as stream:
                    writer = csv.writer(stream); writer.writerow(["index", "component", field])
                    array = np.asarray(values["values"])
                    if array.ndim == 1:
                        for index, value in enumerate(array): writer.writerow([index, 0, value])
                    else:
                        for index, row in enumerate(array.reshape(array.shape[0], -1)):
                            for component, value in enumerate(row): writer.writerow([index, component, value])
            temporary.replace(target); os.chmod(target, 0o444)
        finally:
            temporary.unlink(missing_ok=True)
        digest = _file_hash(target); created = utc_now()
        metadata = {"state_id": state_id, "field": field, "shape": values["shape"]}
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO app_result_exports VALUES(?,?,?,?,?,?,?)",
                (export_id, run.run_id, kind, relative.as_posix(), digest,
                 json.dumps(metadata), created),
            )
        return ResultExport(export_id, run.run_id, kind, relative.as_posix(), digest, metadata, created)


__all__ = ["ResultService"]
