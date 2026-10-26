"""Bounded, process-isolated installation workload qualification."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
from time import monotonic
from uuid import uuid4

from ..database.models import AppDatabase
from ..schemas._common import json_value, utc_now
from ..schemas.workload import (
    WorkloadBudget, WorkloadDefinition, WorkloadQualification, WorkloadRequest,
    WorkloadStatus, WorkloadTier,
)


def _json(value) -> str:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


_DEFAULT_BUDGETS = {
    WorkloadTier.SMOKE: WorkloadBudget(30, 30, 2 * 1024**3, 1 * 1024**3),
    WorkloadTier.ENGINEERING: WorkloadBudget(180, 180, 4 * 1024**3, 8 * 1024**3),
    WorkloadTier.FIELD_SCALE: WorkloadBudget(900, 900, 12 * 1024**3, 40 * 1024**3),
    WorkloadTier.ENDURANCE: WorkloadBudget(3600, 3600, 8 * 1024**3, 80 * 1024**3),
}


WORKLOAD_CATALOG = (
    WorkloadDefinition(
        "numerical-core", "Vectorized assembly and sparse solve",
        "Exercises deterministic NumPy face-flux/divergence assembly and a SciPy sparse linear solve without changing simulator state.",
        tuple(WorkloadTier), {key.value: value for key, value in _DEFAULT_BUDGETS.items()},
    ),
    WorkloadDefinition(
        "state-io", "State persistence throughput",
        "Exercises compressed numeric state writes and safe non-pickle round trips in a disposable qualification directory.",
        tuple(WorkloadTier), {key.value: value for key, value in _DEFAULT_BUDGETS.items()},
    ),
    WorkloadDefinition(
        "spe1-blackoil", "SPE1 black-oil simulation",
        "Runs the canonical 300-cell, three-phase dissolved-gas SPE1 model through the production nonlinear solver and certifies convergence, state, saturation, well and material-balance invariants.",
        (WorkloadTier.SMOKE, WorkloadTier.ENGINEERING),
        {
            WorkloadTier.SMOKE.value: WorkloadBudget(
                60, 60, 3 * 1024**3, 1 * 1024**3,
            ),
            WorkloadTier.ENGINEERING.value: WorkloadBudget(
                300, 300, 4 * 1024**3, 2 * 1024**3,
            ),
        },
        requirements=("BENCH_SPE1.DATA", "NumPy/SciPy ADI backend"),
    ),
    WorkloadDefinition(
        "mandel-poromechanics", "Mandel coupled poromechanics",
        "Runs the fully coupled VEM mechanics and single-phase flow model under constant plate load, then certifies force equilibrium and the analytical Mandel-Cryer pressure overshoot.",
        (WorkloadTier.SMOKE, WorkloadTier.ENGINEERING),
        {
            WorkloadTier.SMOKE.value: WorkloadBudget(
                60, 60, 3 * 1024**3, 1 * 1024**3,
            ),
            WorkloadTier.ENGINEERING.value: WorkloadBudget(
                300, 300, 6 * 1024**3, 2 * 1024**3,
            ),
        },
        requirements=("SciPy sparse solver", "VEM mechanics backend"),
    ),
)
_CATALOG = {item.workload_id: item for item in WORKLOAD_CATALOG}


def _from_row(row) -> WorkloadQualification:
    decode = lambda value, fallback: fallback if value is None else json.loads(value)
    return WorkloadQualification(
        row["qualification_id"], row["workload_id"], row["tier"], row["status"],
        row["progress"], row["message"], json.loads(row["request_json"]),
        json.loads(row["budgets_json"]), json.loads(row["metrics_json"]),
        json.loads(row["comparison_json"]), row["evidence_relative_path"],
        row["content_hash"], row["process_id"], bool(row["cancellation_requested"]),
        decode(row["error_json"], None), row["started_at"], row["finished_at"],
        row["created_at"], row["updated_at"],
    )


class WorkloadQualificationService:
    """Own workload processes and retain immutable evidence outside SQLite."""

    def __init__(self, database: AppDatabase, workspace_root: str | Path,
                 *, maximum_concurrent: int = 1) -> None:
        self.database, self.workspace_root = database, Path(workspace_root).resolve()
        self.output_root = self.workspace_root / ".pymrst-app" / "workload-qualifications"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.maximum_concurrent = int(maximum_concurrent)
        if self.maximum_concurrent <= 0:
            raise ValueError("maximum_concurrent must be positive")
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._spe1_deck = self._resolve_spe1_deck()
        self.recover_interrupted()

    def _resolve_spe1_deck(self) -> Path | None:
        configured = os.environ.get("PYMRST_SPE1_DECK", "").strip()
        candidates = [Path(configured).expanduser()] if configured else []
        for root in (self.workspace_root, *self.workspace_root.parents):
            candidates.extend((
                root / "mrst-2024a-Geomech" / "examples" / "data" / "SPE1" / "BENCH_SPE1.DATA",
                root / "examples" / "data" / "SPE1" / "BENCH_SPE1.DATA",
            ))
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved not in seen and resolved.is_file():
                return resolved
            seen.add(resolved)
        return None

    def catalog(self) -> tuple[WorkloadDefinition, ...]:
        reason = (
            "BENCH_SPE1.DATA was not found beside the workspace; install the SPE1 "
            "dataset or set PYMRST_SPE1_DECK to its absolute path."
        )
        return tuple(
            replace(item, available=self._spe1_deck is not None,
                    unavailable_reason="" if self._spe1_deck is not None else reason)
            if item.workload_id == "spe1-blackoil" else item
            for item in WORKLOAD_CATALOG
        )

    def get(self, qualification_id: str) -> WorkloadQualification:
        row = self.database.fetch_one(
            "SELECT * FROM app_workload_qualifications WHERE qualification_id=?",
            (str(qualification_id),),
        )
        if row is None:
            raise KeyError(f"unknown workload qualification: {qualification_id}")
        return _from_row(row)

    def list(self, *, limit: int = 50, workload_id: str = "") -> tuple[WorkloadQualification, ...]:
        limit = int(limit)
        if not 1 <= limit <= 1000:
            raise ValueError("workload history limit must be between 1 and 1000")
        if workload_id:
            rows = self.database.fetch_all(
                "SELECT * FROM app_workload_qualifications WHERE workload_id=? "
                "ORDER BY created_at DESC LIMIT ?", (workload_id, limit),
            )
        else:
            rows = self.database.fetch_all(
                "SELECT * FROM app_workload_qualifications ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return tuple(_from_row(row) for row in rows)

    def submit(self, request: WorkloadRequest) -> WorkloadQualification:
        if not isinstance(request, WorkloadRequest):
            raise TypeError("submit requires WorkloadRequest")
        definition = _CATALOG.get(request.workload_id)
        if definition is None or request.tier not in definition.tiers:
            raise ValueError("unknown workload or unsupported tier")
        available = {item.workload_id: item for item in self.catalog()}[request.workload_id]
        if not available.available:
            raise ValueError(available.unavailable_reason)
        with self._lock:
            active = sum(process.poll() is None for process in self._processes.values())
            queued = self.database.fetch_one(
                "SELECT COUNT(*) FROM app_workload_qualifications "
                "WHERE status IN ('queued','running')"
            )[0]
            if active >= self.maximum_concurrent or queued >= self.maximum_concurrent:
                raise RuntimeError("workload qualification concurrency limit reached")
            identifier, now = f"workload_{uuid4().hex}", utc_now()
            budget = request.budget or definition.default_budgets[request.tier.value]
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO app_workload_qualifications VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (identifier, request.workload_id, request.tier.value,
                     WorkloadStatus.QUEUED.value, 0.0, "Queued",
                     _json(request.to_dict()), _json(budget.to_dict()), "{}", "{}",
                     "", "", None, 0, None, "", "", now, now),
                )
            thread = threading.Thread(target=self._execute, args=(identifier,), daemon=True,
                                      name=f"pymrst-workload-{identifier[-8:]}")
            self._threads[identifier] = thread
            thread.start()
        return self.get(identifier)

    def _update(self, identifier: str, **values) -> None:
        if not values:
            return
        allowed = {
            "status", "progress", "message", "metrics_json", "comparison_json",
            "evidence_relative_path", "content_hash", "process_id",
            "cancellation_requested", "error_json", "started_at", "finished_at",
        }
        if not set(values) <= allowed:
            raise ValueError("invalid workload update fields")
        values["updated_at"] = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_workload_qualifications SET "
                + ",".join(f"{name}=?" for name in values) + " WHERE qualification_id=?",
                (*values.values(), identifier),
            )

    def _command(self, request_path: Path, output: Path) -> list[str]:
        arguments = ["--request", str(request_path), "--output", str(output)]
        if getattr(sys, "frozen", False):
            return [sys.executable, "--workload-worker", *arguments]
        return [sys.executable, "-m", "pymrst.app.workers.workload_entry", *arguments]

    def _baseline(self, record: WorkloadQualification) -> WorkloadQualification | None:
        rows = self.database.fetch_all(
            "SELECT * FROM app_workload_qualifications WHERE workload_id=? AND tier=? "
            "AND status='passed' AND qualification_id<>? ORDER BY created_at DESC LIMIT 50",
            (record.workload_id, record.tier.value, record.qualification_id),
        )
        for row in rows:
            candidate = _from_row(row)
            if int(candidate.metrics.get("measured_runs", 1)) >= 3:
                return candidate
        return None

    @staticmethod
    def _assess(metrics: dict, budget: WorkloadBudget,
                baseline: WorkloadQualification | None) -> tuple[bool, dict]:
        telemetry_summary = dict(dict(metrics.get("telemetry") or {}).get("summary") or {})
        process_io = sum(
            int(telemetry_summary.get(name) or 0)
            for name in ("read_bytes_delta", "write_bytes_delta")
        )
        telemetry_bytes = int(dict(metrics.get("telemetry") or {}).get(
            "artifact_bytes", 0
        ))
        checks = {
            "wall_seconds": {
                "actual": float(metrics["wall_seconds"]),
                "maximum": budget.maximum_wall_seconds,
            },
            "cpu_seconds": {
                "actual": float(metrics["cpu_seconds"]),
                "maximum": budget.maximum_cpu_seconds,
            },
            "peak_rss_bytes": {
                "actual": int(metrics["peak_rss_bytes"]),
                "maximum": budget.maximum_peak_rss_bytes,
            },
            "disk_bytes": {
                "actual": max(
                    int(metrics.get("io_bytes", 0)), process_io + telemetry_bytes,
                ),
                "maximum": budget.maximum_disk_bytes,
            },
        }
        wall_statistics = dict(dict(metrics.get("statistics") or {}).get(
            "wall_seconds") or {})
        wall_median = float(wall_statistics.get("median", metrics["wall_seconds"]))
        wall_mad = float(wall_statistics.get("mad", 0.0))
        checks["wall_relative_mad"] = {
            "actual": wall_mad / wall_median if wall_median > 0 else 0.0,
            "maximum": budget.maximum_wall_relative_mad,
        }
        for value in checks.values():
            value["passed"] = value["actual"] <= value["maximum"]
        correctness = dict(metrics.get("correctness") or {})
        for name, value in correctness.items():
            if not isinstance(value, bool):
                raise ValueError(f"correctness check {name!r} must be boolean")
        comparison = {
            "baseline_qualification_id": "", "ratios": {}, "checks": checks,
            "method": "median_of_measured_runs",
            "warmup_runs": int(metrics.get("warmup_runs", 0)),
            "measured_runs": int(metrics.get("measured_runs", 1)),
        }
        if baseline is not None:
            comparison["baseline_qualification_id"] = baseline.qualification_id
            comparison["baseline_measured_runs"] = int(
                baseline.metrics.get("measured_runs", 1)
            )
            for metric, maximum in (
                ("wall_seconds", budget.maximum_wall_regression_ratio),
                ("peak_rss_bytes", budget.maximum_memory_regression_ratio),
            ):
                previous = float(baseline.metrics.get(metric, 0))
                ratio = float(metrics[metric]) / previous if previous > 0 else 1.0
                comparison["ratios"][metric] = {
                    "actual": ratio, "maximum": maximum, "passed": ratio <= maximum,
                }
        comparison["correctness"] = correctness
        passed = all(correctness.values()) and all(item["passed"] for item in checks.values()) and all(
            item["passed"] for item in comparison["ratios"].values()
        )
        return passed, comparison

    def _execute(self, identifier: str) -> None:
        record = self.get(identifier)
        directory = self.output_root / identifier
        output, request_path = directory / "work", directory / "request.json"
        directory.mkdir(parents=True, exist_ok=False)
        request_path.write_text(_json(record.request) + "\n", encoding="utf-8")
        try:
            request_path.chmod(0o444)
        except OSError:
            pass
        stderr_path = directory / "stderr.log"
        process = None
        try:
            self._update(identifier, status="running", message="Starting isolated workload",
                         started_at=utc_now())
            environment = os.environ.copy()
            if record.workload_id == "spe1-blackoil":
                if self._spe1_deck is None:
                    raise FileNotFoundError("SPE1 deck became unavailable after submission")
                environment["PYMRST_QUALIFICATION_SPE1_DECK"] = str(self._spe1_deck)
            package_root = str(Path(__file__).resolve().parents[3])
            environment["PYTHONPATH"] = package_root + (
                os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
            )
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            with stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    self._command(request_path, output), cwd=directory,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                    text=True, bufsize=1, env=environment,
                    start_new_session=os.name != "nt", creationflags=creationflags,
                )
                with self._lock:
                    self._processes[identifier] = process
                self._update(identifier, process_id=process.pid,
                             message="Isolated workload running")
                result, failure, cancelled = None, None, False
                assert process.stdout is not None
                for line in process.stdout:
                    current = self.get(identifier)
                    if current.cancellation_requested and process.poll() is None:
                        cancelled = True
                        self._terminate(process)
                    try:
                        event = json.loads(line)
                        if event.get("protocol_version") != 1:
                            raise ValueError("unsupported workload protocol")
                    except Exception as error:
                        failure = {"error_type": "WorkloadProtocolError", "message": str(error)}
                        self._terminate(process)
                        continue
                    if event["event"] == "progress":
                        progress = min(max(float(event["progress"]), current.progress), 0.99)
                        self._update(identifier, progress=progress,
                                     message=str(event.get("message", "Running")))
                    elif event["event"] == "result":
                        result = json_value(dict(event["metrics"]))
                    elif event["event"] == "failure":
                        failure = {"error_type": event.get("error_type", "WorkloadError"),
                                   "message": event.get("message", "workload failed")}
                    elif event["event"] == "cancelled":
                        cancelled = True
                return_code = process.wait()
            if cancelled or return_code == 130 or self.get(identifier).cancellation_requested:
                self._update(identifier, status="cancelled", progress=1.0,
                             message="Qualification cancelled", finished_at=utc_now())
            elif return_code == 0 and result is not None and failure is None:
                current = self.get(identifier)
                budget = WorkloadBudget.from_dict(current.budgets)
                passed, comparison = self._assess(result, budget, self._baseline(current))
                evidence = {
                    "schema_version": 2, "qualification_id": identifier,
                    "workload_id": current.workload_id, "tier": current.tier.value,
                    "request": current.request, "budgets": current.budgets,
                    "metrics": result, "comparison": comparison,
                    "status": "passed" if passed else "failed",
                    "platform": {"system": platform.system(), "release": platform.release(),
                                 "machine": platform.machine(), "python": platform.python_version()},
                    "finished_at": utc_now(),
                }
                encoded = (_json(evidence) + "\n").encode()
                evidence_path, temporary = directory / "evidence.json", directory / "evidence.tmp"
                temporary.write_bytes(encoded); os.replace(temporary, evidence_path)
                digest = hashlib.sha256(encoded).hexdigest()
                relative = evidence_path.relative_to(self.workspace_root).as_posix()
                correctness_failed = any(
                    not value for value in comparison.get("correctness", {}).values()
                )
                failure_type = (
                    "CorrectnessInvariantFailed" if correctness_failed
                    else "PerformanceBudgetExceeded"
                )
                failure_message = (
                    "one or more simulator correctness invariants failed"
                    if correctness_failed
                    else "one or more absolute or regression budgets were exceeded"
                )
                self._update(
                    identifier, status="passed" if passed else "failed", progress=1.0,
                    message="Qualification passed" if passed else failure_message,
                    metrics_json=_json(result), comparison_json=_json(comparison),
                    evidence_relative_path=relative, content_hash=digest,
                    error_json=None if passed else _json({
                        "error_type": failure_type, "message": failure_message,
                    }), finished_at=evidence["finished_at"],
                )
            else:
                failure = failure or {"error_type": "WorkloadExitError",
                                      "message": f"worker exited with code {return_code}"}
                self._update(identifier, status="failed", progress=1.0,
                             message="Qualification failed", error_json=_json(failure),
                             finished_at=utc_now())
        except BaseException as error:
            self._update(identifier, status="failed", progress=1.0,
                         message="Qualification orchestration failed",
                         error_json=_json({"error_type": type(error).__name__,
                                           "message": str(error)}), finished_at=utc_now())
        finally:
            with self._lock:
                self._processes.pop(identifier, None)
                self._threads.pop(identifier, None)

    @staticmethod
    def _terminate(process: subprocess.Popen, *, force: bool = False) -> None:
        try:
            if os.name == "nt":
                process.kill() if force else process.terminate()
            else:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    def cancel(self, qualification_id: str) -> WorkloadQualification:
        current = self.get(qualification_id)
        if current.status not in {WorkloadStatus.QUEUED, WorkloadStatus.RUNNING}:
            raise ValueError("only queued or running workload qualifications can be cancelled")
        self._update(qualification_id, cancellation_requested=1,
                     message="Cancellation requested")
        with self._lock:
            process = self._processes.get(qualification_id)
        if process is not None and process.poll() is None:
            self._terminate(process)
        return self.get(qualification_id)

    def verify_evidence(self, qualification_id: str) -> dict:
        record = self.get(qualification_id)
        if not record.evidence_relative_path or not record.content_hash:
            raise ValueError("qualification has no completed evidence bundle")
        path = (self.workspace_root / record.evidence_relative_path).resolve()
        if self.workspace_root not in path.parents:
            raise RuntimeError("qualification evidence escaped the workspace boundary")
        if not path.is_file():
            return {"qualification_id": qualification_id, "valid": False,
                    "reason": "evidence file is missing", "content_hash": ""}
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence_valid = actual == record.content_hash
        artifacts = []
        telemetry = dict(record.metrics.get("telemetry") or {})
        relative = str(telemetry.get("artifact_relative_path", ""))
        expected = str(telemetry.get("artifact_sha256", ""))
        if relative or expected:
            artifact = (path.parent / relative).resolve()
            safe = path.parent in artifact.parents and self.workspace_root in artifact.parents
            artifact_actual = (
                hashlib.sha256(artifact.read_bytes()).hexdigest()
                if safe and artifact.is_file() else ""
            )
            artifacts.append({
                "relative_path": relative, "valid": bool(safe and artifact_actual == expected),
                "content_hash": artifact_actual, "expected_content_hash": expected,
                "reason": (
                    "checksum verified" if safe and artifact_actual == expected
                    else "artifact is missing, unsafe, or has a checksum mismatch"
                ),
            })
        valid = evidence_valid and all(item["valid"] for item in artifacts)
        return {
            "qualification_id": qualification_id, "valid": valid,
            "reason": "all checksums verified" if valid else "evidence bundle checksum mismatch",
            "content_hash": actual, "expected_content_hash": record.content_hash,
            "artifacts": artifacts,
        }

    def recover_interrupted(self) -> tuple[WorkloadQualification, ...]:
        rows = self.database.fetch_all(
            "SELECT qualification_id FROM app_workload_qualifications "
            "WHERE status IN ('queued','running')"
        )
        recovered = []
        for row in rows:
            self._update(row["qualification_id"], status="interrupted", progress=1.0,
                         message="Application restarted during qualification",
                         error_json=_json({"error_type": "ApplicationRestart",
                                           "message": "qualification process ownership was lost"}),
                         finished_at=utc_now())
            recovered.append(self.get(row["qualification_id"]))
        return tuple(recovered)

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            processes, threads = tuple(self._processes.items()), tuple(self._threads.values())
        for identifier, process in processes:
            if process.poll() is None:
                self._update(identifier, cancellation_requested=1,
                             message="Application shutdown requested cancellation")
                self._terminate(process)
        if wait:
            for thread in threads:
                thread.join(timeout=10)
        with self._lock:
            survivors = tuple(self._processes.values())
        for process in survivors:
            if process.poll() is None:
                self._terminate(process, force=True)


__all__ = ["WORKLOAD_CATALOG", "WorkloadQualificationService"]
