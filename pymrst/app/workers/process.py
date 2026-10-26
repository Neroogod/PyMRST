"""Subprocess-isolated numerical task manager with a structured event protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from time import monotonic
from typing import Any

from ..schemas._common import utc_now
from ..schemas.run import RunStatus
from ..schemas.task import TaskStatus
from ..services.run_service import RunService
from ..services.task_service import TaskService


_ENTRYPOINT = re.compile(r"^pymrst(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class RunnerRegistration:
    kind: str
    entrypoint: str
    requires_model: bool = False


class IsolatedRunManager:
    """Owns child processes; user payloads can never select executable code."""

    def __init__(self, runs: RunService, tasks: TaskService, *, cancellation_grace=3.0) -> None:
        self.runs, self.tasks = runs, tasks
        self.cancellation_grace = float(cancellation_grace)
        self._registrations: dict[str, RunnerRegistration] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def register(self, kind: str, entrypoint: str, *, requires_model: bool = False) -> None:
        kind, entrypoint = str(kind).strip(), str(entrypoint).strip()
        if not kind or not _ENTRYPOINT.fullmatch(entrypoint):
            raise ValueError("isolated runner requires a trusted PyMRST entrypoint")
        with self._lock:
            if kind in self._registrations:
                raise KeyError(f"runner already registered: {kind}")
            self._registrations[kind] = RunnerRegistration(kind, entrypoint, bool(requires_model))

    def submit(self, configuration_id: str) -> Any:
        configuration = self.runs.configuration(configuration_id)
        if configuration.runner_kind not in self._registrations:
            raise KeyError(f"no isolated runner registered for {configuration.runner_kind}")
        if self._registrations[configuration.runner_kind].requires_model and not configuration.model_id:
            raise ValueError(f"runner {configuration.runner_kind} requires an inspected model")
        run = self.runs.create_run(configuration_id)
        self._launch(run.run_id)
        return run

    def resume(self, run_id: str, checkpoint_id: str) -> Any:
        parent = self.runs.get(run_id); checkpoint = self.runs.checkpoint(checkpoint_id)
        if checkpoint.run_id != run_id:
            raise ValueError("checkpoint does not belong to parent run")
        self.runs.checkpoint_path(checkpoint_id)
        run = self.runs.create_run(
            parent.configuration_id, parent_run_id=run_id,
            resume_checkpoint_id=checkpoint_id,
        )
        self._launch(run.run_id)
        return run

    def _launch(self, run_id: str) -> None:
        thread = threading.Thread(
            target=self._execute, args=(run_id,), daemon=True,
            name=f"pymrst-run-{run_id[-8:]}",
        )
        with self._lock:
            self._threads[run_id] = thread
        thread.start()

    def _execute(self, run_id: str) -> None:
        run = self.runs.get(run_id); task = self.tasks.get(run.task_id)
        configuration = self.runs.configuration(run.configuration_id)
        registration = self._registrations[configuration.runner_kind]
        project = self.runs.projects.get(run.project_id)
        directory = (Path(project.root_path) / run.run_relative_path).resolve()
        request = dict(configuration.payload)
        request["_application"] = {
            "project_id": configuration.project_id,
            "model_id": configuration.model_id,
            "run_id": run_id,
            "workspace_root": str(self.runs.projects.workspace_root),
        }
        if configuration.model_id:
            model = self.runs.database.fetch_one(
                "SELECT asset_id,source_hash FROM app_models WHERE model_id=? AND project_id=?",
                (configuration.model_id, configuration.project_id),
            )
            if model is None:
                raise RuntimeError("run configuration model is no longer available")
            request["_application"]["source_path"] = str(
                self.runs.projects.asset_path(configuration.project_id, model["asset_id"])
            )
            request["_application"]["source_hash"] = model["source_hash"]
        if run.resume_checkpoint_id:
            checkpoint = self.runs.checkpoint(run.resume_checkpoint_id)
            request["_resume"] = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "path": str(self.runs.checkpoint_path(checkpoint.checkpoint_id)),
                "content_hash": checkpoint.content_hash,
                "metadata": dict(checkpoint.metadata),
            }
        request_path = directory / "worker_request.json"
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        os.chmod(request_path, 0o444)
        worker_arguments = [
            "--entrypoint", registration.entrypoint, "--request", str(request_path),
            "--run-id", run_id, "--run-directory", str(directory),
        ]
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--worker", *worker_arguments]
        else:
            command = [
                sys.executable, "-m", "pymrst.app.workers.process_entry",
                *worker_arguments,
            ]
        stderr_path = directory / "stderr.log"
        started = monotonic(); result = None; failure = None; cancelled = False
        try:
            self.tasks.transition(task.task_id, TaskStatus.RUNNING, message="Starting isolated worker")
            self.runs.set_status(run_id, RunStatus.STARTING, started_at=utc_now())
            with stderr_path.open("wb") as stderr:
                environment = os.environ.copy()
                package_root = str(Path(__file__).resolve().parents[3])
                environment["PYTHONPATH"] = package_root + (
                    os.pathsep + environment["PYTHONPATH"]
                    if environment.get("PYTHONPATH") else ""
                )
                process = subprocess.Popen(
                    command, cwd=directory, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
                    start_new_session=True, env=environment,
                )
                with self._lock:
                    self._processes[run_id] = process
                self.runs.set_status(run_id, RunStatus.RUNNING, process_id=process.pid)
                self.runs.append_event(run_id, "process_started", 0, {
                    "process_id": process.pid, "runner_kind": configuration.runner_kind,
                    "attempt_number": run.attempt_number,
                })
                assert process.stdout is not None
                for line in process.stdout:
                    if self.tasks.get(task.task_id).cancellation_requested and process.poll() is None:
                        cancelled = True; self._signal(process, signal.SIGTERM)
                    try:
                        record = json.loads(line)
                        if record.get("protocol_version") != 1:
                            raise ValueError("unsupported worker protocol")
                        event_type = str(record["event_type"]); payload = dict(record.get("payload") or {})
                        elapsed = float(record.get("elapsed_seconds", monotonic() - started))
                    except Exception as error:
                        failure = {"error_type": "WorkerProtocolError", "message": str(error), "line": line[:2000]}
                        self._signal(process, signal.SIGTERM); continue
                    self.runs.append_event(run_id, event_type, elapsed, payload)
                    if event_type == "worker_started":
                        self.tasks.update_progress(task.task_id, 0, "Worker started")
                    elif event_type == "progress":
                        progress = min(float(payload.get("progress", 0)), 0.999999)
                        current = self.tasks.get(task.task_id)
                        if progress >= current.progress:
                            self.tasks.update_progress(task.task_id, progress, str(payload.get("message", "Running")))
                    elif event_type == "checkpoint":
                        self.runs.add_checkpoint(run_id, payload["relative_path"], payload.get("metadata", {}))
                    elif event_type == "result":
                        result = payload.get("result")
                    elif event_type == "failure":
                        failure = payload
                    elif event_type == "cancelled":
                        cancelled = True
                return_code = process.wait()
            if cancelled or return_code == 130 or self.tasks.get(task.task_id).cancellation_requested:
                if self.tasks.get(task.task_id).status is TaskStatus.RUNNING:
                    self.tasks.transition(task.task_id, TaskStatus.CANCELLED, message="Run cancelled")
                # Publish the terminal run state only after its task record is
                # durable.  Readers treat a terminal run as an atomic promise
                # that result/error metadata is immediately available.
                self.runs.set_status(run_id, RunStatus.CANCELLED, finished_at=utc_now())
            elif return_code == 0 and failure is None and result is not None:
                self.tasks.transition(task.task_id, TaskStatus.COMPLETED, message="Run completed", result=result)
                self.runs.set_status(run_id, RunStatus.COMPLETED, finished_at=utc_now())
            else:
                failure = failure or {"error_type": "WorkerExitError", "message": f"worker exited with code {return_code}"}
                self.tasks.transition(task.task_id, TaskStatus.FAILED, message="Run failed", error=failure)
                self.runs.set_status(run_id, RunStatus.FAILED, finished_at=utc_now())
        except BaseException as error:
            failure = {"error_type": type(error).__name__, "message": str(error)}
            current = self.tasks.get(task.task_id)
            if current.status is TaskStatus.RUNNING:
                self.tasks.transition(task.task_id, TaskStatus.FAILED, message="Worker orchestration failed", error=failure)
            self.runs.set_status(run_id, RunStatus.FAILED, finished_at=utc_now())
            try:
                self.runs.append_event(run_id, "orchestration_failure", monotonic() - started, failure)
            except Exception:
                pass
        finally:
            with self._lock:
                self._processes.pop(run_id, None); self._threads.pop(run_id, None)

    @staticmethod
    def _signal(process: subprocess.Popen, requested_signal) -> None:
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            pass

    def cancel(self, run_id: str) -> Any:
        run = self.runs.get(run_id)
        task = self.tasks.request_cancellation(run.task_id)
        with self._lock:
            process = self._processes.get(run_id)
        if process is not None and process.poll() is None:
            self.runs.set_status(run_id, RunStatus.CANCELLING)
            self._signal(process, signal.SIGTERM)
            threading.Thread(
                target=self._force_after_grace, args=(run_id, process), daemon=True,
                name=f"pymrst-cancel-{run_id[-8:]}",
            ).start()
        return task

    def _force_after_grace(self, run_id: str, process: subprocess.Popen) -> None:
        deadline = monotonic() + self.cancellation_grace
        while process.poll() is None and monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            self.runs.append_event(run_id, "forced_termination", 0, {
                "reason": "cancellation grace period exceeded",
                "grace_seconds": self.cancellation_grace,
            })
            self._signal(process, signal.SIGKILL)

    def shutdown(self, *, wait=True) -> None:
        with self._lock:
            owners = tuple(self._processes.items()); threads = tuple(self._threads.values())
        for run_id, process in owners:
            if process.poll() is None:
                self.tasks.request_cancellation(self.runs.get(run_id).task_id)
                self._signal(process, signal.SIGTERM)
        if wait:
            for thread in threads:
                thread.join(timeout=max(1, self.cancellation_grace + 1))
        with self._lock:
            survivors = tuple(self._processes.values())
        for process in survivors:
            if process.poll() is None:
                self._signal(process, signal.SIGKILL)


__all__ = ["IsolatedRunManager", "RunnerRegistration"]
