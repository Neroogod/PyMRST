"""M5.1 UI-independent projects, workflows, runs, events and cancellation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
import threading
import time
from typing import Any, Callable, Mapping

from pymrst.research_studio.experiments.definition import canonical_value, content_digest
from pymrst.research_studio.runtime.cancellation import (
    CancellationRequested, CancellationToken,
)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Workflow:
    workflow_id: str
    title: str
    description: str
    execute: Callable
    teaching_safe: bool = False


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    project_id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(slots=True)
class RunRecord:
    run_id: str
    project_id: str
    workflow_id: str
    parameters: Mapping[str, Any]
    status: RunStatus = RunStatus.QUEUED
    progress: float = 0.0
    message: str = "queued"
    created_monotonic: float = field(default_factory=time.monotonic)
    elapsed_seconds: float = 0.0
    result: Any = None
    error_type: str = ""
    error_message: str = ""
    parent_run_id: str = ""
    events: list[dict] = field(default_factory=list)

    def to_dict(self, *, include_result=False):
        value = {"run_id": self.run_id, "project_id": self.project_id,
                 "workflow_id": self.workflow_id,
                 "parameters": dict(self.parameters), "status": self.status.value,
                 "progress": self.progress, "message": self.message,
                 "elapsed_seconds": self.elapsed_seconds,
                 "error_type": self.error_type, "error_message": self.error_message,
                 "parent_run_id": self.parent_run_id,
                 "events": list(self.events)}
        if include_result:
            value["result"] = self.result
        return value


@dataclass(frozen=True, slots=True)
class WorkflowContext:
    run_id: str
    cancellation: CancellationToken
    emit: Callable[[str, float, str, Mapping[str, Any] | None], None]
    parent_result: Any = None


class ResearchStudioService:
    def __init__(self, *, max_workers=2, metadata_store=None):
        if max_workers <= 0:
            raise ValueError("service workers must be positive")
        self._projects = {}
        self._workflows = {}
        self._runs = {}
        self._tokens = {}
        self._resources = {kind: {} for kind in
                           ("stores", "evidence", "comparisons", "packages")}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="pymrst-studio")
        self._metadata = metadata_store

    def _persist_project(self, project):
        if self._metadata is not None:
            self._metadata.put("project", project.project_id, project.to_dict())

    def _persist_run(self, record):
        if self._metadata is not None:
            self._metadata.put("run", record.run_id, record.to_dict())

    def _append_event(self, record, event):
        record.events.append(event)
        if self._metadata is not None:
            self._metadata.append_event(record.run_id, len(record.events)-1, event)
            self._persist_run(record)

    def register_workflow(self, workflow: Workflow):
        with self._lock:
            if workflow.workflow_id in self._workflows:
                raise KeyError(f"workflow already registered: {workflow.workflow_id}")
            self._workflows[workflow.workflow_id] = workflow

    def create_project(self, name, *, description="", tags=()):
        if not name:
            raise ValueError("project name is required")
        project_id = f"project_{content_digest({'name': name, 'description': description, 'tags': sorted(tags)})[:16]}"
        project = ProjectRecord(project_id, str(name), str(description),
                                tuple(sorted(map(str, tags))))
        with self._lock:
            self._projects.setdefault(project_id, project)
            self._persist_project(project)
        return project

    def projects(self):
        with self._lock:
            return tuple(self._projects[key] for key in sorted(self._projects))

    def workflows(self):
        with self._lock:
            return tuple(self._workflows[key] for key in sorted(self._workflows))

    def submit(self, project_id, workflow_id, parameters=None, *, parent_run_id=""):
        with self._lock:
            if project_id not in self._projects:
                raise KeyError(f"unknown project: {project_id}")
            if workflow_id not in self._workflows:
                raise KeyError(f"unknown workflow: {workflow_id}")
            parameters = canonical_value(parameters or {})
            ordinal = len(self._runs)
            run_id = f"run_{content_digest({'project': project_id, 'workflow': workflow_id, 'parameters': parameters, 'parent': parent_run_id, 'ordinal': ordinal})[:20]}"
            record = RunRecord(run_id, project_id, workflow_id, parameters,
                               parent_run_id=parent_run_id)
            token = CancellationToken()
            self._runs[run_id], self._tokens[run_id] = record, token
            self._persist_run(record)
            self._executor.submit(self._execute, record, token)
            return record

    def _execute(self, record, token):
        started = time.monotonic()
        with self._lock:
            record.status, record.message = RunStatus.RUNNING, "running"
            self._append_event(record, {"kind": "run_started", "progress": 0.0,
                                       "message": "running", "payload": {}})
        def emit(kind, progress, message, payload=None):
            token.raise_if_requested()
            if not 0 <= progress <= 1:
                raise ValueError("workflow progress must lie in [0, 1]")
            with self._lock:
                record.progress, record.message = float(progress), str(message)
                self._append_event(record, {"kind": str(kind), "progress": float(progress),
                                           "message": str(message),
                                           "payload": canonical_value(payload or {})})
        try:
            workflow = self._workflows[record.workflow_id]
            parent = (self._runs[record.parent_run_id].result
                      if record.parent_run_id else None)
            context = WorkflowContext(record.run_id, token, emit, parent)
            result = workflow.execute(context, dict(record.parameters))
            token.raise_if_requested()
            with self._lock:
                record.result, record.status = result, RunStatus.COMPLETED
                record.progress, record.message = 1.0, "completed"
                self._append_event(record, {"kind": "run_completed", "progress": 1.0,
                                           "message": "completed", "payload": {}})
        except CancellationRequested as error:
            with self._lock:
                record.status, record.message = RunStatus.CANCELLED, str(error)
                self._append_event(record, {"kind": "run_cancelled",
                                           "progress": record.progress,
                                           "message": str(error), "payload": {}})
        except BaseException as error:
            with self._lock:
                record.status, record.message = RunStatus.FAILED, "failed"
                record.error_type, record.error_message = type(error).__name__, str(error)
                self._append_event(record, {"kind": "run_failed", "progress": record.progress,
                                           "message": str(error),
                                           "payload": {"error_type": type(error).__name__}})
        finally:
            with self._lock:
                record.elapsed_seconds = time.monotonic()-started
                self._persist_run(record)

    def cancel(self, run_id):
        with self._lock:
            if run_id not in self._tokens:
                raise KeyError(f"unknown run: {run_id}")
            self._tokens[run_id].request()

    def resume(self, run_id, *, parameters=None):
        with self._lock:
            source = self._runs[run_id]
            if source.status not in {RunStatus.COMPLETED, RunStatus.FAILED,
                                     RunStatus.CANCELLED}:
                raise ValueError("only terminal runs can be resumed or branched")
            merged = dict(source.parameters)
            merged.update(parameters or {})
        return self.submit(source.project_id, source.workflow_id, merged,
                           parent_run_id=run_id)

    def run(self, run_id):
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as error:
                raise KeyError(f"unknown run: {run_id}") from error

    def runs(self, *, project_id=None):
        with self._lock:
            values = tuple(self._runs.values())
            if project_id is not None:
                values = tuple(value for value in values
                               if value.project_id == project_id)
            return tuple(sorted(values, key=lambda value: value.created_monotonic))

    def snapshot(self):
        return {"projects": [value.to_dict() for value in self.projects()],
                "workflows": [{"workflow_id": value.workflow_id,
                               "title": value.title,
                               "description": value.description,
                               "teaching_safe": value.teaching_safe}
                              for value in self.workflows()],
                "runs": [value.to_dict() for value in self.runs()],
                "resources": {kind: len(values)
                              for kind, values in self._resources.items()}}

    def publish_resource(self, kind, resource_id, value, *, run_id=""):
        if kind not in self._resources:
            raise ValueError(f"unsupported resource kind: {kind}")
        if not resource_id:
            raise ValueError("resource id is required")
        normalized = canonical_value(value)
        record = {"resource_id": str(resource_id), "kind": kind,
                  "run_id": str(run_id), "value": normalized}
        with self._lock:
            if resource_id in self._resources[kind]:
                raise KeyError(f"resource already published: {kind}/{resource_id}")
            self._resources[kind][resource_id] = record
        return record

    def resources(self, kind):
        if kind not in self._resources:
            raise ValueError(f"unsupported resource kind: {kind}")
        with self._lock:
            return tuple(self._resources[kind][key]
                         for key in sorted(self._resources[kind]))

    def resource(self, kind, resource_id):
        if kind not in self._resources:
            raise ValueError(f"unsupported resource kind: {kind}")
        with self._lock:
            try:
                return self._resources[kind][resource_id]
            except KeyError as error:
                raise KeyError(f"unknown resource: {kind}/{resource_id}") from error

    def shutdown(self, *, wait=True):
        self._executor.shutdown(wait=wait, cancel_futures=True)
        if self._metadata is not None:
            self._metadata.close()


__all__ = ["ProjectRecord", "ResearchStudioService", "RunRecord", "RunStatus",
           "Workflow", "WorkflowContext"]
