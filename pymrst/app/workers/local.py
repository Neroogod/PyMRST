"""Threaded executor restricted to explicitly safe, non-simulation application work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Any, Callable, Mapping

from ..schemas.task import TaskStatus
from ..services.task_service import TaskService


class TaskCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TaskExecutionContext:
    task_id: str
    _service: TaskService

    @property
    def cancellation_requested(self) -> bool:
        return self._service.get(self.task_id).cancellation_requested

    def raise_if_cancelled(self) -> None:
        if self.cancellation_requested:
            raise TaskCancelled("Cancellation requested")

    def progress(self, value: float, message: str) -> None:
        self.raise_if_cancelled()
        self._service.update_progress(self.task_id, value, message)


@dataclass(frozen=True, slots=True)
class _Handler:
    function: Callable[[TaskExecutionContext, Mapping[str, Any]], Mapping[str, Any]]
    in_process_safe: bool


class BackgroundTaskManager:
    """Execute UI bookkeeping work; numerical workloads require isolated workers."""

    _isolated_prefixes = ("simulation", "ensemble", "matlab", "opm", "hpc", "remote")

    def __init__(self, tasks: TaskService, *, max_workers: int = 2) -> None:
        if max_workers <= 0:
            raise ValueError("background task worker count must be positive")
        self.tasks = tasks
        self._handlers: dict[str, _Handler] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pymrst-app-safe",
        )

    def register(self, kind: str, handler, *, in_process_safe: bool = False) -> None:
        kind = str(kind).strip()
        if not kind or not callable(handler):
            raise ValueError("background handlers require a kind and callable")
        if not in_process_safe or kind.lower().startswith(self._isolated_prefixes):
            raise ValueError(
                "in-process execution is restricted to explicitly safe non-simulation tasks"
            )
        with self._lock:
            if kind in self._handlers:
                raise KeyError(f"background handler already registered: {kind}")
            self._handlers[kind] = _Handler(handler, True)

    def submit(self, project_id: str, kind: str, request=None):
        with self._lock:
            if kind not in self._handlers:
                raise KeyError(f"unknown safe background handler: {kind}")
            task = self.tasks.create(project_id, kind, request)
            self._executor.submit(self._run, task.task_id)
            return task

    def _run(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if task.status is TaskStatus.CANCELLED:
            return
        try:
            task = self.tasks.transition(
                task_id, TaskStatus.RUNNING, progress=0.0, message="Running",
            )
            context = TaskExecutionContext(task_id, self.tasks)
            result = self._handlers[task.kind].function(context, dict(task.request))
            context.raise_if_cancelled()
            if not isinstance(result, Mapping):
                raise TypeError("background task handlers must return a mapping")
            self.tasks.transition(
                task_id, TaskStatus.COMPLETED, message="Completed", result=result,
            )
        except TaskCancelled as error:
            current = self.tasks.get(task_id)
            if current.status is TaskStatus.RUNNING:
                self.tasks.transition(
                    task_id, TaskStatus.CANCELLED, message=str(error),
                )
        except BaseException as error:
            current = self.tasks.get(task_id)
            if current.status is TaskStatus.RUNNING:
                self.tasks.transition(
                    task_id, TaskStatus.FAILED, message="Task failed",
                    error={"error_type": type(error).__name__, "message": str(error)},
                )

    def shutdown(self, *, wait=True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)


__all__ = ["BackgroundTaskManager", "TaskCancelled", "TaskExecutionContext"]
