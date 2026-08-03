"""Hard-bounded subprocess execution for production Research Studio workflows."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import multiprocessing
import queue
import time


class WorkflowTimeoutError(TimeoutError):
    pass


class WorkflowProcessError(RuntimeError):
    pass


def _resolve(owner):
    module, separator, name = owner.partition(":")
    if not separator or not module or not name:
        raise ValueError("workflow owner must use module:function syntax")
    return getattr(importlib.import_module(module), name)


def _child(owner, parameters, messages):
    try:
        result = _resolve(owner)(parameters)
        messages.put(("result", result))
    except BaseException as error:
        messages.put(("error", (type(error).__name__, str(error))))


@dataclass(frozen=True, slots=True)
class ProcessExecutionPolicy:
    maximum_wall_seconds: float = 3600.0
    start_method: str = "spawn"

    def __post_init__(self):
        if self.maximum_wall_seconds <= 0:
            raise ValueError("maximum wall time must be positive")


class IsolatedWorkflowExecutor:
    """Execute an importable workflow with a hard wall-clock kill boundary."""

    def __init__(self, policy=ProcessExecutionPolicy()):
        self.policy = policy

    def execute(self, owner, parameters, *, cancellation=None):
        _resolve(owner)  # fail before creating a process
        context = multiprocessing.get_context(self.policy.start_method)
        messages = context.Queue()
        process = context.Process(target=_child,
                                  args=(owner, dict(parameters), messages), daemon=True)
        process.start()
        deadline = time.monotonic() + self.policy.maximum_wall_seconds
        try:
            while process.is_alive():
                if cancellation is not None and cancellation.requested:
                    process.terminate(); process.join(2)
                    raise WorkflowProcessError("workflow cancelled and subprocess terminated")
                if time.monotonic() >= deadline:
                    process.terminate(); process.join(2)
                    raise WorkflowTimeoutError(
                        f"workflow exceeded {self.policy.maximum_wall_seconds:g} seconds")
                process.join(min(.05, max(0.0, deadline-time.monotonic())))
            process.join()
            try:
                kind, value = messages.get(timeout=.5)
            except queue.Empty as error:
                raise WorkflowProcessError(
                    f"workflow subprocess exited with code {process.exitcode} without a result") from error
            if kind == "error":
                raise WorkflowProcessError(f"{value[0]}: {value[1]}")
            return value
        finally:
            if process.is_alive():
                process.terminate(); process.join(2)
            messages.close()


__all__ = ["IsolatedWorkflowExecutor", "ProcessExecutionPolicy",
           "WorkflowProcessError", "WorkflowTimeoutError"]
