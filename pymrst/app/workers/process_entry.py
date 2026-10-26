"""Private child-process entrypoint for the structured worker protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import sys
from time import monotonic
import traceback


_cancelled = False


def _cancel(*_):
    global _cancelled
    _cancelled = True


class WorkerContext:
    def __init__(self, run_id: str, run_directory: Path) -> None:
        self.run_id, self.run_directory = run_id, run_directory
        self.started = monotonic()

    @property
    def cancellation_requested(self) -> bool:
        return _cancelled

    def raise_if_cancelled(self) -> None:
        if _cancelled:
            raise InterruptedError("worker cancellation requested")

    def emit(self, event_type: str, **payload) -> None:
        record = {
            "protocol_version": 1, "event_type": str(event_type),
            "elapsed_seconds": max(0.0, monotonic() - self.started), "payload": payload,
        }
        sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def progress(self, fraction: float, message: str, **details) -> None:
        value = float(fraction)
        if not 0 <= value <= 1:
            raise ValueError("worker progress must lie in [0,1]")
        self.emit("progress", progress=value, message=str(message), **details)

    def checkpoint(self, relative_path: str, **metadata) -> None:
        path = (self.run_directory / relative_path).resolve()
        if self.run_directory.resolve() not in path.parents or not path.is_file():
            raise ValueError("checkpoint must exist inside run directory")
        self.emit("checkpoint", relative_path=path.relative_to(self.run_directory).as_posix(), metadata=metadata)


def _resolve(path: str):
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name.startswith("pymrst.") or not attribute:
        raise ValueError("worker entrypoint must be an explicit pymrst.module:callable")
    owner = importlib.import_module(module_name)
    function = getattr(owner, attribute)
    if not callable(function):
        raise TypeError("worker entrypoint is not callable")
    return function


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-directory", required=True)
    args = parser.parse_args(argv)
    request_path, run_directory = Path(args.request).resolve(), Path(args.run_directory).resolve()
    if request_path.parent != run_directory or not request_path.is_file():
        raise ValueError("worker request must be inside its run directory")
    signal.signal(signal.SIGTERM, _cancel)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _cancel)
    context = WorkerContext(args.run_id, run_directory)
    try:
        request = json.loads(request_path.read_text())
        context.emit("worker_started", process_id=os.getpid())
        result = _resolve(args.entrypoint)(context, request)
        context.raise_if_cancelled()
        if not isinstance(result, dict):
            raise TypeError("isolated worker must return a dictionary")
        context.emit("result", result=result)
        context.progress(1.0, "Completed")
        return 0
    except InterruptedError as error:
        context.emit("cancelled", message=str(error))
        return 130
    except BaseException as error:
        context.emit(
            "failure", error_type=type(error).__name__, message=str(error),
            traceback="".join(traceback.format_exception(error))[-12000:],
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["WorkerContext", "main"]
