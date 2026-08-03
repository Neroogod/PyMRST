"""Standalone and background execution for packed problems."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback

from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem,
)


@dataclass
class BackgroundSimulationInfo:
    path: Path
    process: mp.Process | None = None

    @property
    def lock_path(self) -> Path:
        return self.path / "lock.pymrst"

    @property
    def log_path(self) -> Path:
        return self.path / "simulation.log"

    @property
    def events_path(self) -> Path:
        return self.path / "events.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.path / "run_manifest.json"

    @property
    def running(self) -> bool:
        return self.lock_path.exists()


def _json_safe(value):
    """A payload value reduced to something `json` can write.

    Observer payloads are documented as copied scalars, but this writes to a
    file a detached run is read back from, so anything unexpected is stringified
    rather than allowed to raise inside the hook.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def simulate_packed_problem_standalone(path: str | Path, *,
                                       record_evidence: bool = True):
    """Load ``problem.pkl`` from a work directory and simulate under a lock.

    With ``record_evidence`` the run also leaves ``events.jsonl`` and
    ``run_manifest.json`` beside the existing log. A detached run is the one
    case where nothing can be observed as it happens: the only account of what
    the child did was a single line naming the restart index and a message.

    Written with the standard library alone. This module is replica-layer, and
    the covenant forbids it depending on Research Studio - so the event log is
    JSON Lines by hand rather than through `JsonLinesHandler`, and the manifest
    is a plain dict rather than a `ReproducibilityManifest`. Both are readable
    by the extension without either side importing the other.
    """
    work = Path(path)
    payload = work / "problem.pkl"
    lock = work / "lock.pymrst"
    log = work / "simulation.log"
    work.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        raise RuntimeError("this packed simulation is already running")
    lock.write_text(str(mp.current_process().pid), encoding="utf-8")
    started = time.time()
    counts: dict[str, int] = {}
    stream_out = None
    observer = None
    if record_evidence:
        stream_out = (work / "events.jsonl").open("w", encoding="utf-8")

        def observer(event, payload):          # noqa: F811 - bound only here
            counts[event] = counts.get(event, 0) + 1
            record = {"event": str(event), "sequence": sum(counts.values()),
                      "monotonic": time.monotonic(),
                      "payload": _json_safe(dict(payload))}
            stream_out.write(json.dumps(record, sort_keys=True) + "\n")
            stream_out.flush()

    def write_manifest(problem, ok, status, error=None):
        if not record_evidence:
            return
        manifest = {
            "schema_version": 1,
            "kind": "packed_problem_background_child",
            "base_name": getattr(problem, "BaseName", None),
            "name": getattr(problem, "Name", None),
            "pid": mp.current_process().pid,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "started_unix": started,
            "finished_unix": time.time(),
            "ok": ok,
            "restart_index": (None if status is None
                              else _json_safe(status.RestartIndex)),
            "message": None if status is None else _json_safe(status.Message),
            "error": error,
            "event_counts": dict(sorted(counts.items())),
        }
        (work / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
        )

    problem = None
    try:
        with payload.open("rb") as stream:
            problem = pickle.load(stream)
        ok, status = simulate_packed_problem(
            problem, continue_on_error=True, observer=observer,
        )
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"ok={ok} restart={status.RestartIndex} "
                         f"message={status.Message}\n")
        write_manifest(problem, ok, status)
        return ok, status
    except Exception as error:
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        # A crashed child is exactly when its account matters most, so the
        # manifest is written on the way out rather than only on success.
        write_manifest(problem, False, None, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if stream_out is not None:
            stream_out.close()
        if lock.exists():
            lock.unlink()


def simulate_packed_problem_background(
    problem, *, workdir: str | Path | None = None
) -> BackgroundSimulationInfo:
    """Serialize a problem and launch it in a separate Python process."""
    if workdir is None:
        workdir = Path.cwd() / "simulation_output" / "sim_runner"
    path = Path(workdir) / f"{problem.BaseName}_{problem.Name}"
    path.mkdir(parents=True, exist_ok=True)
    with (path / "problem.pkl").open("wb") as stream:
        pickle.dump(problem, stream, protocol=pickle.HIGHEST_PROTOCOL)
    process = mp.Process(
        target=simulate_packed_problem_standalone, args=(path,),
        name=f"pymrst-{problem.Name}",
    )
    process.start()
    return BackgroundSimulationInfo(path=path, process=process)


def monitor_background_simulations(
    simulations, *, wait: bool = False, timeout: float | None = None
):
    """Return status dictionaries for background simulations."""
    items = ([simulations] if isinstance(simulations, BackgroundSimulationInfo)
             else list(simulations))
    result = []
    for info in items:
        process = info.process
        if wait and process is not None:
            process.join(timeout)
        result.append({
            "path": info.path,
            "running": info.running or (
                process is not None and process.is_alive()
            ),
            "exitcode": None if process is None else process.exitcode,
            "log": (info.log_path.read_text(encoding="utf-8")
                    if info.log_path.exists() else ""),
        })
    return result


simulatePackedProblemStandalone = simulate_packed_problem_standalone
simulatePackedProblemBackground = simulate_packed_problem_background
monitorBackgroundSimulations = monitor_background_simulations

__all__ = [
    "BackgroundSimulationInfo",
    "simulate_packed_problem_standalone",
    "simulatePackedProblemStandalone",
    "simulate_packed_problem_background",
    "simulatePackedProblemBackground",
    "monitor_background_simulations",
    "monitorBackgroundSimulations",
]
