"""Standalone and background execution for packed problems."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import pickle
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
    def running(self) -> bool:
        return self.lock_path.exists()


def simulate_packed_problem_standalone(path: str | Path):
    """Load ``problem.pkl`` from a work directory and simulate under a lock."""
    work = Path(path)
    payload = work / "problem.pkl"
    lock = work / "lock.pymrst"
    log = work / "simulation.log"
    work.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        raise RuntimeError("this packed simulation is already running")
    lock.write_text(str(mp.current_process().pid), encoding="utf-8")
    try:
        with payload.open("rb") as stream:
            problem = pickle.load(stream)
        ok, status = simulate_packed_problem(
            problem, continue_on_error=True
        )
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"ok={ok} restart={status.RestartIndex} "
                         f"message={status.Message}\n")
        return ok, status
    except Exception:
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise
    finally:
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
