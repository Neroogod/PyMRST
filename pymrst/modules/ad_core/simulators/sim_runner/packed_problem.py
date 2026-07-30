"""Atomic packed-problem representation and disk result handlers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any, Iterable

from pymrst.modules.ad_core.utils.obj2hash import obj2hash


class ResultHandler:
    """Small, restart-safe analogue of MRST's numbered ``ResultHandler``."""

    def __init__(self, directory: str | Path, prefix: str):
        self.directory = Path(directory)
        self.prefix = str(prefix)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, index: int) -> Path:
        if index < 0:
            raise IndexError("result indices must be non-negative")
        return self.directory / f"{self.prefix}_{index + 1:06d}.pkl"

    def indices(self) -> list[int]:
        out = []
        for path in self.directory.glob(f"{self.prefix}_[0-9]*.pkl"):
            try:
                out.append(int(path.stem.rsplit("_", 1)[1]) - 1)
            except ValueError:
                continue
        return sorted(out)

    def __len__(self) -> int:
        return len(self.indices())

    numel_data = __len__
    numelData = __len__

    def __getitem__(self, index: int):
        with self._path(index).open("rb") as stream:
            return pickle.load(stream)

    def __setitem__(self, index: int, value: Any):
        target = self._path(index)
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)

    def reset_data(self, indices: Iterable[int] | None = None):
        selected = self.indices() if indices is None else list(indices)
        for index in selected:
            path = self._path(int(index))
            if path.exists():
                path.unlink()

    resetData = reset_data


@dataclass
class PackedSimulationProblem:
    BaseName: str
    Name: str
    Description: str
    SimulatorSetup: Any
    Modules: list[str]
    OutputHandlers: Any

    @property
    def base_name(self):
        return self.BaseName

    @property
    def name(self):
        return self.Name


def pack_simulation_problem(
    state0,
    model,
    schedule,
    base_name: str,
    *,
    directory: str | Path | None = None,
    name: str | None = None,
    nonlinear_solver=None,
    modules: Iterable[str] = (),
    description: str | None = None,
    extra_arguments: Iterable[Any] = (),
    output_ministeps: bool = False,
    use_hash: bool = False,
) -> PackedSimulationProblem:
    """Pack simulation inputs and create isolated on-disk output handlers."""
    base_name = str(base_name)
    plain_name = type(model).__name__ if not name else str(name)
    problem_name = plain_name
    if use_hash:
        digest = obj2hash((state0, model, schedule))
        problem_name = f"{plain_name}_{digest}"
    if directory is None:
        directory = Path.cwd() / "simulation_output" / base_name
    root = Path(directory) / problem_name
    prefix = "sub" if output_ministeps else ""
    setup = SimpleNamespace(
        state0=state0,
        model=model,
        schedule=schedule,
        OutputMinisteps=bool(output_ministeps),
        NonLinearSolver=nonlinear_solver,
        ExtraArguments=list(extra_arguments),
    )
    handlers = SimpleNamespace(
        states=ResultHandler(root, f"{prefix}state"),
        reports=ResultHandler(root, f"{prefix}report"),
        wellSols=ResultHandler(root, f"{prefix}wellSols"),
    )
    return PackedSimulationProblem(
        BaseName=base_name,
        Name=problem_name,
        Description=(description if description is not None
                     else f"{plain_name}_{type(model).__name__}"),
        SimulatorSetup=setup,
        Modules=list(modules),
        OutputHandlers=handlers,
    )


packSimulationProblem = pack_simulation_problem

__all__ = [
    "PackedSimulationProblem",
    "ResultHandler",
    "pack_simulation_problem",
    "packSimulationProblem",
]
