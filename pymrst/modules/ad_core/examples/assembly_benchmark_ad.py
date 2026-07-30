"""Equation-assembly timing utility corresponding to MRST ``assemblyBenchmarkAD``."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class AssemblyBenchmarkCase:
    """One model/state combination to benchmark."""

    model: Any
    state0: Any
    state: Any
    dt: float
    physics: str = "reservoir"
    backend: str | None = None
    forces: Any = None


@dataclass(frozen=True)
class AssemblyBenchmarkResult:
    """Median state conversion and equation assembly timings."""

    physics: str
    backend: str
    cell_count: int
    degrees_of_freedom: int
    state_time: float
    equation_time: float
    iterations: int

    @property
    def total_time(self) -> float:
        return self.state_time + self.equation_time

    @property
    def seconds_per_cell(self) -> float:
        return self.total_time / self.cell_count

    @property
    def seconds_per_dof(self) -> float:
        return self.total_time / self.degrees_of_freedom

    @property
    def state_fraction(self) -> float:
        return self.state_time / self.total_time


def _backend_label(case: AssemblyBenchmarkCase) -> str:
    if case.backend is not None:
        return str(case.backend)
    backend = getattr(
        case.model,
        "auto_diff_backend",
        getattr(case.model, "AutoDiffBackend", None),
    )
    if backend is None:
        return "Sparse"
    label = getattr(backend, "label", None)
    return str(label() if callable(label) else getattr(backend, "name", backend))


def _initialize_state(model, state):
    function = getattr(model, "init_state_ad", None)
    if function is None:
        function = getattr(model, "getStateAD", None)
    if function is None:
        raise TypeError("model must implement init_state_ad or getStateAD")
    return function(state)


def _assemble(case, state0_ad, state_ad):
    model = case.model
    function = getattr(model, "get_model_equations", None)
    if function is None:
        function = getattr(model, "getModelEquations", None)
    if function is None:
        function = getattr(model, "get_equations", None)
    if function is None:
        raise TypeError(
            "model must implement get_model_equations, getModelEquations, or get_equations"
        )
    if case.forces is None:
        return function(state0_ad, state_ad, case.dt)
    try:
        return function(state0_ad, state_ad, case.dt, case.forces)
    except TypeError:
        # Native PyMRST reservoir models keep driving forces on the model.
        return function(state0_ad, state_ad, case.dt)


def _cell_count(model) -> int:
    grid = getattr(model, "G", None)
    cells = getattr(grid, "cells", None)
    count = getattr(cells, "num", None)
    if count is None:
        raise TypeError("model.G.cells.num is required for normalized timings")
    count = int(count)
    if count < 1:
        raise ValueError("model must contain at least one cell")
    return count


def _degrees_of_freedom(model, state_ad, cell_count: int) -> int:
    for value in getattr(state_ad, "values", lambda: ())():
        jacobians = getattr(value, "jac", None)
        if jacobians:
            return int(sum(block.shape[1] for block in jacobians))
    components = getattr(model, "get_number_of_components", None)
    if components is not None:
        return max(1, int(components()) * cell_count)
    return cell_count


def assembly_benchmark_ad(
    cases: AssemblyBenchmarkCase | Iterable[AssemblyBenchmarkCase],
    *,
    iterations: int = 10,
    reducer: Callable[[np.ndarray], float] = np.median,
) -> list[AssemblyBenchmarkResult]:
    """Benchmark AD state initialization separately from residual assembly.

    Unlike MRST's script, model construction is intentionally outside this
    routine. This makes the benchmark usable for black-oil, compositional, and
    user-defined models without hidden dataset or plotting dependencies.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if isinstance(cases, AssemblyBenchmarkCase):
        cases = (cases,)
    results = []
    for case in cases:
        if not isinstance(case, AssemblyBenchmarkCase):
            raise TypeError("cases must contain AssemblyBenchmarkCase instances")
        cell_count = _cell_count(case.model)
        state_times = []
        equation_times = []
        last_state = None
        for _ in range(iterations):
            start = perf_counter()
            state0_ad = _initialize_state(case.model, case.state0)
            state_ad = _initialize_state(case.model, case.state)
            state_times.append(perf_counter() - start)
            start = perf_counter()
            _assemble(case, state0_ad, state_ad)
            equation_times.append(perf_counter() - start)
            last_state = state_ad
        dof = _degrees_of_freedom(case.model, last_state, cell_count)
        results.append(
            AssemblyBenchmarkResult(
                physics=str(case.physics),
                backend=_backend_label(case),
                cell_count=cell_count,
                degrees_of_freedom=dof,
                state_time=float(reducer(np.asarray(state_times))),
                equation_time=float(reducer(np.asarray(equation_times))),
                iterations=iterations,
            )
        )
    return results


assemblyBenchmarkAD = assembly_benchmark_ad

__all__ = [
    "AssemblyBenchmarkCase",
    "AssemblyBenchmarkResult",
    "assembly_benchmark_ad",
    "assemblyBenchmarkAD",
]
