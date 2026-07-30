"""Runnable packed-problem demonstrations without plotting dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymrst.modules.ad_core.simulators.sim_runner import (
    PackedProblemManager,
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    simulate_packed_problem,
)
from pymrst.modules.ad_core.utils.schedule import simple_schedule


@dataclass
class _RateModel:
    rate: float

    def step(self, state, dt):
        return {"x": float(state["x"]) + self.rate * float(dt)}, True, 1


@dataclass(frozen=True)
class PackedDemoResult:
    names: tuple[str, ...]
    final_values: tuple[float, ...]
    report_times: tuple[np.ndarray, ...]
    complete_steps: tuple[int, ...]


def demo_packed_problems(
    directory: str | Path,
    *,
    timesteps=(1.0, 2.0, 3.0),
) -> PackedDemoResult:
    """Run MRST ``demoPackedProblems`` persistence and collection workflow."""

    schedule = simple_schedule(timesteps)
    problems = [
        pack_simulation_problem(
            {"x": 0.0},
            _RateModel(rate),
            schedule,
            "PackedDemo",
            directory=directory,
            name=name,
            description=f"Scalar displacement surrogate with rate {rate:g}",
        )
        for name, rate in (("linear", 1.0), ("quadratic", 2.0))
    ]
    for problem in problems:
        ok, _ = simulate_packed_problem(problem)
        if not ok:
            raise RuntimeError(f"packed simulation failed: {problem.Name}")
    _, states, _, names, times, _, _ = get_multiple_packed_simulator_outputs(
        problems, min_steps=1
    )
    return PackedDemoResult(
        names=tuple(names),
        final_values=tuple(float(case[-1]["x"]) for case in states),
        report_times=tuple(np.asarray(time, dtype=float) for time in times),
        complete_steps=tuple(len(case) for case in states),
    )


def ensemble_packed_problems_example(
    directory: str | Path,
    *,
    realization_count: int = 5,
    timesteps=(1.0, 1.0, 2.0),
    seed: int = 0,
) -> PackedDemoResult:
    """Run the non-plotting workflow of MRST ``ensemblePackedProblemsExample``."""

    if realization_count < 1:
        raise ValueError("realization_count must be positive")
    rng = np.random.default_rng(seed)
    rates = rng.uniform(0.5, 1.5, realization_count)
    schedule = simple_schedule(timesteps)
    problems = [
        pack_simulation_problem(
            {"x": 0.0},
            _RateModel(float(rate)),
            schedule,
            "PackedEnsemble",
            directory=directory,
            name=f"realization_{index + 1}",
            description=f"Deterministic ensemble rate {rate:.8g}",
        )
        for index, rate in enumerate(rates)
    ]
    manager = PackedProblemManager(problems)
    ok, _ = manager.simulate_problem()
    if not np.all(ok):
        failed = [problem.Name for problem, success in zip(problems, ok) if not success]
        raise RuntimeError(f"packed ensemble failed: {failed}")
    _, states, _, names, times, _, _ = get_multiple_packed_simulator_outputs(
        problems, min_steps=1
    )
    return PackedDemoResult(
        names=tuple(names),
        final_values=tuple(float(case[-1]["x"]) for case in states),
        report_times=tuple(np.asarray(time, dtype=float) for time in times),
        complete_steps=tuple(len(case) for case in states),
    )


demoPackedProblems = demo_packed_problems
ensemblePackedProblemsExample = ensemble_packed_problems_example

__all__ = [
    "PackedDemoResult",
    "demo_packed_problems",
    "ensemble_packed_problems_example",
    "demoPackedProblems",
    "ensemblePackedProblemsExample",
]
