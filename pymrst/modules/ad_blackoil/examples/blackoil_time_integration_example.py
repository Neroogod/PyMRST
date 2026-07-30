"""Compare FIM, CFL-controlled IMPES, and adaptive-implicit black oil.

This ports the executable core of MRST ``blackoilTimeIntegrationExample.m``.
Each variant gets an independent SPE1 model and state, crosses identical
report steps, and records nonlinear convergence, actual ministeps, and AIM's
implicit-cell fraction. SPE9 remains guarded until its deck/schedule parser is
available; selecting it fails before any simulation is run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    SPE1TutorialResult,
    _collect_history,
    _well_indices,
    setup_spe1_tutorial,
)
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport
from pymrst.modules.ad_core.utils import set_time_discretization


@dataclass
class TimeIntegrationDiagnostics:
    report_steps: np.ndarray
    ministeps: np.ndarray
    iterations: np.ndarray
    cuts: np.ndarray
    implicit_fraction: np.ndarray


@dataclass
class TimeIntegrationVariant:
    result: SPE1TutorialResult
    diagnostics: TimeIntegrationDiagnostics


def _run_spe1_variant(mode: str, nsteps: int | None) -> TimeIntegrationVariant:
    deck, model, state0 = setup_spe1_tutorial()
    set_time_discretization(model, mode)
    timesteps = np.asarray(deck["tsteps"], dtype=float)
    if nsteps is not None:
        timesteps = timesteps[:int(nsteps)]

    solver = NonLinearSolver(max_iterations=30, use_line_search=True)
    report = SimulationReport()
    states = []
    elapsed = []
    state = state0.copy()
    total = 0.0
    actual_steps = []
    implicit_fraction = []
    for step_no, dt in enumerate(timesteps, start=1):
        state, step = solver.solve_timestep(model, state, float(dt))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"{mode} SPE1 report step {step_no} failed")
        states.append(state.copy())
        total += float(dt)
        elapsed.append(total)
        actual_steps.extend(step.ministeps or [float(dt)])
        implicit_fraction.append(
            float(np.mean(state["implicit"])) if "implicit" in state else
            (1.0 if mode == "fully-implicit" else 0.0)
        )

    producer, injector = _well_indices(model)
    result = SPE1TutorialResult(
        deck=deck, model=model, initial_state=state0, states=states,
        report=report, history=_collect_history(states),
        time=np.asarray(elapsed), producer_index=producer,
        injector_index=injector,
    )
    diagnostics = TimeIntegrationDiagnostics(
        report_steps=timesteps,
        ministeps=np.asarray(actual_steps),
        iterations=np.asarray([s.iterations for s in report.steps], dtype=int),
        cuts=np.asarray([s.cuts for s in report.steps], dtype=int),
        implicit_fraction=np.asarray(implicit_fraction),
    )
    return TimeIntegrationVariant(result, diagnostics)


def run_blackoil_time_integration_example(
    name: str = "spe1", nsteps: int | None = 10,
) -> dict[str, TimeIntegrationVariant]:
    """Run all three temporal discretizations on the selected benchmark."""
    if name.lower() != "spe1":
        if name.lower() == "spe9":
            raise NotImplementedError(
                "SPE9 time integration requires the pending SPE9 deck, VFP, "
                "and schedule importer"
            )
        raise ValueError(f"unknown black-oil time-integration case: {name!r}")
    return {
        "Fully-Implicit": _run_spe1_variant("fully-implicit", nsteps),
        "Explicit": _run_spe1_variant("explicit", nsteps),
        "Adaptive-Implicit (AIM)": _run_spe1_variant("adaptive-implicit", nsteps),
    }


if __name__ == "__main__":
    variants = run_blackoil_time_integration_example("spe1", 5)
    for name, variant in variants.items():
        d = variant.diagnostics
        print(
            f"{name}: {d.ministeps.size} ministeps, "
            f"{d.iterations.sum()} Newton iterations, "
            f"mean implicit fraction {d.implicit_fraction.mean():.2%}"
        )


__all__ = [
    "TimeIntegrationDiagnostics", "TimeIntegrationVariant",
    "run_blackoil_time_integration_example",
]
