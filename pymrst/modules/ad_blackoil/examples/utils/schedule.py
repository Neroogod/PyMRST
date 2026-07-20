"""Small schedule helpers for the ad-blackoil example ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from pymrst.modules.ad_core import NonLinearSolver, SimulationReport, State


@dataclass
class ExampleResult:
    states: list[State]
    report: SimulationReport
    history: dict[str, np.ndarray]


def collect_history(states: Iterable[State], keys: Iterable[str]) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for st in states:
        for key in keys:
            if key in st:
                out[key].append(np.asarray(st[key], dtype=float).copy())
    return {k: np.vstack(v) if v else np.zeros((0, 0)) for k, v in out.items()}


def run_schedule(
    model,
    state0: State,
    timesteps,
    *,
    controls: Mapping | None = None,
    history_keys=("pressure",),
    solver: NonLinearSolver | None = None,
) -> ExampleResult:
    """Run a simple MRST-like schedule with fixed model driving forces.

    `controls` can set `bc`, `src`, and `wells`/`W` on models that expose those
    attributes. This mirrors MRST's `simpleSchedule` for the example ports.
    """
    if controls:
        if "bc" in controls and hasattr(model, "bc"):
            model.bc = controls["bc"]
        if "src" in controls and hasattr(model, "src"):
            model.src = controls["src"]
        wells = controls.get("wells", controls.get("W"))
        if wells is not None and hasattr(model, "facility"):
            model.facility.wells = list(wells)

    solver = solver or NonLinearSolver(max_iterations=25)
    report = SimulationReport()
    states: list[State] = []
    st = state0.copy()
    for dt in np.asarray(timesteps, dtype=float).ravel():
        st, step = solver.solve_timestep(model, st, float(dt))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"schedule step failed: {step}")
        states.append(st.copy())
    return ExampleResult(states, report, collect_history(states, history_keys))


__all__ = ["ExampleResult", "collect_history", "run_schedule"]
