"""Small schedule helpers for the ad-blackoil example ports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from pymrst.modules.ad_core import NonLinearSolver, SimulationReport, State
from pymrst.utils.format_time_range import format_time_range


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
    verbose: bool = False,
) -> ExampleResult:
    """Run a simple MRST-like schedule with fixed model driving forces.

    `controls` can set `bc`, `src`, and `wells`/`W` on models that expose those
    attributes. This mirrors MRST's `simpleSchedule` for the example ports.

    `verbose` prints one line per accepted step, MRST-style. It defaults to
    False because several of these example scripts also run headless inside
    the desktop app's isolated worker, whose stdout is a JSON-lines protocol
    channel - a stray print there breaks the worker, not just the console.
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
    dts = np.asarray(timesteps, dtype=float).ravel()
    n_steps = dts.size
    n_digits = len(str(n_steps))
    cumulative = np.concatenate(([0.0], np.cumsum(dts)))
    # MRST's simulateScheduleAD pads the "from" column to the width of the
    # run's final cumulative time, so the "->" tokens line up down the column.
    column_width = len(format_time_range(cumulative[-1], limit=2)) if verbose else 0
    for index, dt in enumerate(dts):
        st, step = solver.solve_timestep(model, st, float(dt))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"schedule step failed: {step}")
        states.append(st.copy())
        if verbose:
            start = format_time_range(cumulative[index], limit=2)
            end = format_time_range(cumulative[index + 1], limit=2)
            print(
                f"Solving timestep {index + 1:0{n_digits}d}/{n_steps:0{n_digits}d}: "
                f"{start:<{column_width}} -> {end}"
            )
    return ExampleResult(states, report, collect_history(states, history_keys))


def plot_browser(
    model,
    result: ExampleResult,
    timesteps,
    *,
    wells=None,
    filename: str | Path = "pymrst_reservoir_browser.html",
    title: str = "PyMRST reservoir browser",
    open_browser: bool = False,
    **kwargs,
) -> Path:
    """Write the same interactive 3D browser viewer used for the SPE1 tutorial.

    Takes a model and a `run_schedule` result directly - no separate save/load
    step - since `plot_reservoir_browser` already reads a list of `State`
    objects. `model.G`/`model.g` covers both this package's model families;
    `wells` defaults to whatever the model was actually run with.
    """
    from pymrst.plotting.extensions.browser import plot_reservoir_browser

    grid = getattr(model, "G", None)
    if grid is None:
        grid = getattr(model, "g", None)
    if grid is None:
        raise AttributeError(f"{type(model).__name__} exposes neither .G nor .g")
    if wells is None:
        wells = getattr(model, "wells", None)
    times = np.cumsum(np.asarray(timesteps, dtype=float).ravel())
    return plot_reservoir_browser(
        grid, result.states, wells=wells, times=times,
        filename=filename, title=title, open_browser=open_browser, **kwargs,
    )


__all__ = ["ExampleResult", "collect_history", "plot_browser", "run_schedule"]
