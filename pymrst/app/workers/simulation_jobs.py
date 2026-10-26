"""Trusted PyMRST numerical adapters for the isolated application worker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _state_to_npz(path: Path, state) -> None:
    values = {}
    for name, value in state.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise TypeError(f"state field {name} is not a numeric array")
        values[str(name)] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _state_from_npz(path: Path):
    from pymrst.modules.ad_core.statefunctions.state_function import State
    with np.load(path, allow_pickle=False) as values:
        return State({name: values[name].copy() for name in values.files})


def run_spe1(context, request):
    """Run the canonical AD black-oil SPE1 loop with report-boundary evidence."""
    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
        setup_spe1_tutorial,
    )
    from pymrst.modules.ad_core import NonLinearSolver

    application = dict(request.get("_application") or {})
    deck_path = application.get("source_path")
    if not deck_path:
        raise ValueError("pymrst-spe1 requires an inspected deck model")
    requested_steps = int(request.get("report_steps", 30))
    if not 1 <= requested_steps <= 10_000:
        raise ValueError("report_steps must lie in [1,10000]")
    checkpoint_interval = int(request.get("checkpoint_interval", 0))
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval cannot be negative")
    save_states = bool(request.get("save_states", True))
    deck, model, initial_state = setup_spe1_tutorial(deck_path)
    timesteps = np.asarray(deck["tsteps"], dtype=float)[:requested_steps]
    if timesteps.size != requested_steps:
        raise ValueError(
            f"deck contains only {timesteps.size} report steps, requested {requested_steps}"
        )
    resume = dict(request.get("_resume") or {})
    start = int(dict(resume.get("metadata") or {}).get("completed_steps", 0))
    if not 0 <= start < requested_steps:
        raise ValueError("checkpoint completed-step index is incompatible with schedule")
    if resume:
        checkpoint_path = Path(str(resume["path"])).resolve()
        state = _state_from_npz(checkpoint_path)
    else:
        state = initial_state.copy()
    solver = NonLinearSolver(
        max_iterations=int(request.get("max_newton_iterations", 30)),
        max_timestep_cuts=int(request.get("max_timestep_cuts", 6)),
        use_line_search=bool(request.get("use_line_search", True)),
        verbose=False,
        progress="off",
    )
    elapsed = float(timesteps[:start].sum())
    resume_metadata = dict(resume.get("metadata") or {})
    total_iterations = int(resume_metadata.get("cumulative_newton_iterations", 0))
    total_cuts = int(resume_metadata.get("cumulative_timestep_cuts", 0))
    context.emit(
        "simulation_initialized", cells=int(model.G.cells.num), wells=len(model.wells),
        well_names=[str(getattr(well, "name", f"WELL_{index}"))
                    for index, well in enumerate(model.wells)],
        report_steps=requested_steps, resumed_from_step=start,
    )
    for offset in range(start, requested_steps):
        context.raise_if_cancelled()
        dt = float(timesteps[offset])
        context.emit("report_step_started", report_step=offset + 1, dt_seconds=dt,
                     simulated_time_seconds=elapsed)
        state, report = solver.solve_timestep(model, state, dt)
        if not report.converged:
            raise RuntimeError(
                f"SPE1 report step {offset + 1} failed after {report.iterations} iterations"
            )
        total_iterations += int(report.iterations); total_cuts += int(report.cuts)
        elapsed += dt
        residuals = dict(report.residuals[-1]) if report.residuals else {}
        context.emit(
            "convergence", report_step=offset + 1, newton_iterations=int(report.iterations),
            timestep_cuts=int(report.cuts), line_searches=int(report.line_searches),
            residuals=residuals, wall_seconds=float(report.wall_seconds),
            assembly_seconds=float(report.assembly_seconds),
            linear_iterations=int(report.total_linear_iterations),
        )
        state_path = context.run_directory / "states" / f"report_{offset + 1:04d}.npz"
        if save_states or (checkpoint_interval and (offset + 1) % checkpoint_interval == 0):
            _state_to_npz(state_path, state)
            if save_states:
                context.emit("state_saved", report_step=offset + 1,
                             relative_path=state_path.relative_to(context.run_directory).as_posix())
        if checkpoint_interval and (offset + 1) % checkpoint_interval == 0:
            context.checkpoint(
                state_path.relative_to(context.run_directory).as_posix(),
                completed_steps=offset + 1, report_steps=requested_steps,
                simulated_time_seconds=elapsed, source_hash=application.get("source_hash", ""),
                cumulative_newton_iterations=total_iterations,
                cumulative_timestep_cuts=total_cuts,
            )
        context.progress(
            (offset + 1) / requested_steps,
            f"Report step {offset + 1}/{requested_steps} accepted",
            report_step=offset + 1, report_steps=requested_steps,
            simulated_time_seconds=elapsed, newton_iterations=int(report.iterations),
        )
    summary = {
        "report_steps": requested_steps, "resumed_from_step": start,
        "simulated_time_seconds": elapsed, "newton_iterations": total_iterations,
        "timestep_cuts": total_cuts,
        "final_pressure_min": float(np.min(state["pressure"])),
        "final_pressure_max": float(np.max(state["pressure"])),
        "final_bhp": np.asarray(state["bhp"], dtype=float).tolist(),
    }
    (context.run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


__all__ = ["run_spe1"]
