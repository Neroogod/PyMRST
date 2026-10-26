"""Isolated-process entrypoint for running a materialized model-construction revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .simulation_jobs import _state_to_npz


def run_constructed_model(context, request: dict) -> dict:
    """Run the report-step schedule bound to a materialized ``ModelRevision``.

    Unlike the fixed SPE1 benchmark runner, this drives whatever rock, fluid,
    wells and control-period schedule the user defined in the guided
    model-construction builder. The grid itself is rebuilt from the
    ``GridSnapshot``'s recorded provenance rather than re-read from a deck,
    since a snapshot only stores bounded summary arrays for UI display.
    """
    from pymrst.app.database.models import AppDatabase
    from pymrst.app.services.construction_service import ModelConstructionService
    from pymrst.app.services.grid_service import GridInspectionService
    from pymrst.app.services.project_service import ProjectService
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver

    application = dict(request.get("_application") or {})
    workspace_root = str(application.get("workspace_root", "")).strip()
    if not workspace_root:
        raise ValueError("pymrst-construction-run requires a workspace root")
    revision_id = str(request.get("revision_id", "")).strip()
    if not revision_id:
        raise ValueError("pymrst-construction-run requires a revision_id")

    database = AppDatabase(Path(workspace_root) / ".pymrst-app" / "metadata.sqlite")
    try:
        projects = ProjectService(database, workspace_root)
        grids = GridInspectionService(database, projects)
        construction = ModelConstructionService(database, projects, grids)
        _revision, snapshot_id, _document = construction.revision_document(revision_id)
        context.progress(0.02, "Rebuilding grid from recorded provenance")
        grid = grids.rebuild(snapshot_id)
        context.raise_if_cancelled()
        context.progress(0.05, "Compiling rock, fluid, wells and schedule")
        compiled = construction.compile(revision_id, grid)
    finally:
        database.close()

    max_newton_iterations = int(request.get("max_newton_iterations", 30))
    max_timestep_cuts = int(request.get("max_timestep_cuts", 6))
    use_line_search = bool(request.get("use_line_search", True))
    save_states = bool(request.get("save_states", True))

    model = GenericBlackOilModelAD(
        compiled.grid, compiled.rock, compiled.fluid, list(compiled.wells),
    )
    # The document only carries reservoir-cell fields (pressure, saturations);
    # the model's Newton loop also needs a starting guess for every well's
    # surface rates and bhp, matching TwoPhaseOilWaterModelAD.initial_state's
    # convention: bhp defaults to the field's mean pressure for anything but
    # a bhp-controlled well, and rate-controlled wells seed q from val*compi.
    initial_state = dict(compiled.initial_state)
    phase_names = ["qWs", "qOs", "qGs"][:int(compiled.fluid.n_phases_declared)]
    reference_pressure = float(np.mean(initial_state["pressure"]))
    for index, name in enumerate(phase_names):
        initial_state[name] = np.array([
            well.val*well.compi[index] if well.type == "rate" else 0.0
            for well in compiled.wells
        ])
    initial_state["bhp"] = np.array([
        well.val if well.type == "bhp" else reference_pressure
        for well in compiled.wells
    ])
    from pymrst.modules.ad_core.statefunctions.state_function import State
    initial_state = State(initial_state)
    solver = NonLinearSolver(
        max_iterations=max_newton_iterations, max_timestep_cuts=max_timestep_cuts,
        use_line_search=use_line_search, verbose=False, progress="off",
    )
    schedule = compiled.schedule
    total_steps = schedule.n_steps
    context.emit(
        "simulation_initialized", cells=int(compiled.grid.cells.num),
        wells=len(compiled.wells), well_names=[str(well.name) for well in compiled.wells],
        report_steps=total_steps, control_periods=len(schedule.controls),
    )
    total_iterations = total_cuts = 0
    elapsed = 0.0
    last_state = initial_state
    for offset, (state, report) in enumerate(solver.iter_schedule(model, initial_state, schedule)):
        context.raise_if_cancelled()
        dt = float(schedule.step_val[offset])
        total_iterations += int(report.iterations); total_cuts += int(report.cuts)
        elapsed += dt
        last_state = state
        residuals = dict(report.residuals[-1]) if report.residuals else {}
        context.emit(
            "convergence", report_step=offset + 1, newton_iterations=int(report.iterations),
            timestep_cuts=int(report.cuts), line_searches=int(getattr(report, "line_searches", 0)),
            residuals=residuals, wall_seconds=float(getattr(report, "wall_seconds", 0.0)),
            control_index=int(schedule.step_control[offset]),
        )
        if not report.converged:
            raise RuntimeError(
                f"report step {offset + 1}/{total_steps} failed after {report.iterations} iterations"
            )
        if save_states:
            state_path = context.run_directory / "states" / f"report_{offset + 1:04d}.npz"
            _state_to_npz(state_path, state)
            context.emit("state_saved", report_step=offset + 1,
                         relative_path=state_path.relative_to(context.run_directory).as_posix())
        context.progress(
            0.05 + 0.95*(offset + 1)/max(total_steps, 1),
            f"Report step {offset + 1}/{total_steps} accepted",
            report_step=offset + 1, report_steps=total_steps, simulated_time_seconds=elapsed,
            newton_iterations=int(report.iterations),
        )
    summary = {
        "revision_id": revision_id, "report_steps": total_steps,
        "simulated_time_seconds": elapsed, "newton_iterations": total_iterations,
        "timestep_cuts": total_cuts,
        "final_pressure_min": float(np.min(last_state["pressure"])),
        "final_pressure_max": float(np.max(last_state["pressure"])),
        "final_bhp": np.asarray(last_state["bhp"], dtype=float).tolist(),
    }
    (context.run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


__all__ = ["run_constructed_model"]
