"""Repository-owned member runners used by executable experiment definitions."""

from __future__ import annotations

import time
import numpy as np

from .execution import MemberExecutionContext, MemberPayload


def spe1_time_integration_member(context: MemberExecutionContext) -> MemberPayload:
    """Run one actual SPE1 time-integration member from its component coordinate."""
    from pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example import (
        _run_spe1_variant,
    )
    selected = context.coordinate.components.get("time_integration")
    aliases = {"fim": "fully-implicit", "fully-implicit": "fully-implicit",
               "explicit": "explicit", "aim": "adaptive-implicit",
               "adaptive-implicit": "adaptive-implicit"}
    if selected not in aliases:
        raise ValueError(f"unsupported SPE1 time integration component: {selected!r}")
    started = time.perf_counter()
    variant = _run_spe1_variant(aliases[selected], 2)
    history, diagnostics = variant.result.history, variant.diagnostics
    pressure = np.asarray(history["pressure"], float)
    return MemberPayload(
        times=np.asarray(variant.result.time, float),
        fields={name: np.asarray(history[name], float)
                for name in ("pressure", "sW", "sG", "rs")},
        wells={name: np.asarray(history[name], float)
               for name in ("bhp", "qWs", "qOs", "qGs")},
        regions={"field_average_pressure": np.mean(pressure, axis=1)[:, None]},
        convergence={"iterations": diagnostics.iterations[:, None],
                     "cuts": diagnostics.cuts[:, None],
                     "implicit_fraction": diagnostics.implicit_fraction[:, None]},
        conservation={"finite_state": np.ones((pressure.shape[0], 1))},
        performance={"simulation_seconds": time.perf_counter() - started,
                     "total_iterations": float(np.sum(diagnostics.iterations))},
        metadata={"case": "spe1", "time_integration": aliases[selected]},
    )


def deterministic_manufactured_member(context: MemberExecutionContext) -> MemberPayload:
    """Small deterministic runner for failure/resume and process-isolation tests."""
    parameters = context.coordinate.parameters
    if parameters.get("test.fail", False):
        raise RuntimeError("requested manufactured member failure")
    multiplier = float(parameters.get("test.multiplier", 1.0))
    rng = np.random.Generator(np.random.PCG64(context.seed))
    times = np.array([1.0, 2.0, 4.0])
    pressure = multiplier*(1.0e7 + np.arange(4)[None, :] + times[:, None])
    saturation = np.broadcast_to(np.linspace(0.1, 0.4, 4), pressure.shape).copy()
    noise = rng.random(3)*1e-12
    return MemberPayload(
        times=times, fields={"pressure": pressure, "saturation": saturation},
        wells={"producer_rate": multiplier*np.array([[1.0], [0.9], [0.8]])},
        regions={"field_pressure": np.mean(pressure, axis=1)[:, None]},
        convergence={"iterations": np.array([[3], [4], [3]])},
        conservation={"mass_balance": noise[:, None]},
        performance={"runtime_seconds": 0.01*multiplier,
                     "peak_memory_bytes": 1024.0},
        metadata={"manufactured": True},
    )


def checkpoint_resume_member(context: MemberExecutionContext) -> MemberPayload:
    """Fail the first attempt and complete only when its failed checkpoint returns."""
    if context.checkpoint is None or context.checkpoint.get("status") != "failed":
        raise RuntimeError("intentional first-attempt failure")
    times = np.array([1.0, 2.0])
    return MemberPayload(
        times=times, fields={"pressure": np.array([[1.0], [2.0]])},
        performance={"resumed_from_checkpoint": 1.0},
        metadata={"previous_error": context.checkpoint.get("error_type", "")})


def deterministic_branch_member(context) -> MemberPayload:
    """Advance a stored manufactured pressure state under changed controls."""
    pressure = np.asarray(context.parent_fields["pressure"], dtype=float)
    rate = float(context.definition.controls.get("rate", 1.0))
    multiplier = float(context.definition.properties.get("mobility", 1.0))
    increments = np.array([1.0, 2.0])[:, None]*rate*multiplier
    times = context.parent_time + np.array([1.0, 2.0])
    return MemberPayload(
        times=times, fields={"pressure": pressure[None, :]+increments},
        wells={"branch_rate": np.full((2, 1), rate)},
        performance={"branch_steps": 2.0},
        metadata={"runner": "deterministic_branch_member"})


def spe1_branch_member(context) -> MemberPayload:
    """Continue one actual SPE1 report step from a stored branch checkpoint."""
    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
        _collect_history, setup_spe1_tutorial,
    )
    from pymrst.modules.ad_core import NonLinearSolver
    from pymrst.modules.ad_core.utils import set_time_discretization

    required = {"pressure", "sW", "sG", "rs"}
    missing = required-set(context.parent_fields)
    if missing:
        raise ValueError(f"SPE1 branch checkpoint is missing {sorted(missing)}")
    deck, model, state = setup_spe1_tutorial()
    mode = context.definition.components.get("time_integration", "fully-implicit")
    set_time_discretization(model, mode)
    pressure = np.asarray(context.parent_fields["pressure"], dtype=float)
    sw = np.asarray(context.parent_fields["sW"], dtype=float)
    sg = np.asarray(context.parent_fields["sG"], dtype=float)
    state["pressure"] = pressure.copy()
    state["s"] = np.column_stack((sw, 1.0-sw-sg, sg))
    state["rs"] = np.asarray(context.parent_fields["rs"], dtype=float).copy()
    cumulative = np.cumsum(np.asarray(deck["tsteps"], dtype=float))
    next_index = int(np.searchsorted(cumulative, context.parent_time, side="right"))
    if next_index >= cumulative.size:
        raise ValueError("SPE1 schedule has no report step after the branch point")
    dt = float(deck["tsteps"][next_index])
    started = time.perf_counter()
    state, report = NonLinearSolver(max_iterations=30, use_line_search=True).solve_timestep(
        model, state, dt)
    if not report.converged:
        raise RuntimeError(f"SPE1 {mode} branch failed to converge")
    history = _collect_history([state])
    return MemberPayload(
        times=np.array([context.parent_time+dt]),
        fields={name: np.asarray(history[name], float)
                for name in ("pressure", "sW", "sG", "rs")},
        wells={name: np.asarray(history[name], float)
               for name in ("bhp", "qWs", "qOs", "qGs")},
        regions={"field_average_pressure": np.mean(
            np.asarray(history["pressure"], float), axis=1)[:, None]},
        convergence={"iterations": np.array([[report.iterations]]),
                     "cuts": np.array([[report.cuts]])},
        conservation={"finite_state": np.ones((1, 1))},
        performance={"simulation_seconds": time.perf_counter()-started,
                     "total_iterations": float(report.iterations)},
        metadata={"case": "spe1", "time_integration": mode,
                  "continued_from_time": context.parent_time})


__all__ = ["checkpoint_resume_member", "deterministic_branch_member",
           "deterministic_manufactured_member",
           "spe1_branch_member", "spe1_time_integration_member"]
