"""Complete MRST ``runAllNorneExamples.m`` case-matrix driver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np

from pymrst.modules.ad_mechanics.examples.run_norne_example import run_norne_example


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "val"):
        return np.asarray(value.val).tolist()
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray()).tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    # Reports occasionally retain a lightweight solver/model object.  Keep
    # the acceptance artifact writable without silently dropping the field;
    # its qualified type and representation make the fallback explicit.
    return {"python_type": f"{type(value).__module__}.{type(value).__name__}",
            "repr": repr(value)}


@dataclass(frozen=True)
class NorneCaseSpec:
    fluid_model: str
    method: str
    splitting_tolerance: float | None = None

    @property
    def name(self) -> str:
        return f"{self.fluid_model} / {self.method}"


@dataclass
class NorneCaseRun:
    spec: NorneCaseSpec
    result: Any = None
    elapsed_seconds: float = 0.0
    converged: bool = False
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class NorneSweepResult:
    norne_case: str
    bc_case: str
    runs: list[NorneCaseRun] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return bool(self.runs) and all(run.converged for run in self.runs)

    def summary(self) -> list[dict[str, Any]]:
        return [{
            "fluid_model": run.spec.fluid_model,
            "method": run.spec.method,
            "converged": run.converged,
            "elapsed_seconds": run.elapsed_seconds,
            "error": run.error,
            "splitting_tolerance": run.spec.splitting_tolerance,
            **run.metrics,
        } for run in self.runs]


def _result_metrics(result: Any) -> dict[str, Any]:
    """Extract compact engineering outputs from a Norne runner result."""
    if not (isinstance(result, tuple) and len(result) == 3):
        return {}
    grid, model, states = result
    if not states:
        return {"cells": int(grid.cells.num), "report_steps": 0}
    state = states[-1]
    pressure = np.asarray(state["pressure"], dtype=float)
    physical_states = [item for item in states if "pressure" in item]
    metrics: dict[str, Any] = {
        "cells": int(grid.cells.num),
        "report_steps": max(len(states) - 1, 0),
        "pressure_mean_pa": float(np.mean(pressure)),
        "pressure_max_pa": float(np.max(pressure)),
        "displacement_inf_m": float(np.max(np.abs(state["u"]))),
        "bhp_pa": np.asarray(state.get("bhp", []), dtype=float).tolist(),
        "pressure_mean_history_pa": [
            float(np.mean(np.asarray(item["pressure"], dtype=float))) for item in physical_states
        ],
        "pressure_max_history_pa": [
            float(np.max(np.asarray(item["pressure"], dtype=float))) for item in physical_states
        ],
        "bhp_history_pa": [
            np.asarray(item.get("bhp", []), dtype=float).tolist() for item in physical_states
        ],
    }
    if hasattr(grid, "nodes") and hasattr(model, "rock"):
        fingerprint = hashlib.sha256()
        for values in (grid.nodes.coords, grid.cells.volumes, model.rock.poro, model.rock.perm):
            fingerprint.update(np.ascontiguousarray(values).view(np.uint8))
        metrics["grid_rock_sha256"] = fingerprint.hexdigest()
    for state_name, metric_name in (("sW", "water"), ("sG", "gas")):
        if state_name in state:
            saturation = np.asarray(state[state_name], dtype=float)
            metrics[f"{metric_name}_saturation_mean"] = float(np.mean(saturation))
            metrics[f"{metric_name}_saturation_max"] = float(np.max(saturation))
    for state_name, metric_name in (("qWs", "water"), ("qOs", "oil"), ("qGs", "gas")):
        if state_name in state:
            metrics[f"{metric_name}_surface_rates_m3_s"] = np.asarray(
                state[state_name], dtype=float
            ).tolist()
            metrics[f"{metric_name}_surface_rate_history_m3_s"] = [
                np.asarray(item.get(state_name, []), dtype=float).tolist()
                for item in physical_states
            ]
    reports = list(getattr(model, "acceptance_reports", []))
    if reports:
        metrics["timestep_reports"] = reports
        metrics["timestep_iterations"] = [int(item.get("Iterations", 0)) for item in reports]
        metrics["timestep_cuts"] = [int(item.get("Cuts", 0)) for item in reports]
        metrics["all_timesteps_converged"] = all(bool(item.get("Converged", False)) for item in reports)
        metrics["timesteps_seconds"] = np.asarray(
            getattr(model, "acceptance_timesteps", []), dtype=float
        ).tolist()
    split_report = getattr(model, "last_split_report", None)
    if split_report:
        metrics["splitting_iterations_last_step"] = int(
            split_report.get("iterations", 0)
        )
    return metrics


NORNE_CASE_MATRIX = (
    NorneCaseSpec("water", "fully coupled"),
    NorneCaseSpec("water", "fixed stress splitting"),
    NorneCaseSpec("oil water", "fully coupled"),
    NorneCaseSpec("oil water", "fixed stress splitting"),
    NorneCaseSpec("blackoil", "fully coupled"),
    NorneCaseSpec("blackoil", "fixed stress splitting", 1.0e-3),
)


def run_all_norne_examples(
    include_full: bool = False,
    *,
    norne_case: str | None = None,
    bc_case: str = "bottom fixed",
    n_steps: int = 20,
    plot: bool = False,
    continue_on_error: bool = False,
    cases: Iterable[NorneCaseSpec] | None = None,
    runner: Callable[..., Any] | None = None,
) -> NorneSweepResult | list[NorneSweepResult]:
    """Run MRST's six fluid/coupling combinations.

    ``include_full=True`` retains the historical wrapper behavior of running
    both mini and full grids. A specific ``norne_case`` selects one grid.
    """
    runner = runner or run_norne_example
    selected_cases = tuple(cases or NORNE_CASE_MATRIX)
    grids = ([norne_case] if norne_case is not None
             else (["mini Norne", "full"] if include_full else ["mini Norne"]))
    sweeps: list[NorneSweepResult] = []

    for grid_case in grids:
        sweep = NorneSweepResult(str(grid_case), bc_case)
        for spec in selected_cases:
            started = time.perf_counter()
            run = NorneCaseRun(spec=spec)
            options = {
                "fluid_model": spec.fluid_model,
                "method": spec.method,
                "bc_case": bc_case,
                "n_steps": n_steps,
                "plot": plot,
            }
            if spec.splitting_tolerance is not None:
                options["splitting_tolerance"] = spec.splitting_tolerance
            try:
                run.result = runner(str(grid_case), **options)
                run.metrics = _result_metrics(run.result)
                run.converged = True
            except Exception as error:
                run.error = f"{type(error).__name__}: {error}"
                print(f"\nFAILED {grid_case} / {spec.name}: {run.error}")
                if not continue_on_error:
                    raise
            finally:
                run.elapsed_seconds = time.perf_counter() - started
                sweep.runs.append(run)
        sweeps.append(sweep)
    return sweeps if len(sweeps) > 1 else sweeps[0]


def _select_cases(fluid: str | None, method: str | None) -> tuple[NorneCaseSpec, ...]:
    return tuple(
        case for case in NORNE_CASE_MATRIX
        if (fluid is None or case.fluid_model == fluid)
        and (method is None or case.method == method)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run PyMRST's MRST-compatible Norne mechanics matrix"
    )
    parser.add_argument("--grid", choices=("mini Norne", "full"), default="mini Norne")
    parser.add_argument("--fluid", choices=("water", "oil water", "blackoil"))
    parser.add_argument("--method", choices=("fully coupled", "fixed stress splitting"))
    parser.add_argument("--steps", type=int, default=20,
                        help="five-day steps after the initial one-day step")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)
    cases = _select_cases(args.fluid, args.method)
    sweep = run_all_norne_examples(
        norne_case=args.grid, n_steps=args.steps, plot=args.plot,
        continue_on_error=args.continue_on_error, cases=cases,
    )
    payload = {
        "norne_case": sweep.norne_case,
        "bc_case": sweep.bc_case,
        "converged": sweep.converged,
        "runs": sweep.summary(),
    }
    rendered = json.dumps(payload, indent=2, default=_json_default)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.json_path}")
    else:
        print(rendered)
    return 0 if sweep.converged else 1


__all__ = [
    "NORNE_CASE_MATRIX",
    "NorneCaseRun",
    "NorneCaseSpec",
    "NorneSweepResult",
    "run_all_norne_examples",
]


if __name__ == "__main__":
    raise SystemExit(main())
