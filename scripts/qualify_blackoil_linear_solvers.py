#!/usr/bin/env python3
"""Qualify direct, GMRES+ILU, and CPR linear solvers on SPE1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.io

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    _collect_history, setup_spe1_tutorial,
)
from pymrst.modules.ad_core import NonLinearSolver
from pymrst.modules.ad_core.solvers.concrete_solvers import (
    BackslashSolverAD, GMRES_ILUSolverADAD,
)
from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD
from pymrst.research_studio.validation import (
    EvidenceEdge, EvidenceGraph, EvidenceNode, EvidenceNodeKind, Independence,
    TolerancePolicy, edge_from_comparison,
)
try:
    from scripts.qualify_blackoil_time_integration import (
        STRICT_MRST_TOLERANCES, _compare, _reference_arrays,
    )
except ModuleNotFoundError:  # Direct ``python3 scripts/this_file.py`` execution.
    from qualify_blackoil_time_integration import (
        STRICT_MRST_TOLERANCES, _compare, _reference_arrays,
    )


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples/validation/ref_time_integration_spe1.mat"
DEFAULT_OUTPUT = ROOT / "docs/research_studio/qualification/blackoil_linear_solver_evidence_graph.json"
CROSS_TOLERANCES = {
    "pressure": 0.01, "saturation": 1e-10, "solution_gas_ratio": 1e-7,
    "well_bhp": 0.05, "well_water_rate": 1e-16,
    "well_oil_rate": 1e-16, "well_gas_rate": 1e-12,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(history):
    return {
        "pressure": history["pressure"],
        "saturation": np.stack((history["sW"], history["sG"]), axis=-1),
        "solution_gas_ratio": history["rs"], "well_bhp": history["bhp"],
        "well_water_rate": history["qWs"], "well_oil_rate": history["qOs"],
        "well_gas_rate": history["qGs"],
    }


def _run(linear_solver, nsteps: int):
    deck, model, state = setup_spe1_tutorial()
    reports, states = [], []
    solver = NonLinearSolver(
        max_iterations=30, use_line_search=True, linear_solver=linear_solver,
    )
    started = time.perf_counter()
    for dt in np.asarray(deck["tsteps"], dtype=float)[:nsteps]:
        state, report = solver.solve_timestep(model, state, float(dt))
        if not report.converged:
            raise RuntimeError(f"{type(linear_solver).__name__} failed SPE1")
        states.append(state.copy())
        reports.append(report)
    return _arrays(_collect_history(states)), reports, time.perf_counter() - started


def _diagnostics(reports):
    solves = [solve for report in reports for solve in report.linear_solves]
    return {
        "nonlinear_iterations": [report.iterations for report in reports],
        "linear_iterations_per_newton": [solve.iterations for solve in solves],
        "total_linear_iterations": sum(solve.iterations for solve in solves),
        "maximum_true_relative_residual": max(solve.relative_residual for solve in solves),
        "linear_solve_seconds": sum(solve.wall_seconds for solve in solves),
        "solver": solves[0].solver,
    }


def qualify(reference: Path = REFERENCE):
    reference = Path(reference)
    ref = scipy.io.loadmat(reference, squeeze_me=True, struct_as_record=False)
    times = np.cumsum(np.asarray(ref["tsteps"], dtype=float).reshape(-1))
    solvers = {
        "direct": BackslashSolverAD(),
        "gmres_ilu": GMRES_ILUSolverADAD(
            tolerance=1e-10, max_iterations=200, restart=40,
        ),
        "cpr": CPRSolverADAD(tolerance=1e-8, max_iterations=200),
    }
    runs = {key: _run(solver, times.size) for key, solver in solvers.items()}

    graph = EvidenceGraph()
    digest = _sha256(reference)
    graph.add_node(EvidenceNode(
        "mrst_direct", EvidenceNodeKind.MRST, "MRST SPE1 sparse direct",
        version="MRST 2024a", metadata={
            "reference": str(reference.relative_to(ROOT)), "sha256": digest,
            "generator": "examples/validation/dump_time_integration_spe1_reference.m",
        },
    ))
    for key, label in (("direct", "sparse direct"), ("gmres_ilu", "GMRES + ILU"),
                       ("cpr", "CPR + GMRES")):
        graph.add_node(EvidenceNode(
            f"pymrst_{key}", EvidenceNodeKind.PYMRST, f"PyMRST SPE1 {label}",
            version="working-tree", metadata={"runtime_seconds": runs[key][2]},
        ))

    direct = runs["direct"][0]
    mrst = _compare(
        _reference_arrays(ref["fim"]), direct, times, STRICT_MRST_TOLERANCES,
        reference_wells=("injector", "producer"),
        candidate_wells=("producer", "injector"),
    )
    for name, comparison in mrst.items():
        graph.add_edge(edge_from_comparison(
            f"direct_mrst_{name}", "mrst_direct", "pymrst_direct",
            f"Direct-solver {name.replace('_', ' ')} agrees with MRST",
            comparison, independence=Independence.PARTIAL,
            mappings=("report_time:exact", "well_name:v1", "cell_identity:v1"),
            shared_kernels=("PyMRST is a Python translation of the MRST formulation",),
        ))

    cross = {}
    shared = ("assembled Jacobian", "nonlinear solver", "grid", "fluid", "wells")
    for key in ("gmres_ilu", "cpr"):
        comparisons = _compare(
            direct, runs[key][0], times, CROSS_TOLERANCES,
            reference_wells=("producer", "injector"),
            candidate_wells=("producer", "injector"),
        )
        cross[key] = comparisons
        for name, comparison in comparisons.items():
            graph.add_edge(edge_from_comparison(
                f"direct_vs_{key}_{name}", "pymrst_direct", f"pymrst_{key}",
                f"Direct and {key.replace('_', ' ')} {name.replace('_', ' ')} agree",
                comparison, independence=Independence.SHARED_KERNEL,
                mappings=("report_time:exact", "well_name:v1", "cell_identity:v1"),
                shared_kernels=shared,
            ))

    diagnostics = {key: {**_diagnostics(reports), "runtime_seconds": runtime}
                   for key, (_, reports, runtime) in runs.items()}
    metrics = {}
    for key, values in diagnostics.items():
        metrics[f"{key}_total_linear_iterations"] = values["total_linear_iterations"]
        metrics[f"{key}_maximum_true_relative_residual"] = values["maximum_true_relative_residual"]
        metrics[f"{key}_runtime_seconds"] = values["runtime_seconds"]
        metrics[f"{key}_linear_solve_seconds"] = values["linear_solve_seconds"]
    graph.add_edge(EvidenceEdge(
        "linear_solver_work", "pymrst_direct", "pymrst_cpr", "solver_work",
        "Linear work and true residuals are reported without equating factorization and Krylov iterations",
        Independence.SHARED_KERNEL, shared_kernels=shared, metrics=metrics,
    ))
    numerical = [*mrst.values(), *(v for group in cross.values() for v in group.values())]
    summary = {
        "case": "spe1_blackoil_direct_gmres_ilu_cpr_two_steps",
        "passed": all(value.passed for value in numerical),
        "reference_sha256": digest, "diagnostics": diagnostics,
        "direct_vs_mrst": {k: v.to_dict() for k, v in mrst.items()},
        "cross_solver": {k: {n: v.to_dict() for n, v in values.items()}
                         for k, values in cross.items()},
    }
    return graph, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph, summary = qualify(args.reference)
    payload = {"artifact": "pymrst_blackoil_linear_solver_evidence_graph",
               "artifact_schema_version": 1, "graph": graph.to_dict(), "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {len(graph.edges)} evidence edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
