#!/usr/bin/env python3
"""Qualify Phase 64 fixed-stress and fully coupled formulations as a matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.io

from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case
from pymrst.research_studio.validation import (
    ComparisonSeries, EvidenceEdge, EvidenceGraph, EvidenceNode,
    EvidenceNodeKind, Independence, NamedAxis, NamedFieldSeries,
    TolerancePolicy, compare_named_fields, compare_series,
    edge_from_comparison,
)
try:
    from scripts.qualify_research_studio_phase64 import qualify as qualify_fixed_mrst
except ModuleNotFoundError:  # Direct ``python3 scripts/this_file.py`` execution.
    from qualify_research_studio_phase64 import qualify as qualify_fixed_mrst


ROOT = Path(__file__).resolve().parents[1]
FIXED_REFERENCE = ROOT/"examples/validation/ref_p64_run2d_water_fixedstress.mat"
COUPLED_REFERENCE = ROOT/"examples/validation/ref_p63_run2d_water.mat"
DEFAULT_OUTPUT = ROOT/"docs/research_studio/qualification/phase64_cross_formulation_evidence_graph.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(method: str, reference: dict):
    nx, ny = np.asarray(reference["cartDim"], dtype=int).reshape(-1)
    dt = np.asarray(reference["dt"], dtype=float).reshape(-1)
    started = time.perf_counter()
    result = run_2d_case(
        fluid_model="water", method=method, nx=int(nx), ny=int(ny),
        physdims=tuple(np.asarray(reference["L"], dtype=float).reshape(-1)),
        nsteps=dt.size, dt=dt, p_ref=float(reference["pRef"]),
    )
    runtime = time.perf_counter()-started
    if not all(report["converged"] for report in result.reports):
        raise RuntimeError(f"{method} qualification did not converge")
    return result, np.cumsum(dt), runtime


def _arrays(result):
    states, diagnostics = result.states[1:], result.diagnostics[1:]
    return {
        "pressure": np.vstack([state["pressure"] for state in states]),
        "free_displacement": np.vstack([state["xd"] for state in states]),
        "full_displacement": np.vstack([state["u"] for state in states]),
        "volumetric_strain": np.vstack([item["volumetric_strain"] for item in diagnostics]),
        "strain": np.stack([item["strain"] for item in diagnostics]),
        "effective_stress": np.stack([item["effective_stress"] for item in diagnostics]),
        "well_water_rate": np.array([state["qWs"][0] for state in states])[:, None],
        "well_bhp": np.array([state["bhp"][0] for state in states])[:, None],
    }


def _reference_arrays(reference):
    return {
        "pressure": np.asarray(reference["p_hist"], dtype=float),
        "free_displacement": np.asarray(reference["xd_hist"], dtype=float),
        "full_displacement": np.asarray(reference["u_hist"], dtype=float),
        "volumetric_strain": np.asarray(reference["vdiv_hist"], dtype=float),
        "strain": np.asarray(reference["strain_hist"], dtype=float),
        "effective_stress": np.asarray(reference["stress_hist"], dtype=float),
        "well_water_rate": np.asarray(reference["qws_hist"], dtype=float).reshape(-1, 1),
        "well_bhp": np.asarray(reference["bhp_hist"], dtype=float).reshape(-1, 1),
    }


def _compare(reference, candidate, times):
    nc = reference["pressure"].shape[1]
    cell = NamedAxis("cell", tuple(str(i) for i in range(nc)))
    voigt = NamedAxis("voigt", ("xx", "yy", "xy"))
    specs = {
        "pressure": ((cell,), "Pa", TolerancePolicy(relative=2e-6)),
        "free_displacement": ((NamedAxis("free_dof", tuple(str(i) for i in range(reference["free_displacement"].shape[1]))),), "m", TolerancePolicy(absolute=2e-6)),
        "full_displacement": ((NamedAxis("dof", tuple(str(i) for i in range(reference["full_displacement"].shape[1]))),), "m", TolerancePolicy(absolute=2e-6)),
        "volumetric_strain": ((cell,), "dimensionless", TolerancePolicy(absolute=2e-6)),
        "strain": ((cell, voigt), "dimensionless", TolerancePolicy(absolute=2e-6)),
        "effective_stress": ((cell, voigt), "Pa", TolerancePolicy(absolute=5.0, relative=2e-6)),
    }
    comparisons = {}
    for name, (axes, unit, tolerance) in specs.items():
        comparisons[name] = compare_named_fields(
            NamedFieldSeries(reference[name], times, axes, unit=unit),
            NamedFieldSeries(candidate[name], times, axes, unit=unit),
            tolerance=tolerance,
        )
    for name, unit, tolerance in (
        ("well_water_rate", "surface_m3/s", TolerancePolicy(absolute=1e-15)),
        ("well_bhp", "Pa", TolerancePolicy(relative=2e-6)),
    ):
        comparisons[name] = compare_series(
            ComparisonSeries(reference[name], times, ("inj",), unit=unit),
            ComparisonSeries(candidate[name], times, ("inj",), unit=unit),
            tolerance=tolerance,
        )
    return comparisons


def qualify(fixed_reference=FIXED_REFERENCE, coupled_reference=COUPLED_REFERENCE):
    fixed_ref = scipy.io.loadmat(fixed_reference, squeeze_me=True)
    coupled_ref = scipy.io.loadmat(coupled_reference, squeeze_me=True)
    fixed_result, fixed_times, fixed_runtime = _run("fixed stress splitting", fixed_ref)
    coupled_result, coupled_times, coupled_runtime = _run("fully coupled", coupled_ref)
    if not np.array_equal(fixed_times, coupled_times):
        raise ValueError("fixed-stress and fully coupled report times differ")
    fixed_arrays, coupled_arrays = _arrays(fixed_result), _arrays(coupled_result)

    fixed_graph, _ = qualify_fixed_mrst(Path(fixed_reference))
    graph = EvidenceGraph(fixed_graph.nodes, fixed_graph.edges)
    graph.add_node(EvidenceNode(
        "mrst_phase63", EvidenceNodeKind.MRST, "MRST Phase 63 fully coupled",
        version="MRST 2024a", metadata={
            "reference": str(Path(coupled_reference).relative_to(ROOT)),
            "sha256": _sha256(Path(coupled_reference)),
            "generator": "examples/validation/dump_phase63_reference.m",
        },
    ))
    graph.add_node(EvidenceNode(
        "pymrst_phase63", EvidenceNodeKind.PYMRST, "PyMRST Phase 63 fully coupled",
        version="working-tree", metadata={"runtime_seconds": coupled_runtime},
    ))

    coupled_mrst = _compare(_reference_arrays(coupled_ref), coupled_arrays, coupled_times)
    cross = _compare(fixed_arrays, coupled_arrays, coupled_times)
    translated = ("PyMRST is a Python translation of the MRST formulation",)
    shared_python = ("grid", "rock/fluid properties", "VEM mechanics", "well model")
    for name, comparison in coupled_mrst.items():
        graph.add_edge(edge_from_comparison(
            f"phase63_{name}", "mrst_phase63", "pymrst_phase63",
            f"Fully coupled {name.replace('_', ' ')} agrees with MRST",
            comparison, independence=Independence.PARTIAL,
            mappings=("report_time:exact", "entity_identity:v1"),
            shared_kernels=translated,
        ))
    for name, comparison in cross.items():
        graph.add_edge(edge_from_comparison(
            f"cross_{name}", "pymrst_phase64", "pymrst_phase63",
            f"Fixed-stress and fully coupled {name.replace('_', ' ')} agree",
            comparison, independence=Independence.SHARED_KERNEL,
            mappings=("report_time:exact", "entity_identity:v1"),
            shared_kernels=shared_python,
        ))
    fixed_iterations = [int(report["iterations"]) for report in fixed_result.reports]
    coupled_iterations = [int(report["iterations"]) for report in coupled_result.reports]
    graph.add_edge(EvidenceEdge(
        "cross_solver_work", "pymrst_phase64", "pymrst_phase63", "solver_work",
        "Both formulations converge; iteration counts describe different algorithms",
        Independence.SHARED_KERNEL, shared_kernels=shared_python,
        metrics={
            "fixed_stress_total_outer_iterations": sum(fixed_iterations),
            "fully_coupled_total_newton_iterations": sum(coupled_iterations),
            "fixed_stress_runtime_seconds": fixed_runtime,
            "fully_coupled_runtime_seconds": coupled_runtime,
        },
    ))
    summary = {
        "case": "phase64_fixed_stress_vs_phase63_fully_coupled_water",
        "passed": all(item.passed for item in (*coupled_mrst.values(), *cross.values())),
        "fixed_stress": {
            "iterations": fixed_iterations, "runtime_seconds": fixed_runtime,
            "reference_sha256": _sha256(Path(fixed_reference)),
        },
        "fully_coupled": {
            "iterations": coupled_iterations, "runtime_seconds": coupled_runtime,
            "reference_sha256": _sha256(Path(coupled_reference)),
        },
        "cross_formulation": {name: item.to_dict() for name, item in cross.items()},
        "fully_coupled_vs_mrst": {name: item.to_dict() for name, item in coupled_mrst.items()},
    }
    return graph, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph, summary = qualify()
    payload = {
        "artifact": "pymrst_cross_formulation_evidence_graph",
        "artifact_schema_version": 1,
        "graph": graph.to_dict(), "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {len(graph.edges)} evidence edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
