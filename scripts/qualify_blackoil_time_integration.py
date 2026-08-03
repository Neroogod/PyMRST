#!/usr/bin/env python3
"""Qualify SPE1 FIM, explicit transport, and AIM time integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.io

from pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example import (
    run_blackoil_time_integration_example,
)
from pymrst.research_studio.validation import (
    ComparisonSeries, EvidenceEdge, EvidenceGraph, EvidenceNode,
    EvidenceNodeKind, Independence, NamedAxis, NamedFieldSeries,
    TolerancePolicy, compare_named_fields, compare_series,
    edge_from_comparison,
)
from pymrst.research_studio.validation.matlab import (
    MatlabFixtureSpec, MatlabReferenceAdapter, result_mappings,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples/validation/ref_time_integration_spe1.mat"
DEFAULT_OUTPUT = ROOT / "docs/research_studio/qualification/blackoil_time_integration_evidence_graph.json"
METHODS = {
    "fim": "Fully-Implicit",
    "explicit": "Explicit",
    "aim": "Adaptive-Implicit (AIM)",
}


def _reference_arrays(value) -> dict[str, np.ndarray]:
    return {
        "pressure": np.asarray(value.pressure, dtype=float),
        "saturation": np.stack((value.sW, value.sG), axis=-1),
        "solution_gas_ratio": np.asarray(value.rs, dtype=float),
        "well_bhp": np.asarray(value.bhp, dtype=float),
        "well_water_rate": np.asarray(value.qWs, dtype=float),
        "well_oil_rate": np.asarray(value.qOs, dtype=float),
        "well_gas_rate": np.asarray(value.qGs, dtype=float),
    }


def _candidate_arrays(variant) -> dict[str, np.ndarray]:
    history = variant.result.history
    return {
        "pressure": history["pressure"],
        "saturation": np.stack((history["sW"], history["sG"]), axis=-1),
        "solution_gas_ratio": history["rs"],
        "well_bhp": history["bhp"],
        "well_water_rate": history["qWs"],
        "well_oil_rate": history["qOs"],
        "well_gas_rate": history["qGs"],
    }


def _compare(reference, candidate, times, tolerances, *, reference_wells, candidate_wells):
    nc = reference["pressure"].shape[1]
    cell = NamedAxis("cell", tuple(str(i) for i in range(nc)))
    phase = NamedAxis("phase", ("water", "gas"))
    comparisons = {
        "pressure": compare_named_fields(
            NamedFieldSeries(reference["pressure"], times, (cell,), unit="Pa"),
            NamedFieldSeries(candidate["pressure"], times, (cell,), unit="Pa"),
            tolerance=TolerancePolicy(absolute=tolerances["pressure"]),
        ),
        "saturation": compare_named_fields(
            NamedFieldSeries(reference["saturation"], times, (cell, phase), unit="volume_fraction"),
            NamedFieldSeries(candidate["saturation"], times, (cell, phase), unit="volume_fraction"),
            tolerance=TolerancePolicy(absolute=tolerances["saturation"]),
        ),
        "solution_gas_ratio": compare_named_fields(
            NamedFieldSeries(reference["solution_gas_ratio"], times, (cell,), unit="surface_m3/surface_m3"),
            NamedFieldSeries(candidate["solution_gas_ratio"], times, (cell,), unit="surface_m3/surface_m3"),
            tolerance=TolerancePolicy(absolute=tolerances["solution_gas_ratio"]),
        ),
    }
    for name, unit in (
        ("well_bhp", "Pa"), ("well_water_rate", "surface_m3/s"),
        ("well_oil_rate", "surface_m3/s"), ("well_gas_rate", "surface_m3/s"),
    ):
        comparisons[name] = compare_series(
            ComparisonSeries(reference[name], times, reference_wells, unit=unit),
            ComparisonSeries(candidate[name], times, candidate_wells, unit=unit),
            tolerance=TolerancePolicy(absolute=tolerances[name]),
        )
    return comparisons


STRICT_MRST_TOLERANCES = {
    "pressure": 100.0, "saturation": 1e-6, "solution_gas_ratio": 1e-3,
    "well_bhp": 2e3, "well_water_rate": 1e-12, "well_oil_rate": 1e-12,
    "well_gas_rate": 1e-12,
}
MRST_TOLERANCES = {key: dict(STRICT_MRST_TOLERANCES) for key in METHODS}
CROSS_TOLERANCES = {
    "pressure": 2.5e5, "saturation": 2e-2, "solution_gas_ratio": 5.0,
    "well_bhp": 2.5e5, "well_water_rate": 1e-10, "well_oil_rate": 1e-10,
    "well_gas_rate": 1e-10,
}

SPE1_FIXTURE_SPEC = MatlabFixtureSpec(
    fixture_id="mrst_spe1_blackoil_time_integration_v1",
    case_name="SPE1 FIM explicit AIM two-step qualification",
    reference_path=REFERENCE,
    generator_path=ROOT / "examples/validation/dump_time_integration_spe1_reference.m",
    generator_function="dump_time_integration_spe1_reference",
    mrst_root=ROOT.parent / "mrst-2024a-Geomech",
    modules=("ad-core", "ad-blackoil", "deckformat", "ad-props",
             "linearsolvers", "test-suite"),
    mappings=("report_time:exact", "well_name:v1", "cell_identity:v1",
              "phase_name:v1"),
    tolerances={key: MRST_TOLERANCES[key] for key in METHODS},
    input_artifacts={
        "spe1_deck": ROOT.parent / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA",
    },
    solver_settings={
        "timestep_strategy": "none", "report_steps": 2,
        "time_discretizations": ["fully-implicit", "explicit", "adaptive-implicit"],
        "nonlinear_solver": "initEclipseProblemAD default for SPE1",
    },
)


def spe1_reference_adapter(reference: Path = REFERENCE) -> MatlabReferenceAdapter:
    reference = Path(reference).resolve()
    spec = SPE1_FIXTURE_SPEC
    if reference != spec.reference_path:
        spec = MatlabFixtureSpec(
            fixture_id=spec.fixture_id, case_name=spec.case_name,
            reference_path=reference, generator_path=spec.generator_path,
            generator_function=spec.generator_function, mrst_root=spec.mrst_root,
            modules=spec.modules, mappings=spec.mappings,
            tolerances=spec.tolerances, input_artifacts=spec.input_artifacts,
            solver_settings=spec.solver_settings,
        )
    return MatlabReferenceAdapter(
        spec, lambda path: scipy.io.loadmat(
            path, squeeze_me=True, struct_as_record=False))


def qualify(reference: Path = REFERENCE):
    adapter = spe1_reference_adapter(reference)
    ref, manifest = adapter.load()
    reference = adapter.spec.reference_path
    times = np.cumsum(np.asarray(ref["tsteps"], dtype=float).reshape(-1))
    started = time.perf_counter()
    variants = run_blackoil_time_integration_example("spe1", nsteps=times.size)
    total_runtime = time.perf_counter() - started

    graph = EvidenceGraph()
    digest = manifest.reference_sha256
    for key, label in METHODS.items():
        graph.add_node(EvidenceNode(
            f"mrst_{key}", EvidenceNodeKind.MRST, f"MRST SPE1 {label}",
            version=manifest.mrst_version, metadata={
                "reference": str(reference.relative_to(ROOT)), "sha256": digest,
                "generator": str(adapter.spec.generator_path.relative_to(ROOT)),
                "generator_sha256": manifest.generator_sha256,
                "matlab_release": manifest.matlab_release,
                "modules": list(manifest.modules),
                "fixture_manifest": str(adapter.spec.manifest_path.relative_to(ROOT)),
            },
        ))
        graph.add_node(EvidenceNode(
            f"pymrst_{key}", EvidenceNodeKind.PYMRST, f"PyMRST SPE1 {label}",
            version="working-tree",
        ))

    mrst_comparisons = {}
    py_arrays = {}
    mapping_records = {}
    for key, label in METHODS.items():
        py_arrays[key] = _candidate_arrays(variants[label])
        mapping_records[key] = result_mappings(
            report_times=times, candidate_times=times,
            reference_cells=range(ref[key].pressure.shape[1]),
            candidate_cells=range(py_arrays[key]["pressure"].shape[1]),
            reference_wells=("injector", "producer"),
            candidate_wells=("producer", "injector"),
            reference_phases=("water", "gas"), candidate_phases=("water", "gas"),
        )
        comparisons = _compare(
            _reference_arrays(ref[key]), py_arrays[key], times, MRST_TOLERANCES[key],
            reference_wells=("injector", "producer"),
            candidate_wells=("producer", "injector"),
        )
        mrst_comparisons[key] = comparisons
        for name, comparison in comparisons.items():
            graph.add_edge(edge_from_comparison(
                f"{key}_mrst_{name}", f"mrst_{key}", f"pymrst_{key}",
                f"{label} {name.replace('_', ' ')} agrees with MRST",
                comparison, independence=Independence.PARTIAL,
                mappings=("report_time:exact", "well_name:v1", "cell_identity:v1"),
                shared_kernels=("PyMRST is a Python translation of the MRST formulation",),
            ))

    cross = {}
    shared = ("grid", "rock/fluid properties", "well model", "nonlinear solver")
    for key in ("explicit", "aim"):
        comparisons = _compare(
            py_arrays["fim"], py_arrays[key], times, CROSS_TOLERANCES,
            reference_wells=("producer", "injector"),
            candidate_wells=("producer", "injector"),
        )
        cross[key] = comparisons
        for name, comparison in comparisons.items():
            graph.add_edge(edge_from_comparison(
                f"fim_vs_{key}_{name}", "pymrst_fim", f"pymrst_{key}",
                f"FIM and {METHODS[key]} {name.replace('_', ' ')} remain within the declared discretization envelope",
                comparison, independence=Independence.SHARED_KERNEL,
                mappings=("report_time:exact", "well_name:v1", "cell_identity:v1"),
                shared_kernels=shared,
            ))

    metrics = {"total_python_runtime_seconds": total_runtime}
    diagnostics = {}
    for key, label in METHODS.items():
        d = variants[label].diagnostics
        diagnostics[key] = {
            "iterations": d.iterations.tolist(), "cuts": d.cuts.tolist(),
            "ministeps": d.ministeps.tolist(),
            "implicit_fraction": d.implicit_fraction.tolist(),
            "mrst_iterations": np.asarray(ref[key].iterations, dtype=int).reshape(-1).tolist(),
            "mrst_ministeps": np.asarray(ref[key].ministeps, dtype=float).reshape(-1).tolist(),
            "mrst_implicit_fraction": np.asarray(ref[key].implicitFraction, dtype=float).reshape(-1).tolist(),
        }
        metrics[f"{key}_total_iterations"] = float(np.sum(d.iterations))
        metrics[f"{key}_ministep_count"] = float(d.ministeps.size)
        metrics[f"{key}_mean_implicit_fraction"] = float(np.mean(d.implicit_fraction))
    graph.add_edge(EvidenceEdge(
        "time_integration_solver_work", "pymrst_fim", "pymrst_aim", "solver_work",
        "Work, timestep cuts, ministeps and AIM partition are reported without requiring equal algorithms",
        Independence.SHARED_KERNEL, shared_kernels=shared, metrics=metrics,
    ))
    all_results = [v for group in mrst_comparisons.values() for v in group.values()]
    all_results += [v for group in cross.values() for v in group.values()]
    summary = {
        "case": "spe1_blackoil_fim_explicit_aim_two_steps",
        "passed": all(result.passed for result in all_results),
        "reference_sha256": digest, "diagnostics": diagnostics,
        "fixture_manifest": manifest.to_dict(),
        "result_mappings": {key: value.to_dict()
                            for key, value in mapping_records.items()},
        "mrst_comparisons": {k: {n: v.to_dict() for n, v in values.items()}
                             for k, values in mrst_comparisons.items()},
        "cross_formulation": {k: {n: v.to_dict() for n, v in values.items()}
                              for k, values in cross.items()},
    }
    return graph, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regenerate-reference", action="store_true")
    parser.add_argument("--matlab", type=Path)
    args = parser.parse_args(argv)
    if args.regenerate_reference:
        manifest = spe1_reference_adapter(args.reference).regenerate(
            executable=args.matlab)
        print(f"Regenerated {args.reference} with {manifest.matlab_release}")
    graph, summary = qualify(args.reference)
    payload = {"artifact": "pymrst_blackoil_time_integration_evidence_graph",
               "artifact_schema_version": 1, "graph": graph.to_dict(), "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {len(graph.edges)} evidence edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
