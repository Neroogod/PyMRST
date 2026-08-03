#!/usr/bin/env python3
"""Qualify natural-variable and overall-composition SPE1 formulations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.io

from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.examples.run_compositional_spe1 import (
    run_native_compositional_spe1,
)
from pymrst.research_studio.validation import (
    ComparisonSeries, EvidenceEdge, EvidenceGraph, EvidenceNode,
    EvidenceNodeKind, Independence, NamedAxis, NamedFieldSeries,
    TolerancePolicy, compare_named_fields, compare_series,
    edge_from_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples/validation/ref_p114_spe1_native_compositional.mat"
DEFAULT_OUTPUT = ROOT / "docs/research_studio/qualification/compositional_natural_overall_evidence_graph.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(natural: bool):
    started = time.perf_counter()
    result = run_native_compositional_spe1(
        nsteps=2, natural=natural, min_dt=6.0 * 3600.0,
        interpolation="linear",
    )
    return result, _pack(result), time.perf_counter() - started


def _pack(result):
    """Pack formulation-independent report-step fields."""
    model = result.compositional_model
    saturation_rows = []
    for state in result.compositional_states:
        if hasattr(state, "sO") and state.sO is not None:
            s_o, s_g = state.sO, state.sG
        else:
            props = backend_flow_props(model.mix, model.eos_model, state.pressure, model.T, state.z)
            s_o = (1.0 - state.sW) * props.sL
            s_g = (1.0 - state.sW) * props.sV
        saturation_rows.append(np.column_stack((state.sW, s_o, s_g)))
    return {
        "pressure": np.vstack([state.pressure for state in result.compositional_states]),
        "components": np.stack([state.z for state in result.compositional_states]),
        "s": np.stack(saturation_rows),
        "bhp": np.vstack([state.bhp for state in result.compositional_states]),
        "qOs": np.vstack([state.qOs for state in result.compositional_states]),
        "qGs": np.vstack([state.qGs for state in result.compositional_states]),
    }


def _reference_arrays(ref: dict, prefix: str) -> dict:
    return {
        "pressure": np.asarray(ref[f"{prefix}_pressure"], dtype=float),
        "components": np.asarray(ref[f"{prefix}_components"], dtype=float),
        "saturations": np.asarray(ref[f"{prefix}_s"], dtype=float),
        "well_bhp": np.asarray(ref[f"{prefix}_bhp"], dtype=float),
        "well_oil_rate": np.asarray(ref[f"{prefix}_qOs"], dtype=float),
        "well_gas_rate": np.asarray(ref[f"{prefix}_qGs"], dtype=float),
    }


def _candidate_arrays(pack: dict) -> dict:
    return {
        "pressure": pack["pressure"], "components": pack["components"],
        "saturations": pack["s"], "well_bhp": pack["bhp"],
        "well_oil_rate": pack["qOs"], "well_gas_rate": pack["qGs"],
    }


def _compare(reference: dict, candidate: dict, times: np.ndarray):
    nc = reference["pressure"].shape[1]
    ncomp = reference["components"].shape[2]
    nw = reference["well_bhp"].shape[1]
    cell = NamedAxis("cell", tuple(str(i) for i in range(nc)))
    component = NamedAxis("component", tuple(f"pseudo_component_{i + 1}" for i in range(ncomp)))
    phase = NamedAxis("phase", ("water", "oil", "gas"))
    well_names = tuple(("injector", "producer")[:nw])
    if len(well_names) != nw:
        well_names = tuple(f"well_{i}" for i in range(nw))

    comparisons = {
        "pressure": compare_named_fields(
            NamedFieldSeries(reference["pressure"], times, (cell,), unit="Pa"),
            NamedFieldSeries(candidate["pressure"], times, (cell,), unit="Pa"),
            tolerance=TolerancePolicy(absolute=500.0),
        ),
        "components": compare_named_fields(
            NamedFieldSeries(reference["components"], times, (cell, component), unit="mole_fraction"),
            NamedFieldSeries(candidate["components"], times, (cell, component), unit="mole_fraction"),
            tolerance=TolerancePolicy(absolute=1.0e-7),
        ),
        "saturations": compare_named_fields(
            NamedFieldSeries(reference["saturations"], times, (cell, phase), unit="volume_fraction"),
            NamedFieldSeries(candidate["saturations"], times, (cell, phase), unit="volume_fraction"),
            tolerance=TolerancePolicy(absolute=1.0e-6),
        ),
    }
    for name, unit, tolerance in (
        ("well_bhp", "Pa", TolerancePolicy(absolute=500.0)),
        ("well_oil_rate", "surface_m3/s", TolerancePolicy(absolute=5.0e-8)),
        ("well_gas_rate", "surface_m3/s", TolerancePolicy(absolute=1.0e-6)),
    ):
        comparisons[name] = compare_series(
            ComparisonSeries(reference[name], times, well_names, unit=unit),
            ComparisonSeries(candidate[name], times, well_names, unit=unit),
            tolerance=tolerance,
        )
    return comparisons


def qualify(reference: Path = REFERENCE):
    reference = Path(reference)
    ref = scipy.io.loadmat(reference, squeeze_me=False)
    times = np.cumsum(np.asarray(ref["tsteps"], dtype=float).reshape(-1))
    overall_result, overall_pack, overall_runtime = _run(False)
    natural_result, natural_pack, natural_runtime = _run(True)
    overall = _candidate_arrays(overall_pack)
    natural = _candidate_arrays(natural_pack)

    graph = EvidenceGraph()
    digest = _sha256(reference)
    common_ref = {
        "reference": str(reference.relative_to(ROOT)), "sha256": digest,
        "generator": "examples/validation/dump_phase114_reference.m",
    }
    for node_id, label in (("mrst_overall", "MRST overall composition"),
                           ("mrst_natural", "MRST natural variables")):
        graph.add_node(EvidenceNode(node_id, EvidenceNodeKind.MRST, label,
                                   version="MRST 2024a", metadata=common_ref))
    graph.add_node(EvidenceNode("pymrst_overall", EvidenceNodeKind.PYMRST,
                               "PyMRST overall composition", version="working-tree",
                               metadata={"runtime_seconds": overall_runtime}))
    graph.add_node(EvidenceNode("pymrst_natural", EvidenceNodeKind.PYMRST,
                               "PyMRST natural variables", version="working-tree",
                               metadata={"runtime_seconds": natural_runtime}))

    overall_mrst = _compare(_reference_arrays(ref, "overall"), overall, times)
    natural_mrst = _compare(_reference_arrays(ref, "natural"), natural, times)
    cross = _compare(overall, natural, times)
    for prefix, source, target, values in (
        ("overall_mrst", "mrst_overall", "pymrst_overall", overall_mrst),
        ("natural_mrst", "mrst_natural", "pymrst_natural", natural_mrst),
    ):
        for name, comparison in values.items():
            graph.add_edge(edge_from_comparison(
                f"{prefix}_{name}", source, target,
                f"{name.replace('_', ' ')} agrees with MRST", comparison,
                independence=Independence.PARTIAL,
                mappings=("report_time:exact", "named_entity_identity:v1"),
                shared_kernels=("PyMRST is a Python translation of the MRST formulation",),
            ))
    shared = ("grid", "rock/fluid properties", "K-value EOS", "facility model")
    for name, comparison in cross.items():
        graph.add_edge(edge_from_comparison(
            f"cross_{name}", "pymrst_overall", "pymrst_natural",
            f"Overall and natural {name.replace('_', ' ')} agree", comparison,
            independence=Independence.SHARED_KERNEL,
            mappings=("report_time:exact", "named_entity_identity:v1"),
            shared_kernels=shared,
        ))

    overall_iterations = [int(v) for v in overall_result.iterations]
    natural_iterations = [int(v) for v in natural_result.iterations]
    graph.add_edge(EvidenceEdge(
        "cross_solver_work", "pymrst_overall", "pymrst_natural", "solver_work",
        "Both formulations converge; work is reported without requiring equal Newton paths",
        Independence.SHARED_KERNEL, shared_kernels=shared, metrics={
            "overall_total_newton_iterations": sum(overall_iterations),
            "natural_total_newton_iterations": sum(natural_iterations),
            "overall_runtime_seconds": overall_runtime,
            "natural_runtime_seconds": natural_runtime,
            "mrst_overall_total_newton_iterations": int(np.asarray(ref["overall_iterations"], dtype=int).sum()),
            "mrst_natural_total_newton_iterations": int(np.asarray(ref["natural_iterations"], dtype=int).sum()),
        },
    ))
    numerical = (*overall_mrst.values(), *natural_mrst.values(), *cross.values())
    summary = {
        "case": "phase114_spe1_natural_variables_vs_overall_composition",
        "passed": all(item.passed for item in numerical),
        "reference_sha256": digest,
        "overall": {"iterations": overall_iterations, "runtime_seconds": overall_runtime},
        "natural": {"iterations": natural_iterations, "runtime_seconds": natural_runtime},
        "overall_vs_mrst": {k: v.to_dict() for k, v in overall_mrst.items()},
        "natural_vs_mrst": {k: v.to_dict() for k, v in natural_mrst.items()},
        "cross_formulation": {k: v.to_dict() for k, v in cross.items()},
    }
    return graph, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph, summary = qualify(args.reference)
    payload = {"artifact": "pymrst_compositional_cross_formulation_evidence_graph",
               "artifact_schema_version": 1, "graph": graph.to_dict(), "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {len(graph.edges)} evidence edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
