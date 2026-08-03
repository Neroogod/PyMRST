#!/usr/bin/env python3
"""Generate the first real MRST/PyMRST Research Studio evidence graph.

The reference is produced by ``examples/validation/dump_phase64_reference.m``
with MRST 2024a. This script runs the corresponding PyMRST case, compares all
stored field and well histories through the normalized comparison engine, and
writes a strict EvidenceGraph v1 artifact. It exits nonzero if any edge fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case
from pymrst.research_studio.validation import (
    ComparisonSeries, EvidenceGraph, EvidenceNode, EvidenceNodeKind,
    Independence, NamedAxis, NamedFieldSeries, TolerancePolicy,
    compare_named_fields, compare_series, edge_from_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT/"examples/validation/ref_p64_run2d_water_fixedstress.mat"
DEFAULT_OUTPUT = ROOT/"docs/research_studio/qualification/phase64_evidence_graph.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(values, times, axes, unit):
    return NamedFieldSeries(np.asarray(values, dtype=float), times, axes, unit=unit)


def qualify(reference_path: Path) -> tuple[EvidenceGraph, dict]:
    reference = scipy.io.loadmat(reference_path, squeeze_me=True)
    nx, ny = np.asarray(reference["cartDim"], dtype=int).reshape(-1)
    dimensions = tuple(np.asarray(reference["L"], dtype=float).reshape(-1))
    dt = np.asarray(reference["dt"], dtype=float).reshape(-1)
    times = np.cumsum(dt)
    result = run_2d_case(
        fluid_model="water", method="fixed stress splitting",
        nx=int(nx), ny=int(ny), physdims=dimensions,
        nsteps=dt.size, dt=dt, p_ref=float(reference["pRef"]),
    )
    if not all(report["converged"] for report in result.reports):
        raise RuntimeError("PyMRST Phase 64 qualification run did not converge")
    states = result.states[1:]
    diagnostics = result.diagnostics[1:]
    nc = int(nx*ny)
    cell_axis = NamedAxis("cell", tuple(str(i) for i in range(nc)))
    voigt_axis = NamedAxis("voigt", ("xx", "yy", "xy"))

    comparisons = {
        "pressure": compare_named_fields(
            _field(reference["p_hist"], times, (cell_axis,), "Pa"),
            _field(np.vstack([state["pressure"] for state in states]), times, (cell_axis,), "Pa"),
            tolerance=TolerancePolicy(relative=2e-6, scale=1.0),
        ),
        "free_displacement": compare_named_fields(
            _field(reference["xd_hist"], times,
                   (NamedAxis("free_dof", tuple(str(i) for i in range(np.asarray(reference["xd_hist"]).shape[1]))),), "m"),
            _field(np.vstack([state["xd"] for state in states]), times,
                   (NamedAxis("free_dof", tuple(str(i) for i in range(np.asarray(reference["xd_hist"]).shape[1]))),), "m"),
            tolerance=TolerancePolicy(absolute=2e-6, scale=1.0),
        ),
        "full_displacement": compare_named_fields(
            _field(reference["u_hist"], times,
                   (NamedAxis("dof", tuple(str(i) for i in range(np.asarray(reference["u_hist"]).shape[1]))),), "m"),
            _field(np.vstack([state["u"] for state in states]), times,
                   (NamedAxis("dof", tuple(str(i) for i in range(np.asarray(reference["u_hist"]).shape[1]))),), "m"),
            tolerance=TolerancePolicy(absolute=2e-6, scale=1.0),
        ),
        "volumetric_strain": compare_named_fields(
            _field(reference["vdiv_hist"], times, (cell_axis,), "dimensionless"),
            _field(np.vstack([item["volumetric_strain"] for item in diagnostics]), times,
                   (cell_axis,), "dimensionless"),
            tolerance=TolerancePolicy(absolute=2e-6, scale=1.0),
        ),
        "strain": compare_named_fields(
            _field(reference["strain_hist"], times, (cell_axis, voigt_axis), "dimensionless"),
            _field(np.stack([item["strain"] for item in diagnostics]), times,
                   (cell_axis, voigt_axis), "dimensionless"),
            tolerance=TolerancePolicy(absolute=2e-6, scale=1.0),
        ),
        "effective_stress": compare_named_fields(
            _field(reference["stress_hist"], times, (cell_axis, voigt_axis), "Pa"),
            _field(np.stack([item["effective_stress"] for item in diagnostics]), times,
                   (cell_axis, voigt_axis), "Pa"),
            tolerance=TolerancePolicy(absolute=1.0, relative=2e-6, scale=1.0),
        ),
        "well_water_rate": compare_series(
            ComparisonSeries(np.asarray(reference["qws_hist"]).reshape(-1, 1), times,
                             ("inj",), unit="surface_m3/s",
                             sign_convention="positive_into_reservoir"),
            ComparisonSeries(np.array([state["qWs"][0] for state in states])[:, None], times,
                             ("inj",), unit="surface_m3/s",
                             sign_convention="positive_into_reservoir"),
            tolerance=TolerancePolicy(absolute=1e-15, scale=1.0),
        ),
        "well_bhp": compare_series(
            ComparisonSeries(np.asarray(reference["bhp_hist"]).reshape(-1, 1), times,
                             ("inj",), unit="Pa"),
            ComparisonSeries(np.array([state["bhp"][0] for state in states])[:, None], times,
                             ("inj",), unit="Pa"),
            tolerance=TolerancePolicy(relative=2e-6, scale=1.0),
        ),
    }
    nodes = [
        EvidenceNode(
            "mrst_phase64", EvidenceNodeKind.MRST, "MRST Phase 64 reference",
            version="MRST 2024a",
            metadata={
                "reference": str(reference_path.relative_to(ROOT)),
                "sha256": _sha256(reference_path),
                "generator": "examples/validation/dump_phase64_reference.m",
                "matlab_command": (
                    "/Applications/MATLAB_R2021b.app/bin/matlab -batch "
                    "\"run('examples/validation/dump_phase64_reference.m')\""
                ),
            },
        ),
        EvidenceNode(
            "pymrst_phase64", EvidenceNodeKind.PYMRST, "PyMRST Phase 64 run",
            version="working-tree",
            metadata={
                "runner": "pymrst.modules.ad_mechanics.examples.run_2d_case",
                "grid_cells": nc,
                "report_steps": int(dt.size),
                "all_converged": True,
            },
        ),
    ]
    graph = EvidenceGraph(nodes)
    for quantity, comparison in comparisons.items():
        graph.add_edge(edge_from_comparison(
            f"phase64_{quantity}", "mrst_phase64", "pymrst_phase64",
            f"Phase 64 {quantity.replace('_', ' ')} agrees within its declared tolerance",
            comparison, independence=Independence.PARTIAL,
            mappings=("report_time:exact", "entity_identity:v1"),
            shared_kernels=("PyMRST is a Python translation of the MRST formulation",),
        ))
    summary = {
        "case": "phase64_water_fixed_stress",
        "reference_sha256": _sha256(reference_path),
        "passed": all(comparison.passed for comparison in comparisons.values()),
        "edges": {name: comparison.to_dict() for name, comparison in comparisons.items()},
    }
    return graph, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph, summary = qualify(args.reference.resolve())
    payload = {
        "artifact": "pymrst_qualification_evidence_graph",
        "artifact_schema_version": 1,
        "graph": graph.to_dict(),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(f"{'PASS' if summary['passed'] else 'FAIL'} Phase 64: {len(graph.edges)} evidence edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
