#!/usr/bin/env python3
"""Qualify the poromechanics residual and both AD Jacobian blocks against MRST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_mechanics.utils.equations_poro_mechanics import (
    equations_poro_mechanics,
)
from pymrst.research_studio.validation import (
    EvidenceGraph, EvidenceNode, EvidenceNodeKind, Independence, NamedAxis,
    NamedFieldSeries, TolerancePolicy, compare_named_fields, edge_from_comparison,
)
from pymrst.research_studio.validation.matlab import (
    MatlabFixtureSpec, MatlabReferenceAdapter, result_mappings,
)
from pymrst.utils.adi import init_variables_adi


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "scripts/matlab/ad_mechanics/ref_equations_poro_mechanics.mat"
GENERATOR = ROOT / "scripts/matlab/ad_mechanics/dump_equations_poro_mechanics_reference.m"
MRST_ROOT = ROOT.parent / "mrst-2024a-Geomech"
DEFAULT_OUTPUT = ROOT / "docs/research_studio/qualification/poromechanics_equations_evidence_graph.json"
ABSOLUTE_TOLERANCE = 1e-13

POROMECHANICS_FIXTURE_SPEC = MatlabFixtureSpec(
    fixture_id="mrst_poromechanics_equations_v1",
    case_name="Direct poromechanics residual and Jacobian qualification",
    reference_path=REFERENCE, generator_path=GENERATOR,
    generator_function="dump_equations_poro_mechanics_reference",
    mrst_root=MRST_ROOT, modules=("ad-core", "ad-mechanics"),
    mappings=("equation_dof:v1", "displacement_dof:v1", "pressure_cell:v1"),
    tolerances={
        "residual": {"absolute": ABSOLUTE_TOLERANCE},
        "jacobian_displacement": {"absolute": ABSOLUTE_TOLERANCE},
        "jacobian_pressure": {"absolute": ABSOLUTE_TOLERANCE},
    },
    input_artifacts={
        "mrst_equation_owner": MRST_ROOT / "modules/ad-mechanics/utils/equationsPoroMechanics.m",
    },
    solver_settings={
        "automatic_differentiation": "MRST ADI",
        "residual_scale": "1/(1e6*mean(cell_volume))",
        "linear_solve": "none; assembly-only fixture",
    },
)


def poromechanics_reference_adapter() -> MatlabReferenceAdapter:
    return MatlabReferenceAdapter(
        POROMECHANICS_FIXTURE_SPEC,
        lambda path: scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False),
    )


def _candidate(ref):
    model = ref["model"]
    mechanic = object.__new__(__import__(
        "pymrst.modules.ad_mechanics.mechanics", fromlist=["MechanicModel"]
    ).MechanicModel)
    volumes = np.asarray(model.G.cells.volumes, dtype=float).reshape(-1)
    mechanic.G = type("Grid", (), {"cells": type("Cells", (), {
        "volumes": volumes,
    })()})()
    mechanic.alpha = np.asarray(model.rock.alpha, dtype=float).reshape(-1)
    mechanic.operators = type("Operators", (), {
        "A": sp.csr_array(model.operators.A),
        "gradP": sp.csr_array(model.operators.gradP),
        "rhs": np.asarray(model.operators.rhs, dtype=float).reshape(-1),
    })()
    mechanic.res_scale = 1.0 / (1e6 * float(np.mean(volumes)))
    displacement, pressure = init_variables_adi(
        np.atleast_1d(ref["xd0"]), np.atleast_1d(ref["p0"]))
    equation = equations_poro_mechanics(mechanic, displacement, pressure)
    return {
        "residual": equation.val,
        "jacobian_displacement": equation.jac[0].toarray(),
        "jacobian_pressure": equation.jac[1].toarray(),
    }


def _comparison(reference, candidate, row_names, column_names=()):
    axes = (NamedAxis("equation_dof", tuple(row_names)),)
    reference_values = np.asarray(reference, dtype=float)
    candidate_values = np.asarray(candidate, dtype=float)
    if column_names:
        axes += (NamedAxis("primary_variable", tuple(column_names)),)
    return compare_named_fields(
        NamedFieldSeries(reference_values[None, ...], np.array([1.0]), axes),
        NamedFieldSeries(candidate_values[None, ...], np.array([1.0]), axes),
        tolerance=TolerancePolicy(absolute=ABSOLUTE_TOLERANCE),
    )


def qualify():
    adapter = poromechanics_reference_adapter()
    ref, manifest = adapter.load()
    candidate = _candidate(ref)
    ndof = candidate["residual"].size
    npressure = candidate["jacobian_pressure"].shape[1]
    dofs = tuple(f"u{i}" for i in range(ndof))
    cells = tuple(f"cell{i}" for i in range(npressure))
    comparisons = {
        "residual": _comparison(ref["residual"], candidate["residual"], dofs),
        "jacobian_displacement": _comparison(
            ref["jac_xd"], candidate["jacobian_displacement"], dofs, dofs),
        "jacobian_pressure": _comparison(
            ref["jac_p"], candidate["jacobian_pressure"], dofs, cells),
    }
    mappings = result_mappings(
        report_times=(1.0,), candidate_times=(1.0,),
        reference_cells=range(npressure), candidate_cells=range(npressure),
        reference_wells=("mechanics",), candidate_wells=("mechanics",),
    )
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode(
        "mrst_poromechanics_equations", EvidenceNodeKind.MRST,
        "MRST equationsPoroMechanics direct AD assembly",
        version=manifest.mrst_version, metadata={
            "fixture_manifest": str(adapter.spec.manifest_path.relative_to(ROOT)),
            "reference_sha256": manifest.reference_sha256,
            "generator_sha256": manifest.generator_sha256,
            "input_artifacts": manifest.input_artifacts,
            "matlab_release": manifest.matlab_release,
            "modules": list(manifest.modules),
        },
    ))
    graph.add_node(EvidenceNode(
        "pymrst_poromechanics_equations", EvidenceNodeKind.PYMRST,
        "PyMRST equations_poro_mechanics direct AD assembly", version="working-tree",
    ))
    for name, comparison in comparisons.items():
        graph.add_edge(edge_from_comparison(
            f"poromechanics_{name}", "mrst_poromechanics_equations",
            "pymrst_poromechanics_equations",
            f"Poromechanics {name.replace('_', ' ')} agrees with MRST",
            comparison, independence=Independence.PARTIAL,
            mappings=adapter.spec.mappings,
            shared_kernels=("PyMRST ports the MRST equationsPoroMechanics formulation",),
        ))
    summary = {
        "case": adapter.spec.case_name,
        "passed": all(value.passed for value in comparisons.values()),
        "fixture_manifest": manifest.to_dict(),
        "result_mappings": mappings.to_dict(),
        "comparisons": {name: value.to_dict() for name, value in comparisons.items()},
    }
    return graph, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regenerate-reference", action="store_true")
    parser.add_argument("--matlab", type=Path)
    args = parser.parse_args(argv)
    adapter = poromechanics_reference_adapter()
    if args.regenerate_reference:
        manifest = adapter.regenerate(executable=args.matlab)
        print(f"Regenerated {adapter.spec.reference_path} with {manifest.matlab_release}")
    graph, summary = qualify()
    payload = {"artifact": "pymrst_poromechanics_equations_evidence_graph",
               "artifact_schema_version": 1, "graph": graph.to_dict(),
               "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {len(graph.edges)} evidence edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
