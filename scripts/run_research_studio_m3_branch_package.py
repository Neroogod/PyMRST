#!/usr/bin/env python3
"""Integrated M3.6--M3.9 SPE1 branch, export and research-package proof."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pymrst.research_studio.case import CaseComponent, ReservoirCase, write_reservoir_case
from pymrst.research_studio.eclipse import RunnableWaterDeck, write_runnable_water_deck
from pymrst.research_studio.evidence import capture_manifest
from pymrst.research_studio.experiments import (
    BranchDefinition, LocalEnsembleExecutor, MaterializedVariantMatrix,
    MemberResultStore, PackageArtifact, ResearchPackageDefinition,
    ScenarioQuantityPolicy, compare_scenarios, execute_branch,
    generate_research_package, verify_branch_lineage,
)
from pymrst.research_studio.validation import TolerancePolicy


BASE_RUNNER = "pymrst.research_studio.experiments.runners:spe1_time_integration_member"
BRANCH_RUNNER = "pymrst.research_studio.experiments.runners:spe1_branch_member"


def _portable_spe1():
    return ReservoirCase(
        "SPE1 three-step FIM",
        CaseComponent(
            "pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example:_run_spe1_variant",
            {"mode": "fully-implicit", "nsteps": 3}),
        grid={"kind": "SPE1 deck"}, properties={"source": "BENCH_SPE1.DATA"},
        physics={"model": "black-oil", "phases": ["water", "oil", "gas"],
                 "disgas": True, "vapoil": False},
        discretization={"flow": "tpfa", "time": "fully-implicit"},
        facilities={"wells": "deck"}, schedule={"report_steps": 3},
        solver={"nonlinear": "NonLinearSolver", "max_iterations": 30},
        metadata={"work_packages": ["M3.6", "M3.7", "M3.9"]})


def _write_water_deck(root, manifest):
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import Rock
    grid = compute_geometry(cart_grid([1, 1, 1], [10.0, 10.0, 5.0]))
    rock = Rock(perm=np.array([[100.0, 100.0, 10.0]]), poro=np.array([0.2]))
    grdecl = {
        "cartDims": [1, 1, 1],
        "COORD": [0, 0, 0, 0, 0, 5, 10, 0, 0, 10, 0, 5,
                  0, 10, 0, 0, 10, 5, 10, 10, 0, 10, 10, 5],
        "ZCORN": [0, 0, 0, 0, 5, 5, 5, 5], "ACTNUM": [1],
    }
    return write_runnable_water_deck(
        root / "water_export.DATA", grid, rock, grdecl=grdecl,
        specification=RunnableWaterDeck(
            "METRIC", 2.5e7, (1.0, 2.0), (2.5e7, 1.0, 4e-10, 1e-3, 0.0)),
        manifest=manifest)


def run(matrix_path: Path, output: Path, package_path: Path):
    matrix = MaterializedVariantMatrix.from_dict(
        json.loads(matrix_path.read_text())["matrices"][0])
    ensemble = LocalEnsembleExecutor(output / "base", max_workers=3).run(
        matrix, runner=BASE_RUNNER)
    if ensemble.failed:
        raise RuntimeError("base SPE1 ensemble failed")
    parent_outcome = next(value for value in ensemble.completed
                          if value.coordinate.variant_name == "fim")
    parent = MemberResultStore(parent_outcome.store_path)
    parent_id = parent.record()["coordinate"]["member_id"]
    branches = {}
    for name, mode in (("fim-continuation", "fully-implicit"),
                       ("aim-branch", "adaptive-implicit")):
        definition = BranchDefinition(
            name, parent_id, parent_time_index=-1,
            components={"time_integration": mode},
            required_fields=("pressure", "sW", "sG", "rs"),
            comparison_group="spe1-branch")
        outcome = execute_branch(parent, definition, output=output / "branches",
                                 runner=BRANCH_RUNNER, seed=20240803)
        verify_branch_lineage(outcome)
        branches[name] = (definition, outcome, MemberResultStore(outcome.store_path).lazy())
    policies = {
        "fields.pressure": ScenarioQuantityPolicy(
            TolerancePolicy(absolute=2.5e5), unit="Pa"),
        "fields.sW": ScenarioQuantityPolicy(TolerancePolicy(absolute=2e-2)),
        "fields.sG": ScenarioQuantityPolicy(TolerancePolicy(absolute=2e-2)),
        "fields.rs": ScenarioQuantityPolicy(TolerancePolicy(absolute=5.0)),
        "wells.bhp": ScenarioQuantityPolicy(TolerancePolicy(absolute=2.5e5), unit="Pa"),
        "wells.qWs": ScenarioQuantityPolicy(TolerancePolicy(absolute=1e-3)),
        "wells.qOs": ScenarioQuantityPolicy(TolerancePolicy(absolute=1e-3)),
        "wells.qGs": ScenarioQuantityPolicy(TolerancePolicy(absolute=1.0)),
    }
    comparison = compare_scenarios(
        branches["fim-continuation"][2], branches["aim-branch"][2],
        policies=policies, comparison_group="spe1-branch")
    evidence_root = output / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    comparison_path = evidence_root / "branch_comparison.json"
    comparison_path.write_text(json.dumps(comparison.to_dict(), indent=2,
                                          sort_keys=True)+"\n")
    table_path = evidence_root / "branch_errors.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "passed", "maximum_absolute_error",
                         "worst_time", "worst_entity"))
        for name, result in comparison.results.items():
            writer.writerow((name, result.passed, result.maximum_absolute_error,
                             result.worst_time, result.worst_entity))
    case_path = write_reservoir_case(output / "spe1.case.json", _portable_spe1())
    manifest = capture_manifest(
        "run_m3_6_3_9", code_fingerprint=matrix.matrix_id,
        inputs=(matrix_path, case_path),
        command=("python3", "scripts/run_research_studio_m3_branch_package.py"),
        configuration={"matrix_id": matrix.matrix_id,
                       "parent_member_id": parent_id})
    manifest_path = evidence_root / "reproducibility_manifest.json"
    manifest_path.write_text(manifest.to_json()+"\n")
    deck_path = _write_water_deck(output, manifest)
    summary = {
        "schema_version": 1, "work_packages": ["M3.6", "M3.7", "M3.8", "M3.9"],
        "base_ensemble": ensemble.to_dict(),
        "branches": {name: {"definition": definition.to_dict(),
                             "outcome": outcome.to_dict()}
                     for name, (definition, outcome, _) in branches.items()},
        "comparison": comparison.to_dict(),
        "portable_case_hash": _portable_spe1().content_hash,
        "eclipse_deck": str(deck_path),
    }
    summary_path = evidence_root / "qualification.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    package_definition = ResearchPackageDefinition(
        "PyMRST M3 SPE1 branch qualification", matrix.definition.experiment_id,
        tuple(value[1].member_id for value in branches.values()),
        ("python3", "-c",
         "from pymrst.research_studio.case import read_reservoir_case; "
         "c=read_reservoir_case('cases/spe1.case.json'); c.construct(); "
         "print(c.content_hash)"),
        artifacts=(
            PackageArtifact(str(case_path), "cases"),
            PackageArtifact(str(deck_path), "cases"),
            PackageArtifact(str(deck_path.with_suffix('.pymrst-manifest.json')), "evidence"),
            PackageArtifact(str(manifest_path), "evidence"),
            PackageArtifact(str(summary_path), "evidence"),
            PackageArtifact(str(comparison_path), "comparisons"),
            PackageArtifact(str(table_path), "tables"),
        ), metadata={"matrix_id": matrix.matrix_id})
    package = generate_research_package(package_definition, package_path)
    return summary, package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        default=Path("docs/research_studio/experiments/variant_matrices.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/research_studio/m3_branch_package"))
    parser.add_argument("--package", type=Path,
                        default=Path("results/research_studio/m3_research_package"))
    arguments = parser.parse_args()
    summary, package = run(arguments.matrix, arguments.output, arguments.package)
    print(json.dumps({"branch_comparison_passed": summary["comparison"]["passed"],
                      "package": str(package)}, indent=2))


if __name__ == "__main__":
    main()
