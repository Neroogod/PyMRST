#!/usr/bin/env python3
"""Execute and qualify the persisted M3.3--M3.5 SPE1 experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pymrst.research_studio.experiments import (
    LocalEnsembleExecutor, MaterializedVariantMatrix, MemberResultStore,
    ScenarioQuantityPolicy, compare_performance, compare_scenarios,
)
from pymrst.research_studio.export import (
    NormalizedResultDataset, read_zarr, write_csv_summary,
    write_parquet_summary, write_zarr,
)
from pymrst.research_studio.validation import TolerancePolicy


RUNNER = "pymrst.research_studio.experiments.runners:spe1_time_integration_member"


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_matrix(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = MaterializedVariantMatrix.from_dict(payload["matrices"][0])
    if tuple(item.variant_name for item in matrix.coordinates) != (
            "fim", "explicit", "aim"):
        raise ValueError("the first persisted matrix is not the SPE1 integration matrix")
    return matrix


def _policies():
    # These are cross-formulation engineering tolerances, not MRST parity limits.
    return {
        "fields.pressure": ScenarioQuantityPolicy(
            TolerancePolicy(absolute=2.5e5), unit="Pa"),
        "fields.sW": ScenarioQuantityPolicy(TolerancePolicy(absolute=2.0e-2)),
        "fields.sG": ScenarioQuantityPolicy(TolerancePolicy(absolute=2.0e-2)),
        "fields.rs": ScenarioQuantityPolicy(TolerancePolicy(absolute=5.0)),
        "regions.field_average_pressure": ScenarioQuantityPolicy(
            TolerancePolicy(absolute=2.0e5), unit="Pa"),
        "wells.bhp": ScenarioQuantityPolicy(
            TolerancePolicy(absolute=2.0e5), unit="Pa"),
        "wells.qWs": ScenarioQuantityPolicy(TolerancePolicy(absolute=1.0e-3)),
        "wells.qOs": ScenarioQuantityPolicy(TolerancePolicy(absolute=1.0e-3)),
        "wells.qGs": ScenarioQuantityPolicy(TolerancePolicy(absolute=1.0)),
        "conservation.finite_state": ScenarioQuantityPolicy(
            TolerancePolicy(absolute=0.0)),
    }


def run(matrix_path: Path, output: Path, evidence: Path, workers: int):
    matrix = _load_matrix(matrix_path)
    report = LocalEnsembleExecutor(output / "members", max_workers=workers).run(
        matrix, runner=RUNNER)
    if report.failed:
        failures = "; ".join(
            f"{value.coordinate.variant_name}: {value.error_type}: {value.error_message}"
            for value in report.failed)
        raise RuntimeError(f"SPE1 ensemble has failed members: {failures}")

    results = {
        outcome.coordinate.variant_name:
        MemberResultStore(outcome.store_path).lazy()
        for outcome in report.completed
    }
    comparisons = {}
    for name in ("explicit", "aim"):
        physics = compare_scenarios(
            results["fim"], results[name], policies=_policies(),
            comparison_group="time_integration")
        performance = compare_performance(
            results["fim"], results[name],
            tolerances={
                "simulation_seconds": TolerancePolicy(relative=10.0, absolute=1.0),
                "total_iterations": TolerancePolicy(relative=10.0, absolute=100.0),
            })
        comparisons[name] = {
            "physics": physics.to_dict(),
            "performance": performance.to_dict(),
        }

    round_trips = {}
    export_root = output / "exports"
    for name, result in results.items():
        member_root = export_root / name
        zarr_path = write_zarr(
            member_root / "result.zarr", NormalizedResultDataset(result))
        csv_path = write_csv_summary(member_root / "summary.csv", result)
        parquet_path = write_parquet_summary(member_root / "summary.parquet", result)
        loaded = read_zarr(zarr_path, lazy=False)
        checks = {}
        for category in ("fields", "regions", "wells", "convergence", "conservation"):
            for quantity in result.names(category):
                key = f"{category}__{quantity}"
                checks[key] = bool(np.array_equal(
                    loaded[key].values, result.array(category, quantity)))
        loaded.close()
        csv_frame = pd.read_csv(csv_path)
        parquet_frame = pd.read_parquet(parquet_path)
        metadata_columns = [name for name in csv_frame.columns if name != "value"]
        metadata_equal = bool(csv_frame[metadata_columns].equals(
            parquet_frame[metadata_columns]))
        csv_parquet_maximum_absolute_error = float(np.max(np.abs(
            csv_frame["value"].to_numpy()-parquet_frame["value"].to_numpy())))
        value_scale = max(1.0, float(np.max(np.abs(
            parquet_frame["value"].to_numpy()))))
        csv_parquet_within_one_ulp_scale = bool(
            csv_parquet_maximum_absolute_error <= np.finfo(float).eps*value_scale)
        round_trips[name] = {
            "zarr_arrays_bit_identical": all(checks.values()),
            "zarr_array_checks": checks,
            "csv_parquet_metadata_equal": metadata_equal,
            "csv_parquet_maximum_absolute_error": csv_parquet_maximum_absolute_error,
            "csv_parquet_within_one_ulp_scale": csv_parquet_within_one_ulp_scale,
            "csv_sha256": _sha256(csv_path),
            "parquet_sha256": _sha256(parquet_path),
            "tabular_rows": int(len(csv_frame)),
        }

    artifact = {
        "schema_version": 1,
        "work_packages": ["M3.3", "M3.4", "M3.5"],
        "matrix_id": matrix.matrix_id,
        "ensemble": report.to_dict(),
        "aggregation": LocalEnsembleExecutor.aggregate(report),
        "comparisons": comparisons,
        "format_round_trips": round_trips,
        "acceptance": {
            "all_members_completed": len(report.completed) == 3,
            "all_physics_comparisons_passed": all(
                value["physics"]["passed"] for value in comparisons.values()),
            "all_format_round_trips_passed": all(
                value["zarr_arrays_bit_identical"]
                and value["csv_parquet_metadata_equal"]
                and value["csv_parquet_within_one_ulp_scale"]
                for value in round_trips.values()),
        },
    }
    evidence.parent.mkdir(parents=True, exist_ok=True)
    temporary = evidence.with_suffix(evidence.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(evidence)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path,
        default=Path("docs/research_studio/experiments/variant_matrices.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/research_studio/spe1_ensemble"))
    parser.add_argument(
        "--evidence", type=Path,
        default=Path("docs/research_studio/experiments/spe1_ensemble_qualification.json"))
    parser.add_argument("--workers", type=int, default=3)
    arguments = parser.parse_args()
    artifact = run(arguments.matrix, arguments.output,
                   arguments.evidence, arguments.workers)
    print(json.dumps(artifact["acceptance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
