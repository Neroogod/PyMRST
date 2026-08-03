#!/usr/bin/env python3
"""Build reproducible explicit, Cartesian, and sampled M3.2 artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from pymrst.research_studio.experiments import (
    CartesianVariantMatrix, ComponentAxis, DistributionSpec,
    ExperimentRegistry, ExplicitVariantMatrix, ParameterAxis, SampledParameter,
    SampledVariantMatrix, materialize_variant_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/research_studio/experiments/variant_matrices.json"
EXPERIMENT_REGISTRY = ROOT / "docs/research_studio/experiments/registry.json"


def build_variant_matrices():
    registry = ExperimentRegistry.load(EXPERIMENT_REGISTRY)
    explicit_definition = registry.get(registry.experiment_ids[0])
    base = replace(explicit_definition, variants=())
    matrices = (
        ExplicitVariantMatrix(explicit_definition.variants),
        CartesianVariantMatrix(
            parameter_axes=(ParameterAxis("solver.max_iterations", (8, 12, 16)),),
            component_axes=(ComponentAxis("linear_solver", ("direct", "cpr")),),
            comparison_group="solver_configuration",
        ),
        SampledVariantMatrix(
            parameters=(
                SampledParameter("rock.perm_multiplier", DistributionSpec(
                    "log_uniform", {"low": 0.25, "high": 4.0})),
                SampledParameter("well.rate_multiplier", DistributionSpec(
                    "uniform", {"low": 0.8, "high": 1.2})),
            ),
            sample_count=8, seed=20240803,
            comparison_group="property_uncertainty",
        ),
    )
    return tuple(materialize_variant_matrix(
        explicit_definition if isinstance(matrix, ExplicitVariantMatrix) else base,
        matrix,
    ) for matrix in matrices)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    materialized = build_variant_matrices()
    payload = {"schema_version": 1,
               "matrices": [matrix.to_dict() for matrix in materialized]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    for matrix in materialized:
        print(matrix.matrix_id, len(matrix.coordinates))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
