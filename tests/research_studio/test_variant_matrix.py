"""M3.2 explicit, Cartesian, and sampled variant matrices."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pymrst.research_studio.experiments import (
    CartesianVariantMatrix, ComponentAxis, DistributionSpec,
    ExplicitVariantMatrix, MaterializedVariantMatrix, ParameterAxis,
    SampledParameter, SampledVariantMatrix, materialize_variant_matrix,
    variant_matrix_from_dict,
)
from scripts.build_research_studio_experiment_registry import (
    spe1_time_integration_experiment,
)


def _base():
    return replace(spe1_time_integration_experiment(), variants=())


def test_cartesian_matrix_has_stable_product_order_and_coordinates():
    matrix = CartesianVariantMatrix(
        parameter_axes=(ParameterAxis("solver.max_iterations", (8, 12)),
                        ParameterAxis("solver.relaxation", (0.5, 1.0, 1.5))),
        component_axes=(ComponentAxis("linear_solver", ("direct", "cpr")),),
        comparison_group="solver_matrix",
    )
    result = materialize_variant_matrix(_base(), matrix)
    assert len(result.coordinates) == 12
    assert result.coordinates[0].parameters == {
        "solver.max_iterations": 8, "solver.relaxation": 0.5}
    assert result.coordinates[0].components == {"linear_solver": "direct"}
    assert result.coordinates[-1].parameters == {
        "solver.max_iterations": 12, "solver.relaxation": 1.5}
    assert result.coordinates[-1].components == {"linear_solver": "cpr"}
    assert all(item.comparison_group == "solver_matrix" for item in result.coordinates)
    assert len({item.member_id for item in result.coordinates}) == 12


def test_sampled_matrix_is_seeded_bounded_and_records_algorithm():
    matrix = SampledVariantMatrix(
        parameters=(
            SampledParameter("rock.perm_multiplier",
                             DistributionSpec("log_uniform", {"low": 0.1, "high": 10.0})),
            SampledParameter("well.rate_multiplier",
                             DistributionSpec("uniform", {"low": 0.8, "high": 1.2})),
            SampledParameter("solver.relaxation",
                             DistributionSpec("choice", {"values": [0.5, 1.0]})),
        ), sample_count=16, seed=731, comparison_group="uncertainty",
    )
    first = materialize_variant_matrix(_base(), matrix)
    second = materialize_variant_matrix(_base(), matrix)
    changed = materialize_variant_matrix(_base(), replace(matrix, seed=732))
    assert first.to_dict() == second.to_dict()
    assert first.matrix_id != changed.matrix_id
    assert [item.parameters for item in first.coordinates] != [
        item.parameters for item in changed.coordinates]
    for coordinate in first.coordinates:
        assert 0.1 <= coordinate.parameters["rock.perm_multiplier"] <= 10.0
        assert 0.8 <= coordinate.parameters["well.rate_multiplier"] <= 1.2
        assert coordinate.parameters["solver.relaxation"] in (0.5, 1.0)
        assert coordinate.comparison_group == "uncertainty"


def test_normal_sampling_is_reproducible_and_finite():
    matrix = SampledVariantMatrix(
        parameters=(SampledParameter(
            "fluid.compressibility",
            DistributionSpec("normal", {"mean": 1e-9,
                                         "standard_deviation": 1e-10})),),
        sample_count=100, seed=42,
    )
    values = [variant.parameters["fluid.compressibility"]
              for variant in matrix.expand()]
    assert all(__import__("math").isfinite(value) for value in values)
    assert abs(sum(values)/len(values) - 1e-9) < 3e-11


@pytest.mark.parametrize("matrix", [
    ExplicitVariantMatrix(spe1_time_integration_experiment().variants),
    CartesianVariantMatrix(parameter_axes=(ParameterAxis("solver.max_iterations", (8, 12)),)),
    SampledVariantMatrix(parameters=(SampledParameter(
        "rock.multiplier", DistributionSpec("uniform", {"low": 0.5, "high": 1.5})),),
        sample_count=3, seed=4),
])
def test_matrix_and_materialized_artifact_round_trip(matrix):
    restored_matrix = variant_matrix_from_dict(matrix.to_dict())
    assert restored_matrix == matrix
    base = spe1_time_integration_experiment() if isinstance(
        matrix, ExplicitVariantMatrix) else _base()
    materialized = materialize_variant_matrix(base, matrix)
    restored = MaterializedVariantMatrix.from_dict(materialized.to_dict())
    assert restored.to_dict() == materialized.to_dict()


def test_materialized_artifact_refuses_identity_or_coordinate_tampering():
    matrix = CartesianVariantMatrix(
        parameter_axes=(ParameterAxis("solver.max_iterations", (8, 12)),))
    payload = materialize_variant_matrix(_base(), matrix).to_dict()
    corrupted = json.loads(json.dumps(payload))
    corrupted["matrix_id"] = "matrix_" + "0"*64
    with pytest.raises(ValueError, match="identity mismatch"):
        MaterializedVariantMatrix.from_dict(corrupted)
    corrupted = json.loads(json.dumps(payload))
    corrupted["coordinates"][0]["parameters"]["solver.max_iterations"] = 99
    with pytest.raises(ValueError, match="coordinates are inconsistent"):
        MaterializedVariantMatrix.from_dict(corrupted)


def test_persisted_matrix_artifact_matches_current_materialization():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs/research_studio/experiments/variant_matrices.json")
                         .read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    restored = [MaterializedVariantMatrix.from_dict(value)
                for value in payload["matrices"]]
    assert [len(item.coordinates) for item in restored] == [3, 6, 8]
    assert len({item.matrix_id for item in restored}) == 3
