"""M3.4 field, region, well, convergence, conservation and performance comparison."""

from dataclasses import replace

import numpy as np

from pymrst.research_studio.experiments import (
    ExplicitVariantMatrix, LocalEnsembleExecutor, ScenarioQuantityPolicy,
    VariantDefinition, compare_performance, compare_scenarios,
    materialize_variant_matrix,
)
from pymrst.research_studio.validation import TolerancePolicy, identity_mapping
from scripts.build_research_studio_experiment_registry import spe1_time_integration_experiment


RUNNER = "pymrst.research_studio.experiments.runners:deterministic_manufactured_member"


def _results(tmp_path):
    base = replace(spe1_time_integration_experiment(), variants=())
    matrix = ExplicitVariantMatrix((
        VariantDefinition("reference", parameters={"test.multiplier": 1.0},
                          comparison_group="rates"),
        VariantDefinition("candidate", parameters={"test.multiplier": 2.0},
                          comparison_group="rates"),
    ))
    materialized = materialize_variant_matrix(base, matrix)
    report = LocalEnsembleExecutor(
        tmp_path, max_workers=1, memory_per_worker_bytes=1024).run(
            materialized, runner=RUNNER)
    return [__import__("pymrst.research_studio.experiments", fromlist=["MemberResultStore"])
            .MemberResultStore(value.store_path).lazy() for value in report.completed]


def test_cross_category_scenario_comparison_quantifies_expected_differences(tmp_path):
    reference, candidate = _results(tmp_path)
    policies = {
        "fields.pressure": ScenarioQuantityPolicy(TolerancePolicy(absolute=1.0)),
        "fields.saturation": ScenarioQuantityPolicy(TolerancePolicy(absolute=0.0)),
        "regions.field_pressure": ScenarioQuantityPolicy(TolerancePolicy(absolute=1.0)),
        "wells.producer_rate": ScenarioQuantityPolicy(TolerancePolicy(absolute=1e-12)),
        "convergence.iterations": ScenarioQuantityPolicy(TolerancePolicy(absolute=0.0)),
        "conservation.mass_balance": ScenarioQuantityPolicy(TolerancePolicy(absolute=1e-11)),
    }
    comparison = compare_scenarios(
        reference, candidate, policies=policies, comparison_group="rates")
    assert not comparison.passed
    assert not comparison.results["fields.pressure"].passed
    assert comparison.results["fields.saturation"].passed
    assert not comparison.results["regions.field_pressure"].passed
    assert not comparison.results["wells.producer_rate"].passed
    assert comparison.results["convergence.iterations"].passed
    assert comparison.results["conservation.mass_balance"].passed
    assert comparison.comparison_group == "rates"


def test_performance_comparison_is_separate_from_physics(tmp_path):
    reference, candidate = _results(tmp_path)
    result = compare_performance(
        reference, candidate,
        tolerances={"runtime_seconds": TolerancePolicy(absolute=0.001)})
    assert not result.passed
    assert result.results["performance.runtime_seconds"].maximum_absolute_error == 0.01


def test_explicit_spatial_mapping_can_reverse_candidate_entity_order(tmp_path):
    reference, candidate = _results(tmp_path)
    # Rewrite the candidate pressure order only, preserving all other categories.
    from pymrst.research_studio.experiments import MemberPayload, MemberResultStore
    store = MemberResultStore(candidate.store.path)
    payload = MemberPayload(
        candidate.times,
        fields={"pressure": reference.array("fields", "pressure")[:, ::-1],
                "saturation": candidate.array("fields", "saturation")},
        wells={name: candidate.array("wells", name) for name in candidate.names("wells")},
        regions={name: candidate.array("regions", name) for name in candidate.names("regions")},
        convergence={name: candidate.array("convergence", name)
                     for name in candidate.names("convergence")},
        conservation={name: candidate.array("conservation", name)
                      for name in candidate.names("conservation")},
        performance=candidate.record["performance"])
    coordinate = candidate.record["coordinate"]
    from pymrst.research_studio.experiments.variants import VariantCoordinate
    store.commit(payload, coordinate=VariantCoordinate.from_dict(coordinate),
                 seed=candidate.record["seed"], runner=RUNNER, elapsed_seconds=0.0)
    remapped = store.lazy()
    mapping = identity_mapping(tuple(range(4)), tuple(reversed(range(4))), kind="cell")
    result = compare_scenarios(
        reference, remapped,
        policies={"fields.pressure": ScenarioQuantityPolicy(
            TolerancePolicy(absolute=0.0))},
        mappings={"fields.pressure": mapping})
    assert result.passed


def test_time_alignment_interpolates_candidate_to_reference_times(tmp_path):
    from pymrst.research_studio.experiments import MemberPayload, MemberResultStore
    from pymrst.research_studio.experiments.variants import VariantCoordinate
    reference, candidate = _results(tmp_path)
    store = MemberResultStore(candidate.store.path)
    values = reference.array("fields", "pressure")
    payload = MemberPayload(
        np.array([1.0, 4.0]), fields={"pressure": values[[0, 2]]},
        performance={})
    store.commit(payload, coordinate=VariantCoordinate.from_dict(candidate.record["coordinate"]),
                 seed=candidate.record["seed"], runner=RUNNER, elapsed_seconds=0.0)
    result = compare_scenarios(
        reference, store.lazy(), policies={"fields.pressure": ScenarioQuantityPolicy(
            TolerancePolicy(absolute=1e-8))})
    assert result.passed
