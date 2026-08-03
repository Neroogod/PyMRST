"""M3.3 bounded process execution, isolation, stores, checkpoint and resume."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from pymrst.research_studio.experiments import (
    ExplicitVariantMatrix, LocalEnsembleExecutor, MemberResultStore,
    VariantDefinition, deterministic_member_seed, materialize_variant_matrix,
)
from scripts.build_research_studio_experiment_registry import (
    spe1_time_integration_experiment,
)


MANUFACTURED_RUNNER = (
    "pymrst.research_studio.experiments.runners:deterministic_manufactured_member")
CHECKPOINT_RUNNER = "pymrst.research_studio.experiments.runners:checkpoint_resume_member"


def _materialized(*variants):
    base = replace(spe1_time_integration_experiment(), variants=())
    matrix = ExplicitVariantMatrix(tuple(variants))
    return materialize_variant_matrix(base, matrix)


def test_bounded_process_ensemble_isolates_failure_and_writes_member_stores(tmp_path):
    materialized = _materialized(
        VariantDefinition("low", parameters={"test.multiplier": 0.5}),
        VariantDefinition("broken", parameters={"test.fail": True}),
        VariantDefinition("high", parameters={"test.multiplier": 2.0}),
    )
    executor = LocalEnsembleExecutor(tmp_path, max_workers=2,
                                     memory_per_worker_bytes=1024)
    report = executor.run(materialized, runner=MANUFACTURED_RUNNER)
    assert report.effective_workers <= 2
    assert len(report.completed) == 2 and len(report.failed) == 1
    assert report.failed[0].error_type == "RuntimeError"
    for outcome in report.completed:
        store = MemberResultStore(outcome.store_path)
        assert store.complete(outcome.coordinate, MANUFACTURED_RUNNER)
        lazy = store.lazy()
        assert lazy.array("fields", "pressure").shape == (3, 4)
        assert lazy.array("fields", "pressure", time_slice=0).shape == (4,)
        assert lazy.record["coordinate"] == outcome.coordinate.to_dict()
    aggregate = executor.aggregate(report)
    assert aggregate["completed_members"] == 2
    assert aggregate["failed_members"] == 1
    assert aggregate["peak_memory_bytes_mean"] == 1024.0


def test_completed_and_failed_members_resume_without_reexecution(tmp_path):
    materialized = _materialized(
        VariantDefinition("ok", parameters={"test.multiplier": 1.0}),
        VariantDefinition("bad", parameters={"test.fail": True}),
    )
    executor = LocalEnsembleExecutor(tmp_path, max_workers=2,
                                     memory_per_worker_bytes=1024)
    first = executor.run(materialized, runner=MANUFACTURED_RUNNER)
    mtimes = {outcome.coordinate.member_id:
              MemberResultStore(outcome.store_path).record_path.stat().st_mtime_ns
              for outcome in first.outcomes}
    second = executor.run(materialized, runner=MANUFACTURED_RUNNER,
                          retry_failed=False)
    assert all(outcome.resumed for outcome in second.outcomes)
    assert [outcome.status for outcome in second.outcomes] == ["completed", "failed"]
    assert mtimes == {outcome.coordinate.member_id:
                      MemberResultStore(outcome.store_path).record_path.stat().st_mtime_ns
                      for outcome in second.outcomes}


def test_failed_member_receives_checkpoint_on_retry(tmp_path):
    materialized = _materialized(
        VariantDefinition("checkpoint", parameters={"test.mode": "resume"}),)
    executor = LocalEnsembleExecutor(tmp_path, max_workers=1,
                                     memory_per_worker_bytes=1024)
    first = executor.run(materialized, runner=CHECKPOINT_RUNNER)
    assert len(first.failed) == 1
    second = executor.run(materialized, runner=CHECKPOINT_RUNNER, retry_failed=True)
    assert len(second.completed) == 1
    store = MemberResultStore(second.completed[0].store_path).lazy()
    assert store.record["metadata"]["previous_error"] == "RuntimeError"


def test_member_seeds_are_stable_unique_and_experiment_seed_sensitive():
    materialized = _materialized(
        VariantDefinition("a", parameters={"test.multiplier": 1.0}),
        VariantDefinition("b", parameters={"test.multiplier": 2.0}),)
    seeds = [deterministic_member_seed(materialized.definition.random_seed, coordinate)
             for coordinate in materialized.coordinates]
    assert seeds == [deterministic_member_seed(materialized.definition.random_seed, coordinate)
                     for coordinate in materialized.coordinates]
    assert len(set(seeds)) == 2
    assert seeds != [deterministic_member_seed(99, coordinate)
                     for coordinate in materialized.coordinates]


def test_worker_count_is_capped_by_memory(monkeypatch, tmp_path):
    import pymrst.research_studio.experiments.ensemble as ensemble
    monkeypatch.setattr(ensemble, "available_memory_bytes", lambda: 1000)
    executor = LocalEnsembleExecutor(tmp_path, max_workers=8,
                                     memory_per_worker_bytes=400,
                                     memory_fraction=0.8)
    assert executor.max_workers == 2
