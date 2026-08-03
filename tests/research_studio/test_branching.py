"""M3.6 checkpoint ancestry, compatibility and changed-input branching."""

from dataclasses import replace
import json

import numpy as np
import pytest

from pymrst.research_studio.experiments import (
    BranchDefinition, ExplicitVariantMatrix, IncompatibleBranch,
    LocalEnsembleExecutor, MemberResultStore, VariantDefinition,
    execute_branch, materialize_variant_matrix, verify_branch_lineage,
)
from scripts.build_research_studio_experiment_registry import (
    spe1_time_integration_experiment,
)

BASE_RUNNER = "pymrst.research_studio.experiments.runners:deterministic_manufactured_member"
BRANCH_RUNNER = "pymrst.research_studio.experiments.runners:deterministic_branch_member"


def _parent(tmp_path):
    definition = replace(spe1_time_integration_experiment(), variants=())
    matrix = ExplicitVariantMatrix((VariantDefinition(
        "parent", parameters={"test.multiplier": 1.0}),))
    report = LocalEnsembleExecutor(tmp_path / "base", max_workers=1,
                                   memory_per_worker_bytes=1024).run(
        materialize_variant_matrix(definition, matrix), runner=BASE_RUNNER)
    return MemberResultStore(report.completed[0].store_path)


def test_branch_advances_selected_parent_state_and_records_every_change(tmp_path):
    parent = _parent(tmp_path)
    member_id = parent.record()["coordinate"]["member_id"]
    definition = BranchDefinition(
        "higher-rate", member_id, parent_time_index=1,
        controls={"rate": 3.0}, properties={"mobility": 2.0},
        components={"linear_solver": "gmres"}, required_fields=("pressure",),
        comparison_group="control-study")
    outcome = execute_branch(parent, definition, output=tmp_path / "branches",
                             runner=BRANCH_RUNNER, seed=17)
    child = MemberResultStore(outcome.store_path).lazy()
    expected = parent.lazy().array("fields", "pressure", time_slice=1)
    np.testing.assert_array_equal(child.array("fields", "pressure")[0], expected+6.0)
    assert child.times[0] == parent.lazy().times[1]+1.0
    lineage = child.record["metadata"]["branch_lineage"]
    assert lineage["changes"]["controls"] == {"rate": 3.0}
    assert lineage["changes"]["properties"] == {"mobility": 2.0}
    assert verify_branch_lineage(outcome) == definition


def test_branch_identity_is_stable_and_changes_with_scientific_input(tmp_path):
    parent = _parent(tmp_path)
    member_id = parent.record()["coordinate"]["member_id"]
    first = BranchDefinition("b", member_id, controls={"rate": 1.0})
    same = BranchDefinition.from_dict(first.to_dict())
    changed = BranchDefinition("b", member_id, controls={"rate": 2.0})
    assert first.branch_id == same.branch_id
    assert first.branch_id != changed.branch_id


def test_wrong_parent_and_missing_required_state_are_refused(tmp_path):
    parent = _parent(tmp_path)
    with pytest.raises(IncompatibleBranch, match="does not identify"):
        execute_branch(parent, BranchDefinition("b", "member_wrong"),
                       output=tmp_path, runner=BRANCH_RUNNER)
    member_id = parent.record()["coordinate"]["member_id"]
    with pytest.raises(IncompatibleBranch, match="missing fields"):
        execute_branch(parent, BranchDefinition(
            "b", member_id, required_fields=("temperature",)),
            output=tmp_path, runner=BRANCH_RUNNER)


def test_tampered_branch_identity_is_rejected(tmp_path):
    parent = _parent(tmp_path)
    definition = BranchDefinition(
        "b", parent.record()["coordinate"]["member_id"], controls={"rate": 1.0})
    value = definition.to_dict()
    value["branch_id"] = "branch_tampered"
    with pytest.raises(ValueError, match="identity mismatch"):
        BranchDefinition.from_dict(value)
