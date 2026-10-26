"""M3.1 deterministic experiment definition and registry tests."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pymrst.research_studio.experiments import (
    CaseReference, DatasetReference, ExecutionProfile, ExperimentDefinition,
    ExperimentRegistry, OutputRequest, OwnerReference, ValidationRequirement,
    VariantDefinition, canonical_json, explicit_variant_coordinates,
)
from scripts.build_research_studio_experiment_registry import build_registry


CASE_HASH = "1" * 64


def _definition(dataset, *, name="SPE1 solver study", hypothesis="AIM preserves FIM accuracy"):
    return ExperimentDefinition(
        name=name, hypothesis=hypothesis,
        description="Compare time-integration formulations under fixed physics.",
        base_case=CaseReference(
            "case_spe1",
            OwnerReference("pymrst.modules.ad_blackoil.examples:setup_spe1",
                           {"report_steps": 2}), CASE_HASH),
        model=OwnerReference("pymrst.modules.ad_blackoil:GenericBlackOilModelAD",
                             {"disgas": True, "vapoil": False}),
        schedule=OwnerReference("pymrst.modules.ad_core:Schedule",
                                {"control_policy": "deck"}),
        variants=(
            VariantDefinition("fim", components={"time_integration": "fim"},
                              comparison_group="integration"),
            VariantDefinition("aim", parameters={"solver.aim_threshold": 0.05},
                              components={"time_integration": "aim"},
                              comparison_group="integration"),
        ),
        datasets=(dataset,), execution_profile=ExecutionProfile(
            "quick", {"workers": 1, "checkpoint_every_report": True}),
        validation=(ValidationRequirement(
            "mrst_pressure", {"absolute_tolerance_pa": 100.0}),),
        outputs=(OutputRequest("reservoir", ("pressure", "sW", "sG")),
                 OutputRequest("wells", ("bhp", "qOs"))),
        publication_artifacts=("evidence_graph", "comparison_report"),
        random_seed=42, tags=("blackoil", "time-integration"),
    )


def test_definition_json_round_trip_preserves_both_identities(tmp_path):
    deck = tmp_path / "SPE1.DATA"
    deck.write_text("RUNSPEC\nDIMENS\n 10 10 3 /\n", encoding="utf-8")
    definition = _definition(DatasetReference.from_path("spe1_deck", deck))
    restored = ExperimentDefinition.from_json(definition.to_json())
    assert restored == definition
    assert restored.experiment_id == definition.experiment_id
    assert restored.execution_fingerprint == definition.execution_fingerprint
    assert json.loads(definition.to_json())["schema_version"] == 1


def test_dataset_location_does_not_change_identity_but_content_does(tmp_path):
    first = tmp_path / "a.DATA"
    second = tmp_path / "elsewhere" / "b.DATA"
    second.parent.mkdir()
    first.write_bytes(b"same deck")
    second.write_bytes(b"same deck")
    a = _definition(DatasetReference.from_path("deck", first))
    b = _definition(DatasetReference.from_path("deck", second))
    assert a.experiment_id == b.experiment_id
    assert a.execution_fingerprint == b.execution_fingerprint
    second.write_bytes(b"changed deck")
    c = _definition(DatasetReference.from_path("deck", second))
    assert c.experiment_id != a.experiment_id
    assert c.execution_fingerprint != a.execution_fingerprint


def test_recorded_dataset_content_is_verified_before_execution(tmp_path):
    deck = tmp_path / "case.DATA"
    deck.write_bytes(b"original")
    definition = _definition(DatasetReference.from_path("deck", deck))
    definition.verify_datasets()
    deck.write_bytes(b"modified")
    with pytest.raises(ValueError, match="content changed"):
        definition.verify_datasets()


def test_prose_changes_scientific_identity_but_not_execution_cache_key(tmp_path):
    deck = tmp_path / "case.DATA"
    deck.write_bytes(b"case")
    original = _definition(DatasetReference.from_path("deck", deck))
    revised = replace(original, hypothesis="A revised scientific hypothesis")
    assert revised.experiment_id != original.experiment_id
    assert revised.execution_fingerprint == original.execution_fingerprint


def test_mapping_order_is_canonical_and_nonfinite_values_are_refused():
    assert canonical_json({"b": 2, "a": {"d": 4, "c": 3}}) == \
        canonical_json({"a": {"c": 3, "d": 4}, "b": 2})
    assert canonical_json(-0.0) == "0.0"
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"bad": float("nan")})
    with pytest.raises(TypeError, match="JSON-safe"):
        canonical_json({"bad": object()})


def test_explicit_variant_coordinates_and_member_ids_are_deterministic(tmp_path):
    deck = tmp_path / "case.DATA"
    deck.write_bytes(b"case")
    definition = _definition(DatasetReference.from_path("deck", deck))
    first = explicit_variant_coordinates(definition)
    second = explicit_variant_coordinates(definition)
    assert first == second
    assert [item.variant_name for item in first] == ["fim", "aim"]
    assert len({item.member_id for item in first}) == 2
    assert all(item.experiment_id == definition.experiment_id for item in first)


def test_registry_is_idempotent_persistent_and_refuses_name_ambiguity(tmp_path):
    deck = tmp_path / "case.DATA"
    deck.write_bytes(b"case")
    definition = _definition(DatasetReference.from_path("deck", deck))
    registry = ExperimentRegistry()
    identity = registry.register(definition)
    assert registry.register(definition) == identity
    assert registry.by_name(definition.name) == definition
    path = tmp_path / "experiments.json"
    registry.save(path)
    restored = ExperimentRegistry.load(path)
    assert restored.get(identity) == definition
    with pytest.raises(ValueError, match="already identifies"):
        restored.register(replace(definition, hypothesis="different"))


def test_definition_rejects_duplicate_named_outputs_and_empty_variants(tmp_path):
    deck = tmp_path / "case.DATA"
    deck.write_bytes(b"case")
    definition = _definition(DatasetReference.from_path("deck", deck))
    with pytest.raises(ValueError, match="output names"):
        replace(definition, outputs=(definition.outputs[0], definition.outputs[0]))
    with pytest.raises(ValueError, match="must change"):
        VariantDefinition("empty")


def test_persisted_registry_matches_the_repository_spe1_definition():
    root = Path(__file__).resolve().parents[2]
    path = root / "docs/research_studio/experiments/registry.json"
    persisted = ExperimentRegistry.load(path)
    expected = build_registry()
    assert persisted.to_dict() == expected.to_dict()
    definition = persisted.get(persisted.experiment_ids[0])
    assert len(explicit_variant_coordinates(definition)) == 3
    assert definition.datasets[0].sha256 == definition.base_case.content_hash
