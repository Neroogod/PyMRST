"""Second M2.5 adapter consumer: direct poromechanics AD assembly."""

import json
from pathlib import Path

from pymrst.research_studio.validation import EvidenceGraph, Independence
from pymrst.research_studio.validation.matlab import sha256_file
from scripts.qualify_blackoil_time_integration import SPE1_FIXTURE_SPEC
from scripts.qualify_poromechanics_equations import (
    POROMECHANICS_FIXTURE_SPEC, poromechanics_reference_adapter, qualify,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/research_studio/qualification/poromechanics_equations_evidence_graph.json"


def test_poromechanics_fixture_regenerates_residual_and_jacobian_evidence():
    graph, summary = qualify()
    assert summary["passed"]
    assert len(graph.nodes) == 2 and len(graph.edges) == 3
    assert all(edge.accepted and edge.converged for edge in graph.edges)
    assert all(edge.independence is Independence.PARTIAL for edge in graph.edges)
    assert summary["comparisons"]["residual"]["maximum_absolute_error"] < 1e-13
    assert summary["comparisons"]["jacobian_displacement"]["maximum_absolute_error"] == 0.0
    assert summary["comparisons"]["jacobian_pressure"]["maximum_absolute_error"] == 0.0


def test_poromechanics_manifest_signs_reference_generator_and_mrst_owner():
    manifest = poromechanics_reference_adapter().validate()
    assert manifest.reference_sha256 == sha256_file(POROMECHANICS_FIXTURE_SPEC.reference_path)
    assert manifest.generator_sha256 == sha256_file(POROMECHANICS_FIXTURE_SPEC.generator_path)
    owner = POROMECHANICS_FIXTURE_SPEC.input_artifacts["mrst_equation_owner"]
    assert manifest.input_artifacts["mrst_equation_owner"]["sha256"] == sha256_file(owner)
    assert manifest.modules == ("ad-core", "ad-mechanics")


def test_persisted_poromechanics_graph_embeds_the_signed_manifest():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    graph = EvidenceGraph.from_dict(payload["graph"])
    manifest = poromechanics_reference_adapter().validate()
    assert payload["summary"]["fixture_manifest"] == manifest.to_dict()
    assert len(graph.edges) == 3


def test_two_adapter_consumers_are_physically_and_structurally_distinct():
    poro = POROMECHANICS_FIXTURE_SPEC
    blackoil = SPE1_FIXTURE_SPEC
    assert poro.fixture_id != blackoil.fixture_id
    assert set(poro.modules) != set(blackoil.modules)
    assert set(poro.mappings) != set(blackoil.mappings)
    assert set(poro.input_artifacts) != set(blackoil.input_artifacts)
    assert poro.solver_settings["linear_solve"].startswith("none")
    assert blackoil.solver_settings["report_steps"] == 2
