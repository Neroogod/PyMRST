"""Physics-versus-Victory qualification evidence regression."""

import json
from pathlib import Path

from pymrst.research_studio.validation import EvidenceGraph, EvidenceNodeKind
from scripts.qualify_victory_surrogate import qualify


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/research_studio/qualification/victory_surrogate_evidence_graph.json"


def test_real_victory_matrix_preserves_negative_safety_conclusion():
    graph, summary = qualify()
    assert summary["qualification_completed"]
    assert summary["physics_parity_accepted"]
    assert not summary["surrogate_accuracy_accepted"]
    assert not summary["production_surrogate_accepted"]
    assert not summary["stability_classifier_qualified"]
    assert summary["speedup"] > 10.0
    assert len(graph.nodes) == 12
    assert len(graph.edges) == 29
    assert summary["accepted_edges"] == sum(edge.accepted for edge in graph.edges)


def test_persisted_graph_discloses_phase_flag_dependency_and_failed_gates():
    payload = json.loads(ARTIFACT.read_text())
    graph = EvidenceGraph.from_dict(payload["graph"])
    nodes = {node.node_id: node for node in graph.nodes}
    assert nodes["bundled_reference"].kind is EvidenceNodeKind.MANUFACTURED
    assert nodes["bundled_reference"].metadata["phase_flags_are_model_inputs"]
    surrogate = [edge for edge in graph.edges
                 if edge.evidence_class == "surrogate_accuracy_gate"]
    assert len(surrogate) == 7
    assert any(edge.accepted for edge in surrogate)
    assert any(not edge.accepted for edge in surrogate)
    safety = next(edge for edge in graph.edges
                  if edge.edge_id == "victory_physical_safety")
    assert not safety.accepted
    assert safety.metrics["log_fugacity_rmse"] > 1.0


def test_error_atlas_covers_pressure_phase_and_uncertainty():
    summary = json.loads(ARTIFACT.read_text())["summary"]
    atlas = summary["error_atlas"]
    assert len(atlas["pressure_quintiles"]) == 5
    assert set(atlas["reference_phase"]) == {"gas", "liquid", "two-phase"}
    assert len(atlas["ensemble_uncertainty_quartiles"]) == 4
    assert sum(item["count"] for item in atlas["pressure_quintiles"]) == 10_000

