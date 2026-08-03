"""Natural-variable versus overall-composition qualification regression."""

import json
from pathlib import Path

from pymrst.research_studio.validation import EvidenceGraph, Independence
from scripts.qualify_compositional_natural_overall import qualify


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/research_studio/qualification/compositional_natural_overall_evidence_graph.json"


def test_compositional_matrix_regenerates_with_all_numerical_edges_passing():
    graph, summary = qualify()
    assert summary["passed"]
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 19
    assert summary["overall"]["iterations"] == [3, 4]
    assert summary["natural"]["iterations"] == [3, 4]
    assert summary["cross_formulation"]["pressure"]["maximum_absolute_error"] < 0.01
    assert summary["cross_formulation"]["components"]["maximum_absolute_error"] < 1.0e-9
    numerical = [edge for edge in graph.edges if edge.evidence_class != "solver_work"]
    assert len(numerical) == 18
    assert all(edge.accepted and edge.converged for edge in numerical)


def test_persisted_compositional_graph_declares_independence_limits():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    graph = EvidenceGraph.from_dict(payload["graph"])
    cross = [edge for edge in graph.edges if edge.edge_id.startswith("cross_")]
    assert len(cross) == 7
    assert all(edge.independence is Independence.SHARED_KERNEL for edge in cross)
    assert all(edge.shared_kernels for edge in cross)
    mrst = [edge for edge in graph.edges if edge.source.startswith("mrst_")]
    assert len(mrst) == 12
    assert all(edge.independence is Independence.PARTIAL for edge in mrst)
