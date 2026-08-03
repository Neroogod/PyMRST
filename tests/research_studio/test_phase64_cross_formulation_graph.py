"""Phase 64/63 fixed-stress versus fully coupled qualification regression."""

import json
from pathlib import Path

from pymrst.research_studio.validation import EvidenceGraph, Independence
from scripts.qualify_phase64_cross_formulation import qualify


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT/"docs/research_studio/qualification/phase64_cross_formulation_evidence_graph.json"


def test_cross_formulation_matrix_regenerates_with_all_numerical_edges_passing():
    graph, summary = qualify()
    assert summary["passed"]
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 25
    assert summary["fixed_stress"]["iterations"] == [5, 6]
    assert summary["fully_coupled"]["iterations"] == [2, 2]
    assert summary["cross_formulation"]["pressure"]["maximum_absolute_error"] < 3.5
    assert summary["cross_formulation"]["well_water_rate"]["maximum_absolute_error"] == 0.0
    assert summary["fully_coupled_vs_mrst"]["pressure"]["maximum_absolute_error"] < 0.01
    numerical = [edge for edge in graph.edges if edge.evidence_class != "solver_work"]
    assert len(numerical) == 24
    assert all(edge.accepted and edge.converged for edge in numerical)


def test_persisted_cross_formulation_graph_declares_shared_kernel_limits():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    graph = EvidenceGraph.from_dict(payload["graph"])
    cross = [edge for edge in graph.edges if edge.edge_id.startswith("cross_")]
    assert len(cross) == 9
    assert all(edge.independence is Independence.SHARED_KERNEL for edge in cross)
    assert all(edge.shared_kernels for edge in cross)
    mrst = [edge for edge in graph.edges if edge.source.startswith("mrst_")]
    assert len(mrst) == 16
    assert all(edge.independence is Independence.PARTIAL for edge in mrst)
