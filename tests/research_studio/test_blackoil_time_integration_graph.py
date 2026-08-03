"""SPE1 FIM/explicit/AIM qualification regression."""

import json
from pathlib import Path

from pymrst.research_studio.validation import EvidenceGraph, Independence
from scripts.qualify_blackoil_time_integration import qualify


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/research_studio/qualification/blackoil_time_integration_evidence_graph.json"


def test_time_integration_matrix_regenerates_and_quantifies_method_error():
    graph, summary = qualify()
    assert summary["passed"]
    assert len(graph.nodes) == 6
    assert len(graph.edges) == 36
    assert summary["diagnostics"]["fim"]["iterations"] == [3, 5]
    assert summary["diagnostics"]["aim"]["implicit_fraction"] == [1.0, 2/300]
    assert summary["mrst_comparisons"]["fim"]["pressure"]["maximum_absolute_error"] < 40.0
    assert summary["mrst_comparisons"]["explicit"]["pressure"]["maximum_absolute_error"] < 0.1
    assert summary["mrst_comparisons"]["explicit"]["saturation"]["maximum_absolute_error"] < 1.0e-8
    assert summary["mrst_comparisons"]["aim"]["pressure"]["maximum_absolute_error"] < 40.0
    numerical = [edge for edge in graph.edges if edge.evidence_class != "solver_work"]
    assert len(numerical) == 35
    assert all(edge.accepted and edge.converged for edge in numerical)


def test_persisted_time_integration_graph_records_mapping_and_shared_kernels():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    graph = EvidenceGraph.from_dict(payload["graph"])
    mrst = [edge for edge in graph.edges if edge.source.startswith("mrst_")]
    cross = [edge for edge in graph.edges if edge.edge_id.startswith("fim_vs_")]
    assert len(mrst) == 21
    assert len(cross) == 14
    assert all(edge.independence is Independence.PARTIAL for edge in mrst)
    assert all("well_name:v1" in edge.mappings for edge in mrst)
    assert all(edge.independence is Independence.SHARED_KERNEL for edge in cross)
    assert all(edge.shared_kernels for edge in cross)
