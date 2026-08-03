"""Real MRST/PyMRST qualification graph regression."""

import json
from pathlib import Path

from pymrst.research_studio.validation import EvidenceGraph
from scripts.qualify_research_studio_phase64 import qualify


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/"examples/validation/ref_p64_run2d_water_fixedstress.mat"
ARTIFACT = ROOT/"docs/research_studio/qualification/phase64_evidence_graph.json"


def test_phase64_matlab_reference_generates_eight_accepted_real_edges():
    graph, summary = qualify(REFERENCE)
    assert summary["passed"]
    assert len(graph.edges) == 8
    assert all(edge.accepted and edge.converged for edge in graph.edges)
    assert {edge.source for edge in graph.edges} == {"mrst_phase64"}
    assert {edge.target for edge in graph.edges} == {"pymrst_phase64"}
    assert summary["edges"]["well_water_rate"]["maximum_absolute_error"] == 0.0
    assert summary["edges"]["pressure"]["maximum_absolute_error"] < 2.0


def test_persisted_phase64_graph_matches_current_matlab_fixture_and_schema():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["artifact"] == "pymrst_qualification_evidence_graph"
    assert payload["artifact_schema_version"] == 1
    graph = EvidenceGraph.from_dict(payload["graph"])
    assert len(graph.edges) == 8
    _, current = qualify(REFERENCE)
    assert payload["summary"] == current
