"""Direct/GMRES/CPR SPE1 qualification regression."""

import json
from pathlib import Path

from pymrst.research_studio.validation import EvidenceGraph, Independence
from scripts.qualify_blackoil_linear_solvers import qualify


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/research_studio/qualification/blackoil_linear_solver_evidence_graph.json"


def test_linear_solver_matrix_regenerates_with_true_linear_diagnostics():
    graph, summary = qualify()
    assert summary["passed"]
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 22
    assert summary["diagnostics"]["direct"]["nonlinear_iterations"] == [4, 4]
    assert summary["diagnostics"]["gmres_ilu"]["total_linear_iterations"] > 50
    assert summary["diagnostics"]["cpr"]["total_linear_iterations"] > 30
    for values in summary["diagnostics"].values():
        assert values["maximum_true_relative_residual"] < 1e-7
    assert summary["cross_solver"]["gmres_ilu"]["pressure"]["maximum_absolute_error"] < 0.01
    assert summary["cross_solver"]["cpr"]["pressure"]["maximum_absolute_error"] < 0.01
    numerical = [edge for edge in graph.edges if edge.evidence_class != "solver_work"]
    assert len(numerical) == 21
    assert all(edge.accepted and edge.converged for edge in numerical)


def test_persisted_linear_solver_graph_declares_shared_system_limit():
    graph = EvidenceGraph.from_dict(json.loads(ARTIFACT.read_text())["graph"])
    cross = [edge for edge in graph.edges if edge.edge_id.startswith("direct_vs_")]
    assert len(cross) == 14
    assert all(edge.independence is Independence.SHARED_KERNEL for edge in cross)
    assert all("assembled Jacobian" in edge.shared_kernels for edge in cross)
