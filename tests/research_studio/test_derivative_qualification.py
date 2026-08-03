"""AD-versus-finite-difference qualification regression."""

import json
from pathlib import Path

import numpy as np

from pymrst.research_studio.validation import (
    EvidenceGraph, EvidenceNodeKind, qualify_directional_derivative,
)
from scripts.qualify_ad_finite_difference import qualify


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/research_studio/qualification/ad_finite_difference_evidence_graph.json"


def test_directional_checker_handles_mixed_units_and_nonlinearity():
    point = np.array([2.0e7, 0.3])
    scales = np.array([1.0e5, 0.1])

    def function(x):
        return np.array([np.exp(1e-8 * x[0]) * x[1]**2,
                         x[0] * x[1]])

    jacobian = np.array([
        [1e-8 * np.exp(1e-8 * point[0]) * point[1]**2,
         2 * np.exp(1e-8 * point[0]) * point[1]],
        [point[1], point[0]],
    ])
    result = qualify_directional_derivative(
        jacobian, function, point, scales=scales,
        relative_tolerance=1e-7, component_tolerance=1e-6,
    )
    assert result.passed
    assert result.maximum_relative_l2_error < 1e-7


def test_matrix_regenerates_and_every_derivative_edge_passes():
    graph, summary = qualify()
    assert summary["passed"]
    assert len(graph.edges) == 6
    assert all(edge.accepted for edge in graph.edges)
    assert summary["residual_derivatives"]["full_residual"]["directions"] >= 16
    assert summary["objective_gradient"]["maximum_relative_error"] < 1e-5


def test_persisted_graph_declares_finite_difference_limits():
    payload = json.loads(ARTIFACT.read_text())
    graph = EvidenceGraph.from_dict(payload["graph"])
    sources = {node.node_id: node for node in graph.nodes}
    assert all(sources[edge.source].kind is EvidenceNodeKind.FINITE_DIFFERENCE
               for edge in graph.edges)
    assert all("value implementation" in " ".join(edge.shared_kernels)
               or edge.edge_id == "objective_gradient_adjoint_vs_fd"
               for edge in graph.edges)

