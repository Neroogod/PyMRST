"""Mapping and evidence-graph contract certification."""

import json

import numpy as np
import pytest

from pymrst.research_studio.validation import (
    ComparisonSeries, EntityMapping, EvidenceEdge, EvidenceGraph, EvidenceNode,
    EvidenceNodeKind, Independence, TolerancePolicy,
    active_global_cell_mapping, aggregate_regions, compare_series, component_mapping,
    edge_from_comparison, evidence_graphs, named_entity_mapping, phase_mapping,
    record_evidence_graph,
)
from pymrst.research_studio import CaseId, RunId, RunIdentity, SimulationEvidence
from pymrst.research_studio.storage import HDF5ResultStore


def test_active_global_mapping_aligns_different_active_cell_orders():
    mapping = active_global_cell_mapping([4, 8, 11], [11, 4, 99, 8])
    candidate = np.array([[110.0, 40.0, 990.0, 80.0]])
    reference = np.array([[40.0, 80.0, 110.0]])
    np.testing.assert_array_equal(mapping.align_candidate(candidate), reference)
    assert mapping.identities == ("4", "8", "11")
    assert mapping.complete
    restored = EntityMapping.from_dict(json.loads(json.dumps(mapping.to_dict())))
    np.testing.assert_array_equal(restored.candidate_indices, [1, 3, 0])


def test_partial_cell_mapping_selects_both_reference_and_candidate():
    mapping = active_global_cell_mapping(
        [4, 8, 11], [11, 4], require_complete=False,
    )
    assert not mapping.complete
    np.testing.assert_array_equal(
        mapping.select_reference([[40.0, 80.0, 110.0]]), [[40.0, 110.0]],
    )
    np.testing.assert_array_equal(
        mapping.align_candidate([[110.0, 40.0]]), [[40.0, 110.0]],
    )


def test_phase_component_and_well_mappings_require_explicit_aliases():
    phases = phase_mapping(
        ["water", "oil", "gas"], ["G", "W", "O"],
        aliases={"W": "water", "O": "oil", "G": "gas"},
    )
    np.testing.assert_array_equal(phases.candidate_indices, [1, 2, 0])
    components = component_mapping(["C1", "C3"], ["C3", "C1"])
    np.testing.assert_array_equal(components.candidate_indices, [1, 0])
    wells = named_entity_mapping(["I-1", "P-1"], ["P-1", "I-1"])
    np.testing.assert_array_equal(wells.candidate_indices, [1, 0])
    with pytest.raises(ValueError, match="missing phase"):
        phase_mapping(["water"], ["W"])
    with pytest.raises(ValueError, match="not unique"):
        component_mapping(["C1", "C1"], ["C1"])


def test_comparison_uses_explicit_active_cell_mapping():
    reference = ComparisonSeries(
        [[40.0, 80.0, 110.0]], [1.0], unit="bar",
    )
    candidate = ComparisonSeries(
        [[110.0, 40.0, 999.0, 80.0]], [1.0], unit="bar",
    )
    result = compare_series(
        reference, candidate,
        entity_mapping=active_global_cell_mapping([4, 8, 11], [11, 4, 99, 8]),
        tolerance=TolerancePolicy(absolute=0.0),
    )
    assert result.passed
    assert result.worst_entity in {"4", "8", "11"}


def test_region_aggregation_is_ordered_and_axis_explicit():
    values = np.array([
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
    ])  # time, cell, phase
    regions = np.array([20, 10, 20, 10])
    summed = aggregate_regions(
        values, regions, axis=1, reduction="sum", region_order=[10, 20],
    )
    assert summed.region_ids == ("10", "20")
    assert summed.values.shape == (2, 2, 2)  # time, phase, region
    np.testing.assert_array_equal(summed.values[0], [[6.0, 4.0], [60.0, 40.0]])
    weighted = aggregate_regions(
        values[:, :, 0], regions, axis=1, reduction="weighted_mean",
        weights=[1.0, 1.0, 3.0, 1.0], region_order=[10, 20],
    )
    np.testing.assert_allclose(weighted.values[0], [3.0, 2.5])


def test_region_aggregation_refuses_ambiguous_weighting_and_order():
    with pytest.raises(ValueError, match="requires cell weights"):
        aggregate_regions([[1.0, 2.0]], [1, 2], reduction="weighted_mean")
    with pytest.raises(ValueError, match="region_order mismatch"):
        aggregate_regions([[1.0, 2.0]], [1, 2], region_order=[1])
    with pytest.raises(ValueError, match="zero total weight"):
        aggregate_regions(
            [[1.0, 2.0]], [1, 2], reduction="weighted_mean", weights=[0.0, 1.0],
        )


def _graph():
    nodes = [
        EvidenceNode("mrst", EvidenceNodeKind.MRST, "MRST 2024a", "2024a"),
        EvidenceNode("pymrst", EvidenceNodeKind.PYMRST, "PyMRST", "dev"),
        EvidenceNode("mass", EvidenceNodeKind.INVARIANT, "Component balance"),
    ]
    edges = [
        EvidenceEdge(
            "mrst-pressure", "mrst", "pymrst", "field_history",
            "Pressure history agrees after active-cell and unit mapping",
            Independence.PARTIAL,
            mappings=("active_global_cell:v1", "Pa_to_bar:v1"),
            shared_kernels=("SPE1 input deck",),
            metrics={"max_relative_error": 2e-8},
        ),
        EvidenceEdge(
            "mass-balance", "mass", "pymrst", "conservation",
            "Closed-system component mass is conserved",
            Independence.INDEPENDENT,
            metrics={"normalized_imbalance": 3e-13},
        ),
    ]
    return EvidenceGraph(nodes, edges)


def test_evidence_graph_roundtrip_and_independent_support_query():
    graph = _graph()
    restored = EvidenceGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert len(restored.support_for("pymrst")) == 2
    independent = restored.independent_support_for("pymrst")
    assert [edge.edge_id for edge in independent] == ["mass-balance"]
    assert independent[0].claim.startswith("Closed-system")


def test_evidence_graph_rejects_overclaimed_or_dangling_evidence():
    with pytest.raises(ValueError, match="shared kernels"):
        EvidenceEdge(
            "bad", "a", "b", "comparison", "overclaimed independence",
            Independence.INDEPENDENT, shared_kernels=("same EOS",),
        )
    with pytest.raises(ValueError, match="unconverged"):
        EvidenceEdge(
            "bad", "a", "b", "comparison", "failed nonlinear solve",
            Independence.PARTIAL, converged=False, accepted=True,
        )
    graph = EvidenceGraph([EvidenceNode("a", EvidenceNodeKind.PYMRST, "A")])
    with pytest.raises(ValueError, match="unknown endpoints"):
        graph.add_edge(EvidenceEdge(
            "dangling", "a", "missing", "comparison", "nothing",
            Independence.PARTIAL,
        ))
    with pytest.raises(ValueError, match="duplicate evidence node"):
        graph.add_node(EvidenceNode("a", EvidenceNodeKind.MRST, "duplicate"))


def test_comparison_result_becomes_a_graph_edge_without_overclaiming():
    reference = ComparisonSeries([[1.0], [2.0]], [0.0, 1.0], ("P1",))
    candidate = ComparisonSeries([[1.0], [2.001]], [0.0, 1.0], ("P1",))
    result = compare_series(
        reference, candidate, tolerance=TolerancePolicy(absolute=1e-4),
    )
    edge = edge_from_comparison(
        "pressure", "mrst", "pymrst", "well pressure agrees", result,
        independence=Independence.PARTIAL,
        mappings=("well_name:v1",), shared_kernels=("input deck",),
    )
    assert not edge.accepted
    assert edge.metrics["maximum_absolute_error"] == pytest.approx(0.001)
    assert edge.mappings == ("well_name:v1",)


def test_evidence_graph_persists_through_simulation_evidence_and_hdf5(tmp_path):
    evidence = SimulationEvidence(
        RunIdentity(RunId("run_graph"), CaseId("case_graph")),
    )
    record_evidence_graph(evidence, _graph(), graph_id="qualification")
    with pytest.raises(ValueError, match="duplicate"):
        record_evidence_graph(evidence, _graph(), graph_id="qualification")

    store = HDF5ResultStore(tmp_path/"graph.h5")
    store.begin_run(RunId("run_graph"), {"case": "graph"})
    store.commit_evidence(RunId("run_graph"), evidence)
    restored_evidence = store.load_evidence(RunId("run_graph"))
    restored = evidence_graphs(restored_evidence)
    assert set(restored) == {"qualification"}
    assert len(restored["qualification"].independent_support_for("pymrst")) == 1
