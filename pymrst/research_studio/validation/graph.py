"""Traceable evidence graph for numerical validation claims."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .comparison import ComparisonResult, FieldComparisonResult


EVIDENCE_GRAPH_SCHEMA_VERSION = 1


class EvidenceNodeKind(str, Enum):
    PYMRST = "pymrst"
    MRST = "mrst"
    OPM = "opm"
    ANALYTIC = "analytic"
    MANUFACTURED = "manufactured"
    INVARIANT = "invariant"
    FINITE_DIFFERENCE = "finite_difference"

class Independence(str, Enum):
    INDEPENDENT = "independent"
    PARTIAL = "partial"
    SHARED_KERNEL = "shared_kernel"


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    node_id: str
    kind: EvidenceNodeKind
    label: str
    version: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id or not self.label:
            raise ValueError("evidence nodes require nonempty id and label")
        object.__setattr__(self, "kind", EvidenceNodeKind(self.kind))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self):
        return {
            "node_id": self.node_id, "kind": self.kind.value,
            "label": self.label, "version": self.version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    edge_id: str
    source: str
    target: str
    evidence_class: str
    claim: str
    independence: Independence
    mappings: tuple[str, ...] = ()
    shared_kernels: tuple[str, ...] = ()
    converged: bool = True
    accepted: bool = True
    exceptions: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not all((self.edge_id, self.source, self.target, self.evidence_class, self.claim)):
            raise ValueError("evidence edges require ids, endpoints, class, and claim")
        object.__setattr__(self, "independence", Independence(self.independence))
        object.__setattr__(self, "mappings", tuple(self.mappings))
        object.__setattr__(self, "shared_kernels", tuple(self.shared_kernels))
        object.__setattr__(self, "exceptions", tuple(self.exceptions))
        object.__setattr__(self, "metrics", {str(k): float(v) for k, v in self.metrics.items()})
        if self.independence is Independence.INDEPENDENT and self.shared_kernels:
            raise ValueError("independent evidence cannot declare shared kernels")
        if self.accepted and not self.converged:
            raise ValueError("unconverged comparison evidence cannot be accepted")

    def to_dict(self):
        return {
            "edge_id": self.edge_id, "source": self.source, "target": self.target,
            "evidence_class": self.evidence_class, "claim": self.claim,
            "independence": self.independence.value,
            "mappings": list(self.mappings), "shared_kernels": list(self.shared_kernels),
            "converged": self.converged, "accepted": self.accepted,
            "exceptions": list(self.exceptions), "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


class EvidenceGraph:
    """A strict graph whose edges state the limits of validation evidence."""

    def __init__(self, nodes: Iterable[EvidenceNode] = (), edges: Iterable[EvidenceEdge] = ()):
        self._nodes: dict[str, EvidenceNode] = {}
        self._edges: dict[str, EvidenceEdge] = {}
        for node in nodes:
            self.add_node(node)
        for edge in edges:
            self.add_edge(edge)

    @property
    def nodes(self) -> tuple[EvidenceNode, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[EvidenceEdge, ...]:
        return tuple(self._edges.values())

    def add_node(self, node: EvidenceNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate evidence node: {node.node_id}")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        if edge.edge_id in self._edges:
            raise ValueError(f"duplicate evidence edge: {edge.edge_id}")
        missing = [node for node in (edge.source, edge.target) if node not in self._nodes]
        if missing:
            raise ValueError(f"evidence edge has unknown endpoints: {missing}")
        self._edges[edge.edge_id] = edge

    def support_for(self, node_id: str, *, accepted_only: bool = True) -> tuple[EvidenceEdge, ...]:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        return tuple(
            edge for edge in self._edges.values()
            if edge.target == node_id and (edge.accepted or not accepted_only)
        )

    def independent_support_for(self, node_id: str) -> tuple[EvidenceEdge, ...]:
        return tuple(
            edge for edge in self.support_for(node_id)
            if edge.independence is Independence.INDEPENDENT
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_GRAPH_SCHEMA_VERSION,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceGraph":
        if set(value) != {"schema_version", "nodes", "edges"}:
            raise ValueError("invalid EvidenceGraph fields")
        if value["schema_version"] != EVIDENCE_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported EvidenceGraph schema version")
        return cls(
            (EvidenceNode.from_dict(item) for item in value["nodes"]),
            (EvidenceEdge.from_dict(item) for item in value["edges"]),
        )


def edge_from_comparison(
    edge_id: str,
    source: str,
    target: str,
    claim: str,
    result: ComparisonResult | FieldComparisonResult,
    *,
    independence: Independence,
    evidence_class: str = "normalized_comparison",
    mappings: Iterable[str] = (),
    shared_kernels: Iterable[str] = (),
    converged: bool = True,
    exceptions: Iterable[str] = (),
) -> EvidenceEdge:
    """Create a traceable graph edge from a normalized comparison result."""
    return EvidenceEdge(
        edge_id=edge_id, source=source, target=target,
        evidence_class=evidence_class, claim=claim,
        independence=independence, mappings=tuple(mappings),
        shared_kernels=tuple(shared_kernels), converged=converged,
        accepted=bool(result.passed and converged), exceptions=tuple(exceptions),
        metrics={
            "maximum_absolute_error": result.maximum_absolute_error,
            "maximum_relative_error": result.maximum_relative_error,
            "maximum_scaled_error": result.maximum_scaled_error,
            "worst_time": result.worst_time,
        },
    )


def record_evidence_graph(evidence: Any, graph: EvidenceGraph, *, graph_id: str) -> None:
    """Persist a graph inside ``SimulationEvidence.comparisons``.

    The evidence schema already owns comparison artifacts as JSON-safe
    mappings. A tagged envelope avoids changing Evidence v1 while retaining a
    strict graph schema and refusing duplicate identities.
    """
    if not graph_id:
        raise ValueError("graph_id must be nonempty")
    comparisons = getattr(evidence, "comparisons", None)
    if not isinstance(comparisons, list):
        raise TypeError("evidence must expose a mutable comparisons list")
    if any(item.get("artifact") == "evidence_graph" and item.get("graph_id") == graph_id
           for item in comparisons if isinstance(item, dict)):
        raise ValueError(f"duplicate evidence graph id: {graph_id}")
    comparisons.append({
        "artifact": "evidence_graph",
        "graph_id": graph_id,
        "graph": graph.to_dict(),
    })


def evidence_graphs(evidence: Any) -> dict[str, EvidenceGraph]:
    """Load every strict evidence graph attached to a SimulationEvidence."""
    result: dict[str, EvidenceGraph] = {}
    for item in getattr(evidence, "comparisons", ()):
        if not isinstance(item, dict) or item.get("artifact") != "evidence_graph":
            continue
        graph_id = item.get("graph_id")
        if not isinstance(graph_id, str) or not graph_id or graph_id in result:
            raise ValueError("invalid or duplicate persisted evidence graph id")
        result[graph_id] = EvidenceGraph.from_dict(item["graph"])
    return result


__all__ = [
    "EVIDENCE_GRAPH_SCHEMA_VERSION", "EvidenceEdge", "EvidenceGraph",
    "EvidenceNode", "EvidenceNodeKind", "Independence", "edge_from_comparison",
    "evidence_graphs", "record_evidence_graph",
]
