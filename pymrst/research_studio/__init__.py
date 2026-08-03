"""Additive research orchestration for PyMRST simulations.

The package deliberately has no import-time dependency from MRST-compatible
modules.  Integration is opt-in through adapters and observation hooks.
"""

from .runtime.identifiers import (
    AttemptId, CaseId, CheckpointId, ExperimentId, MemberId, MinistepId,
    ReportStepId, RunId,
)
from .evidence.events import EventKind, SimulationEvent
from .evidence.event_bus import EventBus, JsonLinesHandler
from .evidence.report import RunIdentity, SimulationEvidence
from .evidence.conservation import (
    BlackOilConservationAdapter, BlackOilPseudoComponentConservationAdapter,
    CompositionalConservationAdapter, ConservationReport,
    OilWaterConservationAdapter, PoromechanicsConservationAdapter,
    WaterConservationAdapter, aquifer_component_rates, conservation_report,
)
from .evidence.invariants import (
    InvariantProfile, InvariantSuite, InvariantViolation, bounded_field,
    finite_field,
)
from .evidence.failure_analysis import (
    FailureAnalysis, FailureKind, analyze_residual_history,
    analyze_simulation_evidence,
)
from .evidence.manifest import InputArtifact, ReproducibilityManifest, capture_manifest
from .evidence.failure_bundle import (
    FailureReplay, load_failure_bundle, prepare_failure_replay,
    write_failure_bundle, write_store_failure_bundle,
)
from .runtime.cancellation import CancellationRequested, CancellationToken
from .runtime.session import SimulationResult, SimulationSession, SimulationStatus
from .storage import (
    CheckpointCompatibilityError, CheckpointRecord, HDF5ResultStore,
    ResultStore, StoreCorruptionError,
)
from .validation import (
    ComparisonResult, ComparisonSeries, EntityMapping, EvidenceEdge,
    EvidenceGraph, EvidenceNode, EvidenceNodeKind, FieldComparisonResult,
    Independence, NamedAxis, NamedFieldSeries, RegionAggregation,
    TolerancePolicy, active_global_cell_mapping, aggregate_regions,
    compare_named_fields, compare_series,
    component_mapping, edge_from_comparison, evidence_graphs, identity_mapping,
    named_entity_mapping, phase_mapping, record_evidence_graph,
)
from .experiments import (
    CartesianVariantMatrix, CaseReference, ComponentAxis, DatasetReference,
    DistributionSpec, ExecutionProfile, ExperimentDefinition,
    ExperimentIdentityCollision, ExperimentRegistry, OutputRequest,
    OwnerReference, ParameterAxis, SampledParameter, SampledVariantMatrix,
    ValidationRequirement, VariantCoordinate, VariantDefinition,
    explicit_variant_coordinates, materialize_variant_matrix,
)

__all__ = [
    "AttemptId", "CancellationRequested", "CancellationToken", "CaseId",
    "CheckpointCompatibilityError", "CheckpointId", "CheckpointRecord",
    "ComparisonResult", "ComparisonSeries", "ConservationReport", "EntityMapping",
    "EventBus", "EventKind", "EvidenceEdge", "EvidenceGraph", "EvidenceNode",
    "EvidenceNodeKind", "ExperimentId", "FailureAnalysis", "FailureKind",
    "FieldComparisonResult",
    "FailureReplay",
    "HDF5ResultStore", "InputArtifact", "InvariantProfile", "InvariantSuite",
    "Independence", "InvariantViolation", "JsonLinesHandler", "MemberId", "MinistepId",
    "NamedAxis", "NamedFieldSeries", "RegionAggregation",
    "BlackOilConservationAdapter", "BlackOilPseudoComponentConservationAdapter",
    "CompositionalConservationAdapter",
    "OilWaterConservationAdapter", "ReproducibilityManifest",
    "PoromechanicsConservationAdapter",
    "ReportStepId", "ResultStore", "RunId",
    "RunIdentity", "SimulationEvent", "SimulationEvidence", "SimulationResult",
    "SimulationSession", "SimulationStatus", "StoreCorruptionError",
    "TolerancePolicy",
    "WaterConservationAdapter", "aquifer_component_rates", "bounded_field",
    "conservation_report",
    "finite_field", "analyze_residual_history", "analyze_simulation_evidence",
    "active_global_cell_mapping", "aggregate_regions", "capture_manifest",
    "load_failure_bundle",
    "prepare_failure_replay", "compare_named_fields", "compare_series",
    "component_mapping",
    "edge_from_comparison", "evidence_graphs", "identity_mapping",
    "named_entity_mapping", "phase_mapping", "record_evidence_graph",
    "write_failure_bundle",
    "write_store_failure_bundle",
    "CaseReference", "DatasetReference", "ExecutionProfile",
    "ExperimentDefinition", "ExperimentIdentityCollision", "ExperimentRegistry",
    "OutputRequest", "OwnerReference", "ValidationRequirement",
    "VariantCoordinate", "VariantDefinition", "explicit_variant_coordinates",
    "CartesianVariantMatrix", "ComponentAxis", "DistributionSpec",
    "ParameterAxis", "SampledParameter", "SampledVariantMatrix",
    "materialize_variant_matrix",
]
