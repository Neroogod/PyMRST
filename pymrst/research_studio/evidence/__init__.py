"""Structured, observation-only simulation evidence."""

from .events import EventKind, SimulationEvent
from .event_bus import EventBus, JsonLinesHandler
from .report import RunIdentity, SimulationEvidence
from .conservation import (
    BlackOilConservationAdapter, CompositionalConservationAdapter, ConservationReport,
    OilWaterConservationAdapter, PoromechanicsConservationAdapter,
    WaterConservationAdapter, aquifer_component_rates, conservation_report,
)
from .invariants import (
    InvariantProfile, InvariantSuite, InvariantViolation, bounded_field,
    finite_field,
)
from .failure_analysis import (
    FailureAnalysis, FailureKind, analyze_residual_history,
    analyze_simulation_evidence,
)
from .manifest import InputArtifact, ReproducibilityManifest, capture_manifest
from .failure_bundle import (
    FailureReplay, load_failure_bundle, prepare_failure_replay,
    write_failure_bundle, write_store_failure_bundle,
)

__all__ = [
    "EventBus", "EventKind", "JsonLinesHandler", "SimulationEvent",
    "RunIdentity", "SimulationEvidence",
    "BlackOilConservationAdapter", "CompositionalConservationAdapter",
    "ConservationReport", "OilWaterConservationAdapter",
    "PoromechanicsConservationAdapter",
    "WaterConservationAdapter", "aquifer_component_rates",
    "conservation_report", "InvariantProfile",
    "InvariantSuite", "InvariantViolation", "bounded_field", "finite_field",
    "FailureAnalysis", "FailureKind", "FailureReplay", "analyze_residual_history",
    "analyze_simulation_evidence",
    "InputArtifact", "ReproducibilityManifest", "capture_manifest",
    "load_failure_bundle", "prepare_failure_replay", "write_failure_bundle",
    "write_store_failure_bundle",
]
