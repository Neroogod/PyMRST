"""Qualified interchangeable-component contracts for Milestone 4."""

from .registry import (
    CAPABILITY_SCHEMA_VERSION, CapabilityRegistry, ComponentCapability,
    ComponentRequest, IncompatibleComponents, registered_reference_components,
)
from .surrogate import (
    ReferenceSamplingPolicy, SurrogateContract, SurrogatePrediction, TensorField,
    ValidityDomain,
)
from .guarded import GuardEvent, GuardedSurrogateRuntime

__all__ = ["CAPABILITY_SCHEMA_VERSION", "CapabilityRegistry",
           "ComponentCapability", "ComponentRequest", "IncompatibleComponents",
           "registered_reference_components"]
__all__ += ["ReferenceSamplingPolicy", "SurrogateContract",
            "SurrogatePrediction", "TensorField", "ValidityDomain",
            "GuardEvent", "GuardedSurrogateRuntime"]
