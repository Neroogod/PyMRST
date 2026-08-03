"""First-class, reproducible Research Studio experiment definitions."""

from .definition import (
    EXPERIMENT_DEFINITION_SCHEMA_VERSION, CaseReference, DatasetReference,
    ExecutionProfile, ExperimentDefinition, OutputRequest, OwnerReference,
    ValidationRequirement, canonical_json, canonical_value, content_digest,
)
from .registry import (
    EXPERIMENT_REGISTRY_SCHEMA_VERSION, ExperimentIdentityCollision,
    ExperimentRegistry,
)
from .variants import (
    SAMPLING_ALGORITHM_V1, VARIANT_MATRIX_SCHEMA_VERSION,
    CartesianVariantMatrix, ComponentAxis, DistributionSpec,
    ExplicitVariantMatrix, MaterializedVariantMatrix, ParameterAxis,
    SampledParameter, SampledVariantMatrix, VariantCoordinate,
    VariantDefinition, explicit_variant_coordinates, materialize_variant_matrix,
    variant_matrix_from_dict,
)
from .execution import (
    LazyMemberResult, MemberExecutionContext, MemberOutcome, MemberPayload,
    MemberResultStore,
)
from .ensemble import (
    EnsembleReport, LocalEnsembleExecutor, deterministic_member_seed,
)
from .comparison import (
    ScenarioComparison, ScenarioQuantityPolicy, compare_performance,
    compare_scenarios,
)
from .branching import (
    BranchDefinition, BranchExecutionContext, BranchOutcome, IncompatibleBranch,
    execute_branch, verify_branch_lineage,
)
from .publishing import (
    InvalidResearchPackage, PackageArtifact, ResearchPackageDefinition,
    generate_research_package, verify_research_package,
)

__all__ = [
    "EXPERIMENT_DEFINITION_SCHEMA_VERSION", "EXPERIMENT_REGISTRY_SCHEMA_VERSION",
    "CaseReference", "DatasetReference", "ExecutionProfile",
    "ExperimentDefinition", "ExperimentIdentityCollision", "ExperimentRegistry",
    "OutputRequest", "OwnerReference", "ValidationRequirement",
    "VariantCoordinate", "VariantDefinition", "canonical_json", "canonical_value",
    "content_digest", "explicit_variant_coordinates",
    "SAMPLING_ALGORITHM_V1", "VARIANT_MATRIX_SCHEMA_VERSION",
    "CartesianVariantMatrix", "ComponentAxis", "DistributionSpec",
    "ExplicitVariantMatrix", "MaterializedVariantMatrix", "ParameterAxis",
    "SampledParameter", "SampledVariantMatrix", "materialize_variant_matrix",
    "variant_matrix_from_dict",
    "EnsembleReport", "LazyMemberResult", "LocalEnsembleExecutor",
    "MemberExecutionContext", "MemberOutcome", "MemberPayload",
    "MemberResultStore", "ScenarioComparison", "ScenarioQuantityPolicy",
    "compare_performance", "compare_scenarios", "deterministic_member_seed",
    "BranchDefinition", "BranchExecutionContext", "BranchOutcome",
    "IncompatibleBranch", "execute_branch", "verify_branch_lineage",
    "InvalidResearchPackage", "PackageArtifact", "ResearchPackageDefinition",
    "generate_research_package", "verify_research_package",
]
