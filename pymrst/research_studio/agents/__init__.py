"""Evidence-grounded, provider-independent engineering agent contracts.

This package is additive.  Importing it does not initialize a language-model
provider, open a result store, or start a simulator worker.
"""

from .base import AgentTool, ReasoningProvider, ToolCall, ToolDescriptor, ToolResult
from .schemas.evidence import EvidenceKind, EvidenceReference
from .schemas.findings import (
    ClaimKind,
    EngineeringFinding,
    FindingSeverity,
)
from .schemas.recommendations import Recommendation, RecommendationStatus
from .permissions import AgentPermission, PermissionGrant
from .policies.resource_limits import ExecutionBudget
from .context import InvestigationContext, Selection
from .memory import InvestigationRecord, InvestigationStatus
from .orchestrator import (
    DeterministicOrchestrator,
    InvestigationOutcome,
    InvestigationPlan,
    SingleCaseInvestigator,
)
from .bundle import read_investigation_bundle, write_investigation_bundle
from .graph import (
    GraphInvestigationOutcome,
    InvestigationGraph,
    InvestigationNode,
    SpecialistGraphInvestigator,
)
from .review import (
    FindingConflict,
    HumanReviewDecision,
    ReviewAction,
    ReviewTarget,
    UncertaintyAssessment,
)
from .synthesis import FindingCorroboration, FindingSynthesis, synthesize_findings
from .execution import (
    CancelSimulationTool,
    CheckpointLineage,
    CheckpointLineageCatalog,
    DiagnoseRunFailureTool,
    FailureClass,
    FailureDiagnosis,
    InspectSimulationTool,
    RetryLimits,
    RetryProposal,
    RunRequestCatalog,
    RunResultCatalog,
    SubmitSimulationTool,
    authorize_retry,
    diagnose_run_failure,
    materialize_resume_request,
    materialize_retry_request,
)
from .variants import CaseVariantCatalog, CaseVariantManifest, register_case_variant
from .tools.experiment_tools import (
    AnalyzeEnsembleTool,
    EnsembleAnalysisCatalog,
    ExperimentRequestCatalog,
    ExperimentSubmissionRequest,
    ExperimentSubmissionResult,
    SubmitExperimentTool,
)
from .tools.validation_tools import CompareReferenceRunsTool, ReferenceRunCatalog
from .tools.application_run_tools import (
    AnalyzeApplicationRunTool, RunEventCatalog, RunEventSnapshot,
)
from .tools.application_case_tools import (
    AnalyzeDeckInspectionTool, AnalyzeGridSnapshotTool,
    AnalyzeInitializationSnapshotTool, AnalyzeRockFluidSnapshotTool,
    AnalyzeWellsScheduleSnapshotTool, ApplicationCaseCatalog,
    ApplicationCaseSnapshot,
)
from .tools.application_result_tools import (
    AnalyzeApplicationConservationTool, AnalyzeApplicationPerformanceTool,
    ApplicationResultCatalog, ApplicationResultSnapshot,
)
from .planning import (
    BoundedInvestigationPlanner, DeterministicPlanningProvider, PlanningRecord,
)

__all__ = [
    "AgentPermission",
    "AgentTool",
    "AnalyzeEnsembleTool",
    "AnalyzeApplicationRunTool",
    "AnalyzeDeckInspectionTool",
    "AnalyzeGridSnapshotTool",
    "AnalyzeInitializationSnapshotTool",
    "AnalyzeRockFluidSnapshotTool",
    "AnalyzeWellsScheduleSnapshotTool",
    "ApplicationCaseCatalog",
    "ApplicationCaseSnapshot",
    "AnalyzeApplicationConservationTool",
    "AnalyzeApplicationPerformanceTool",
    "ApplicationResultCatalog",
    "ApplicationResultSnapshot",
    "BoundedInvestigationPlanner",
    "DeterministicPlanningProvider",
    "PlanningRecord",
    "ClaimKind",
    "CancelSimulationTool",
    "CaseVariantCatalog",
    "CaseVariantManifest",
    "CheckpointLineage",
    "CheckpointLineageCatalog",
    "CompareReferenceRunsTool",
    "DiagnoseRunFailureTool",
    "EngineeringFinding",
    "EnsembleAnalysisCatalog",
    "EvidenceKind",
    "EvidenceReference",
    "ExecutionBudget",
    "ExperimentRequestCatalog",
    "ExperimentSubmissionRequest",
    "ExperimentSubmissionResult",
    "FindingSeverity",
    "FailureClass",
    "FailureDiagnosis",
    "FindingConflict",
    "GraphInvestigationOutcome",
    "HumanReviewDecision",
    "InvestigationContext",
    "InvestigationOutcome",
    "InvestigationPlan",
    "InvestigationGraph",
    "InvestigationNode",
    "InvestigationRecord",
    "InvestigationStatus",
    "InspectSimulationTool",
    "PermissionGrant",
    "ReasoningProvider",
    "ReferenceRunCatalog",
    "read_investigation_bundle",
    "register_case_variant",
    "Recommendation",
    "RecommendationStatus",
    "ReviewAction",
    "ReviewTarget",
    "FindingCorroboration", "FindingSynthesis", "synthesize_findings",
    "RetryLimits",
    "RetryProposal",
    "RunRequestCatalog",
    "RunResultCatalog",
    "RunEventCatalog",
    "RunEventSnapshot",
    "Selection",
    "SingleCaseInvestigator",
    "SpecialistGraphInvestigator",
    "SubmitSimulationTool",
    "SubmitExperimentTool",
    "DeterministicOrchestrator",
    "ToolCall",
    "ToolDescriptor",
    "ToolResult",
    "UncertaintyAssessment",
    "authorize_retry",
    "diagnose_run_failure",
    "materialize_resume_request",
    "materialize_retry_request",
    "write_investigation_bundle",
]
