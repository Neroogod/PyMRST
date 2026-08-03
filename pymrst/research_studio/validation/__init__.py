"""Validation Laboratory contracts and normalized comparisons."""

from .comparison import (
    ComparisonResult, ComparisonSeries, FieldComparisonResult, NamedAxis,
    NamedFieldSeries, TolerancePolicy, compare_named_fields, compare_series,
)
from .graph import (
    EvidenceEdge, EvidenceGraph, EvidenceNode, EvidenceNodeKind, Independence,
    edge_from_comparison, evidence_graphs, record_evidence_graph,
)
from .derivatives import (
    DirectionalDerivativeResult,
    deterministic_directions,
    qualify_directional_derivative,
)
from .analytic import (
    AnalyticCaseResult, AnalyticCaseSpec, AnalyticRegistry, RefinementLevel,
    observed_order,
)
from .benchmarks import (
    ACCUMULATION_FLUX_SPEC, BUCKLEY_LEVERETT_SPEC, DIFFUSION_SPEC,
    ELASTICITY_PATCH_SPEC, FLASH_EQUILIBRIUM_SPEC, HYDROSTATIC_SPEC, MANDEL_SPEC,
    RADIAL_SPEC, TERZAGHI_SPEC,
    buckley_leverett_exact, diffusion_exact,
    foundation_analytic_registry, run_accumulation_flux_benchmark,
    run_diffusion_benchmark, run_elasticity_patch_benchmark,
    run_flash_equilibrium_benchmark,
    radial_bessel_exact, run_buckley_leverett_benchmark,
    run_hydrostatic_benchmark, run_mandel_benchmark,
    run_radial_pressure_benchmark, run_terzaghi_benchmark,
)
from .mappings import (
    EntityMapping, RegionAggregation, active_global_cell_mapping,
    aggregate_regions, component_mapping, identity_mapping,
    named_entity_mapping, phase_mapping,
)
from .matlab import (
    MATLAB_FIXTURE_SCHEMA_VERSION, MatlabExecutionResult, MatlabFixtureManifest,
    MatlabFixtureSpec, MatlabReferenceAdapter, MatlabResultMappings,
    StaleMatlabFixtureError, discover_matlab, generate_fixture,
    generation_command, load_manifest, result_mappings, run_generator,
    sha256_file, validate_fixture,
)

__all__ = [
    "AnalyticCaseResult", "AnalyticCaseSpec", "AnalyticRegistry",
    "RefinementLevel", "observed_order",
    "ACCUMULATION_FLUX_SPEC", "BUCKLEY_LEVERETT_SPEC", "DIFFUSION_SPEC",
    "ELASTICITY_PATCH_SPEC", "FLASH_EQUILIBRIUM_SPEC", "HYDROSTATIC_SPEC", "MANDEL_SPEC",
    "RADIAL_SPEC", "TERZAGHI_SPEC",
    "buckley_leverett_exact", "diffusion_exact",
    "foundation_analytic_registry", "run_accumulation_flux_benchmark",
    "run_diffusion_benchmark", "run_elasticity_patch_benchmark",
    "run_flash_equilibrium_benchmark",
    "radial_bessel_exact", "run_buckley_leverett_benchmark",
    "run_hydrostatic_benchmark", "run_mandel_benchmark",
    "run_radial_pressure_benchmark", "run_terzaghi_benchmark",
    "DirectionalDerivativeResult", "deterministic_directions",
    "qualify_directional_derivative",
    "ComparisonResult", "ComparisonSeries", "EntityMapping", "EvidenceEdge",
    "EvidenceGraph", "EvidenceNode", "EvidenceNodeKind", "Independence",
    "FieldComparisonResult", "NamedAxis", "NamedFieldSeries", "RegionAggregation",
    "TolerancePolicy", "active_global_cell_mapping", "aggregate_regions",
    "compare_named_fields", "compare_series",
    "component_mapping", "identity_mapping", "named_entity_mapping",
    "phase_mapping", "edge_from_comparison", "evidence_graphs",
    "record_evidence_graph",
    "MATLAB_FIXTURE_SCHEMA_VERSION", "MatlabExecutionResult",
    "MatlabFixtureManifest", "MatlabFixtureSpec", "MatlabReferenceAdapter",
    "MatlabResultMappings", "StaleMatlabFixtureError", "discover_matlab",
    "generate_fixture", "generation_command", "load_manifest",
    "result_mappings", "run_generator", "sha256_file", "validate_fixture",
]
