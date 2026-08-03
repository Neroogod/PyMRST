"""Reproducible MATLAB/MRST qualification adapter."""

from .adapter import MatlabReferenceAdapter
from .execution import discover_matlab, generation_command, run_generator
from .fixtures import (
    StaleMatlabFixtureError, generate_fixture, load_manifest, validate_fixture,
    write_manifest,
)
from .mappings import MatlabResultMappings, result_mappings
from .provenance import infer_mrst_version, matlab_release_from_output, sha256_file
from .schemas import (
    MATLAB_FIXTURE_SCHEMA_VERSION, MatlabExecutionResult, MatlabFixtureManifest,
    MatlabFixtureSpec,
)

__all__ = [
    "MATLAB_FIXTURE_SCHEMA_VERSION", "MatlabExecutionResult",
    "MatlabFixtureManifest", "MatlabFixtureSpec", "MatlabReferenceAdapter",
    "MatlabResultMappings", "StaleMatlabFixtureError", "discover_matlab",
    "generate_fixture", "generation_command", "infer_mrst_version",
    "load_manifest", "matlab_release_from_output", "result_mappings",
    "run_generator", "sha256_file", "validate_fixture", "write_manifest",
]
