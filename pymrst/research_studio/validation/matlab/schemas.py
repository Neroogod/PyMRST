"""Strict schemas for reproducible MATLAB/MRST qualification fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


MATLAB_FIXTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MatlabFixtureSpec:
    fixture_id: str
    reference_path: Path
    generator_path: Path
    generator_function: str
    mrst_root: Path
    modules: tuple[str, ...]
    mappings: tuple[str, ...]
    tolerances: Mapping[str, Mapping[str, float]]
    input_artifacts: Mapping[str, Path]
    solver_settings: Mapping[str, Any]
    case_name: str
    schema_version: int = MATLAB_FIXTURE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != MATLAB_FIXTURE_SCHEMA_VERSION:
            raise ValueError("unsupported MATLAB fixture schema version")
        if not all((self.fixture_id, self.generator_function, self.case_name)):
            raise ValueError("fixture identity, generator function, and case name are required")
        if (not self.modules or not self.mappings or not self.tolerances
                or not self.input_artifacts or not self.solver_settings):
            raise ValueError(
                "fixture modules, mappings, tolerances, inputs, and solver settings "
                "must be explicit")
        object.__setattr__(self, "reference_path", Path(self.reference_path).resolve())
        object.__setattr__(self, "generator_path", Path(self.generator_path).resolve())
        object.__setattr__(self, "mrst_root", Path(self.mrst_root).resolve())
        object.__setattr__(self, "modules", tuple(str(v) for v in self.modules))
        object.__setattr__(self, "mappings", tuple(str(v) for v in self.mappings))
        object.__setattr__(self, "tolerances", {
            str(group): {str(name): float(value) for name, value in values.items()}
            for group, values in self.tolerances.items()
        })
        object.__setattr__(self, "input_artifacts", {
            str(name): Path(path).resolve() for name, path in self.input_artifacts.items()
        })
        object.__setattr__(self, "solver_settings", dict(self.solver_settings))

    @property
    def manifest_path(self) -> Path:
        return self.reference_path.with_suffix(self.reference_path.suffix + ".manifest.json")


@dataclass(frozen=True, slots=True)
class MatlabExecutionResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class MatlabFixtureManifest:
    fixture_id: str
    case_name: str
    matlab_executable: str
    matlab_release: str
    mrst_root: str
    mrst_version: str
    modules: tuple[str, ...]
    generator_function: str
    generator_sha256: str
    reference_sha256: str
    reference_bytes: int
    mappings: tuple[str, ...]
    tolerances: Mapping[str, Mapping[str, float]]
    input_artifacts: Mapping[str, Mapping[str, Any]]
    solver_settings: Mapping[str, Any]
    regeneration_command: tuple[str, ...]
    generated_at_utc: str
    schema_version: int = MATLAB_FIXTURE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version != MATLAB_FIXTURE_SCHEMA_VERSION:
            raise ValueError("unsupported MATLAB fixture manifest schema")
        for digest in (self.generator_sha256, self.reference_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("fixture digests must be lowercase SHA-256 values")
        if self.reference_bytes <= 0:
            raise ValueError("reference fixture must be nonempty")
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "mappings", tuple(self.mappings))
        object.__setattr__(self, "regeneration_command", tuple(self.regeneration_command))
        object.__setattr__(self, "tolerances", {
            str(group): {str(name): float(value) for name, value in values.items()}
            for group, values in self.tolerances.items()
        })
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "input_artifacts", {
            str(name): dict(value) for name, value in self.input_artifacts.items()
        })
        object.__setattr__(self, "solver_settings", dict(self.solver_settings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixture_id": self.fixture_id, "case_name": self.case_name,
            "matlab_executable": self.matlab_executable,
            "matlab_release": self.matlab_release,
            "mrst_root": self.mrst_root, "mrst_version": self.mrst_version,
            "modules": list(self.modules),
            "generator_function": self.generator_function,
            "generator_sha256": self.generator_sha256,
            "reference_sha256": self.reference_sha256,
            "reference_bytes": self.reference_bytes,
            "mappings": list(self.mappings), "tolerances": dict(self.tolerances),
            "input_artifacts": {name: dict(value)
                                for name, value in self.input_artifacts.items()},
            "solver_settings": dict(self.solver_settings),
            "regeneration_command": list(self.regeneration_command),
            "generated_at_utc": self.generated_at_utc,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatlabFixtureManifest":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError(f"invalid MATLAB manifest fields: {sorted(set(value) ^ expected)}")
        return cls(**value)


__all__ = ["MATLAB_FIXTURE_SCHEMA_VERSION", "MatlabExecutionResult",
           "MatlabFixtureManifest", "MatlabFixtureSpec"]
