"""Reproducibility manifests with explicit input and environment lineage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

from pymrst.research_studio.runtime.identifiers import RunId

MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1024*1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class InputArtifact:
    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, path: str | Path) -> "InputArtifact":
        source = Path(path).resolve()
        return cls(str(source), sha256_file(source), source.stat().st_size)

    def verify(self) -> None:
        source = Path(self.path)
        if not source.is_file():
            raise FileNotFoundError(f"manifest input is missing: {source}")
        if source.stat().st_size != self.size_bytes or sha256_file(source) != self.sha256:
            raise ValueError(f"manifest input changed: {source}")


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    run_id: RunId
    created_utc: str
    python_version: str
    platform: str
    command: tuple[str, ...]
    working_directory: str
    package_versions: Mapping[str, str]
    code_fingerprint: str
    inputs: tuple[InputArtifact, ...] = ()
    configuration: Mapping[str, Any] = None
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self):
        if self.configuration is None:
            object.__setattr__(self, "configuration", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": str(self.run_id), "created_utc": self.created_utc,
            "python_version": self.python_version, "platform": self.platform,
            "command": list(self.command),
            "working_directory": self.working_directory,
            "package_versions": dict(self.package_versions),
            "code_fingerprint": self.code_fingerprint,
            "inputs": [
                {"path": item.path, "sha256": item.sha256,
                 "size_bytes": item.size_bytes} for item in self.inputs
            ],
            "configuration": dict(self.configuration),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReproducibilityManifest":
        data = dict(value)
        if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema {data.get('schema_version')!r}")
        return cls(
            run_id=RunId(data["run_id"]), created_utc=str(data["created_utc"]),
            python_version=str(data["python_version"]), platform=str(data["platform"]),
            command=tuple(map(str, data["command"])),
            working_directory=str(data["working_directory"]),
            package_versions={str(k): str(v) for k, v in data["package_versions"].items()},
            code_fingerprint=str(data["code_fingerprint"]),
            inputs=tuple(InputArtifact(**item) for item in data.get("inputs", ())),
            configuration=dict(data.get("configuration", {})),
        )

    def to_json(self, *, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def verify_inputs(self) -> None:
        for artifact in self.inputs:
            artifact.verify()


def capture_manifest(
    run_id: RunId | str,
    *,
    code_fingerprint: str,
    inputs: Iterable[str | Path] = (),
    command: Iterable[str] | None = None,
    working_directory: str | Path | None = None,
    configuration: Mapping[str, Any] | None = None,
    packages: Iterable[str] = ("pymrst", "numpy", "scipy", "h5py"),
) -> ReproducibilityManifest:
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return ReproducibilityManifest(
        run_id=RunId(run_id),
        created_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.split()[0], platform=platform.platform(),
        command=tuple(sys.argv if command is None else command),
        working_directory=str(Path.cwd() if working_directory is None else Path(working_directory).resolve()),
        package_versions=versions, code_fingerprint=str(code_fingerprint),
        inputs=tuple(InputArtifact.from_path(path) for path in inputs),
        configuration=dict(configuration or {}),
    )


__all__ = [
    "InputArtifact", "MANIFEST_SCHEMA_VERSION", "ReproducibilityManifest",
    "capture_manifest", "sha256_file",
]
