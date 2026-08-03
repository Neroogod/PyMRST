"""Verified, deterministic research-package generation (M3.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
from typing import Iterable, Mapping

from .definition import canonical_value, content_digest


RESEARCH_PACKAGE_SCHEMA_VERSION = 1
ARTIFACT_KINDS = ("cases", "evidence", "comparisons", "figures", "tables")


class InvalidResearchPackage(ValueError):
    pass


def _hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PackageArtifact:
    source: str
    kind: str
    name: str = ""

    def __post_init__(self):
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"artifact kind must be one of {ARTIFACT_KINDS}")
        source = Path(self.source)
        if not source.is_file():
            raise FileNotFoundError(source)
        name = self.name or source.name
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("artifact name must be one safe filename")
        object.__setattr__(self, "source", str(source.resolve()))
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class ResearchPackageDefinition:
    name: str
    experiment_id: str
    member_ids: tuple[str, ...]
    reproduction_command: tuple[str, ...]
    artifacts: tuple[PackageArtifact, ...] = ()
    metadata: Mapping = field(default_factory=dict)
    schema_version: int = RESEARCH_PACKAGE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != RESEARCH_PACKAGE_SCHEMA_VERSION:
            raise ValueError("unsupported research-package schema")
        if not self.name or not self.experiment_id or not self.member_ids:
            raise ValueError("package name, experiment and members are required")
        if not self.reproduction_command:
            raise ValueError("a reproduction command is required")
        object.__setattr__(self, "member_ids", tuple(map(str, self.member_ids)))
        object.__setattr__(self, "reproduction_command",
                           tuple(map(str, self.reproduction_command)))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "metadata", canonical_value(self.metadata))
        destinations = [(item.kind, item.name) for item in self.artifacts]
        if len(set(destinations)) != len(destinations):
            raise ValueError("two artifacts target the same package path")

    @property
    def package_id(self):
        inputs = [{"kind": item.kind, "name": item.name,
                   "sha256": _hash(Path(item.source))} for item in self.artifacts]
        return f"package_{content_digest({'definition': self.to_dict(False), 'inputs': inputs})}"

    def to_dict(self, include_id=True):
        result = {
            "schema_version": self.schema_version, "name": self.name,
            "experiment_id": self.experiment_id,
            "member_ids": list(self.member_ids),
            "reproduction_command": list(self.reproduction_command),
            "artifacts": [{"kind": item.kind, "name": item.name,
                           "source": item.source} for item in self.artifacts],
            "metadata": dict(self.metadata),
        }
        if include_id:
            result["package_id"] = self.package_id
        return result


def _environment():
    packages = {}
    for name in ("pymrst", "numpy", "scipy", "h5py", "xarray", "zarr", "pandas"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": packages}


def _inventory(root: Path):
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative != "package_manifest.json":
            entries.append({"path": relative, "sha256": _hash(path),
                            "size_bytes": path.stat().st_size})
    return entries


def generate_research_package(definition: ResearchPackageDefinition,
                              destination: str | Path):
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"research package destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for kind in ARTIFACT_KINDS:
        (root / kind).mkdir()
    for artifact in definition.artifacts:
        shutil.copyfile(artifact.source, root / artifact.kind / artifact.name)
    (root / "environment.json").write_text(
        json.dumps(_environment(), indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (root / "definition.json").write_text(
        json.dumps(definition.to_dict(), indent=2, sort_keys=True,
                   allow_nan=False)+"\n", encoding="utf-8")
    command = " ".join(shlex.quote(value) for value in definition.reproduction_command)
    reproduce = root / "reproduce.sh"
    reproduce.write_text("#!/bin/sh\nset -eu\n"+command+"\n", encoding="utf-8")
    reproduce.chmod(0o755)
    (root / "README.md").write_text(
        f"# {definition.name}\n\nPackage ID: `{definition.package_id}`\n\n"
        f"Experiment: `{definition.experiment_id}`\n\n"
        f"Members: {', '.join(f'`{value}`' for value in definition.member_ids)}\n\n"
        "Run `./reproduce.sh` from this directory to reproduce the declared workflow.\n",
        encoding="utf-8")
    manifest = {"schema_version": RESEARCH_PACKAGE_SCHEMA_VERSION,
                "package_id": definition.package_id,
                "inventory": _inventory(root)}
    (root / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    verify_research_package(root)
    return root


def verify_research_package(path: str | Path):
    root = Path(path)
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RESEARCH_PACKAGE_SCHEMA_VERSION:
        raise InvalidResearchPackage("unsupported package manifest schema")
    actual = _inventory(root)
    if actual != manifest.get("inventory"):
        raise InvalidResearchPackage("research package inventory or content changed")
    definition = json.loads((root / "definition.json").read_text(encoding="utf-8"))
    if definition.get("package_id") != manifest.get("package_id"):
        raise InvalidResearchPackage("definition and package manifest identities differ")
    return manifest


__all__ = ["ARTIFACT_KINDS", "InvalidResearchPackage", "PackageArtifact",
           "RESEARCH_PACKAGE_SCHEMA_VERSION", "ResearchPackageDefinition",
           "generate_research_package", "verify_research_package"]
