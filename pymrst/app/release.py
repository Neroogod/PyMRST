"""Reproducible, signed release-artifact manifests and offline verification."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable

from .database.models import APP_DATABASE_SCHEMA_VERSION
from .schemas._common import nonempty, utc_now


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe release-artifact path: {value}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ReleaseArtifact:
    path: str
    kind: str
    platform: str
    architecture: str
    size_bytes: int
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "path", _safe_relative(self.path))
        for name in ("kind", "platform", "architecture"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        size = int(self.size_bytes)
        if size < 0:
            raise ValueError("release-artifact size cannot be negative")
        object.__setattr__(self, "size_bytes", size)
        digest = str(self.sha256).casefold()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("release-artifact sha256 must be a SHA-256 digest")
        object.__setattr__(self, "sha256", digest)

    def to_dict(self):
        return {
            "path": self.path, "kind": self.kind, "platform": self.platform,
            "architecture": self.architecture, "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    product: str
    version: str
    database_schema_version: int
    artifacts: tuple[ReleaseArtifact, ...]
    created_at: str
    content_hash: str
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("unsupported release-manifest schema")
        object.__setattr__(self, "product", nonempty(self.product, "product"))
        object.__setattr__(self, "version", nonempty(self.version, "version"))
        object.__setattr__(self, "created_at", nonempty(self.created_at, "created_at"))
        artifacts = tuple(self.artifacts)
        paths = [item.path for item in artifacts]
        if not artifacts or len(paths) != len(set(paths)):
            raise ValueError("release manifest requires unique artifacts")
        object.__setattr__(self, "artifacts", artifacts)
        digest = str(self.content_hash).casefold()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("release-manifest content_hash must be SHA-256")
        object.__setattr__(self, "content_hash", digest)

    def unsigned_dict(self):
        return {
            "schema_version": self.schema_version, "product": self.product,
            "version": self.version,
            "database_schema_version": self.database_schema_version,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "created_at": self.created_at,
        }

    def to_dict(self):
        return {**self.unsigned_dict(), "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, value):
        expected = {
            "schema_version", "product", "version", "database_schema_version",
            "artifacts", "created_at", "content_hash",
        }
        if set(value) != expected:
            raise ValueError("invalid release-manifest fields")
        return cls(
            product=value["product"], version=value["version"],
            database_schema_version=int(value["database_schema_version"]),
            artifacts=tuple(ReleaseArtifact(**item) for item in value["artifacts"]),
            created_at=value["created_at"], content_hash=value["content_hash"],
            schema_version=int(value["schema_version"]),
        )


def create_release_manifest(
    root: str | Path, artifacts: Iterable[str | Path], *, version: str,
    platform: str, architecture: str, product: str = "PyMRST Studio",
    created_at: str | None = None,
) -> ReleaseManifest:
    root = Path(root).resolve()
    records = []
    kinds = {
        ".dmg": "macos_disk_image", ".msi": "windows_installer",
        ".exe": "windows_installer", ".appimage": "linux_appimage",
        ".deb": "linux_package", ".rpm": "linux_package",
        ".whl": "python_wheel", ".gz": "source_archive", ".zip": "archive",
    }
    for supplied in artifacts:
        path = Path(supplied)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"release artifact escapes root: {path}") from error
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"release artifact must be a regular non-symlink file: {path}")
        suffix = path.suffix.casefold()
        records.append(ReleaseArtifact(
            relative, kinds.get(suffix, "binary"), platform, architecture,
            path.stat().st_size, _digest(path),
        ))
    records.sort(key=lambda item: item.path)
    unsigned = {
        "schema_version": 1, "product": product, "version": version,
        "database_schema_version": APP_DATABASE_SCHEMA_VERSION,
        "artifacts": [item.to_dict() for item in records],
        "created_at": created_at or utc_now(),
    }
    return ReleaseManifest(
        product, version, APP_DATABASE_SCHEMA_VERSION, tuple(records),
        unsigned["created_at"], hashlib.sha256(_canonical(unsigned)).hexdigest(),
    )


def write_manifest(manifest: ReleaseManifest, target: str | Path) -> Path:
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        Path(temporary).replace(target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def read_manifest(path: str | Path) -> ReleaseManifest:
    return ReleaseManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def verify_release_manifest(manifest: ReleaseManifest, root: str | Path) -> dict:
    calculated = hashlib.sha256(_canonical(manifest.unsigned_dict())).hexdigest()
    if calculated != manifest.content_hash:
        return {"valid": False, "reason": "manifest content hash mismatch", "checked": 0}
    root, checked = Path(root).resolve(), 0
    for artifact in manifest.artifacts:
        path = (root / artifact.path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return {"valid": False, "reason": "artifact path escaped root", "path": artifact.path,
                    "checked": checked}
        if not path.is_file() or path.is_symlink():
            return {"valid": False, "reason": "artifact missing or not regular",
                    "path": artifact.path, "checked": checked}
        if path.stat().st_size != artifact.size_bytes:
            return {"valid": False, "reason": "artifact size mismatch",
                    "path": artifact.path, "checked": checked}
        if _digest(path) != artifact.sha256:
            return {"valid": False, "reason": "artifact checksum mismatch",
                    "path": artifact.path, "checked": checked}
        checked += 1
    return {"valid": True, "checked": checked, "content_hash": manifest.content_hash}


def sign_manifest(manifest: ReleaseManifest, private_key_pem: bytes) -> dict:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as error:
        raise ImportError("release signing requires `pip install pymrst[release]`") from error
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("release signing key must be Ed25519")
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    )
    signature = key.sign(_canonical(manifest.to_dict()))
    return {
        "schema_version": 1, "algorithm": "Ed25519",
        "key_id": hashlib.sha256(public).hexdigest()[:24],
        "manifest_content_hash": manifest.content_hash,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_manifest_signature(
    manifest: ReleaseManifest, signature_record: dict, public_key_pem: bytes,
) -> bool:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:
        raise ImportError("release verification requires `pip install pymrst[release]`") from error
    if set(signature_record) != {
        "schema_version", "algorithm", "key_id", "manifest_content_hash", "signature",
    } or signature_record["schema_version"] != 1 or signature_record["algorithm"] != "Ed25519":
        raise ValueError("invalid release signature record")
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("release verification key must be Ed25519")
    public = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if signature_record["key_id"] != hashlib.sha256(public).hexdigest()[:24]:
        return False
    if signature_record["manifest_content_hash"] != manifest.content_hash:
        return False
    try:
        key.verify(base64.b64decode(signature_record["signature"], validate=True),
                   _canonical(manifest.to_dict()))
        return True
    except (InvalidSignature, ValueError):
        return False


__all__ = [
    "ReleaseArtifact", "ReleaseManifest", "create_release_manifest", "read_manifest",
    "sign_manifest", "verify_manifest_signature", "verify_release_manifest",
    "write_manifest",
]
