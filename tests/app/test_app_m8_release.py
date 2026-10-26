from __future__ import annotations

import json

import pytest

from pymrst.app.release import (
    ReleaseManifest, create_release_manifest, read_manifest, sign_manifest,
    verify_manifest_signature, verify_release_manifest, write_manifest,
)


def test_release_manifest_is_deterministic_and_detects_artifact_tampering(tmp_path):
    (tmp_path / "PyMRST-Studio.dmg").write_bytes(b"signed-installer-payload")
    (tmp_path / "pymrst.whl").write_bytes(b"python-wheel-payload")
    manifest = create_release_manifest(
        tmp_path, ["pymrst.whl", "PyMRST-Studio.dmg"], version="0.1.0",
        platform="macos", architecture="arm64", created_at="2026-08-13T00:00:00+00:00",
    )
    assert [item.path for item in manifest.artifacts] == [
        "PyMRST-Studio.dmg", "pymrst.whl",
    ]
    target = write_manifest(manifest, tmp_path / "release-manifest.json")
    assert read_manifest(target) == manifest
    assert verify_release_manifest(manifest, tmp_path) == {
        "valid": True, "checked": 2, "content_hash": manifest.content_hash,
    }
    (tmp_path / "pymrst.whl").write_bytes(b"tampered")
    failure = verify_release_manifest(manifest, tmp_path)
    assert failure["valid"] is False
    assert failure["path"] == "pymrst.whl"
    assert failure["reason"] in {"artifact size mismatch", "artifact checksum mismatch"}


def test_manifest_rejects_path_traversal_and_external_artifacts(tmp_path):
    outside = tmp_path.parent / "outside.dmg"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes root"):
        create_release_manifest(
            tmp_path, [outside], version="1", platform="macos", architecture="arm64",
        )
    value = {
        "schema_version": 1, "product": "PyMRST Studio", "version": "1",
        "database_schema_version": 19, "created_at": "now", "content_hash": "0" * 64,
        "artifacts": [{"path": "../escape", "kind": "binary", "platform": "test",
                       "architecture": "test", "size_bytes": 0, "sha256": "0" * 64}],
    }
    with pytest.raises(ValueError, match="unsafe"):
        ReleaseManifest.from_dict(value)


def test_ed25519_signature_binds_manifest_and_public_key(tmp_path):
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    artifact = tmp_path / "studio.AppImage"
    artifact.write_bytes(b"release")
    manifest = create_release_manifest(
        tmp_path, [artifact], version="0.1.0", platform="linux", architecture="x86_64",
        created_at="2026-08-13T00:00:00+00:00",
    )
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signature = sign_manifest(manifest, private_pem)
    assert verify_manifest_signature(manifest, signature, public_pem)
    signature["manifest_content_hash"] = "0" * 64
    assert not verify_manifest_signature(manifest, signature, public_pem)


def test_version_sources_are_synchronized():
    from scripts.check_studio_release_versions import versions
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    found = versions(root)
    assert set(found.values()) == {"0.1.0"}
