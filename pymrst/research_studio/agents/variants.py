"""Immutable case-variant manifests owned by the engineering-agent layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from ._serialization import json_value, require_fields
from .policies.mutation import CasePatchIntent
from .policies.mutation import authorize_case_patch
from .permissions import PermissionGrant


CASE_VARIANT_SCHEMA_VERSION = 1


def _variant_hash(base_case_hash: str, operations: tuple[Mapping[str, Any], ...]) -> str:
    payload = json.dumps(
        json_value({"base_case_hash": base_case_hash, "operations": operations}),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CaseVariantManifest:
    variant_id: str
    base_case_id: str
    base_case_hash: str
    content_hash: str
    patch: CasePatchIntent
    schema_version: int = CASE_VARIANT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CASE_VARIANT_SCHEMA_VERSION:
            raise ValueError("unsupported case variant schema version")
        for name in ("variant_id", "base_case_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if self.patch.base_case_hash != self.base_case_hash:
            raise ValueError("variant patch and manifest base hashes disagree")
        expected = _variant_hash(self.base_case_hash, self.patch.operations)
        if self.content_hash != expected:
            raise ValueError("case variant content hash does not match its patch")

    @classmethod
    def create(
        cls, variant_id: str, base_case_id: str, patch: CasePatchIntent,
    ) -> "CaseVariantManifest":
        return cls(
            variant_id, base_case_id, patch.base_case_hash,
            _variant_hash(patch.base_case_hash, patch.operations), patch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "variant_id": self.variant_id,
            "base_case_id": self.base_case_id, "base_case_hash": self.base_case_hash,
            "content_hash": self.content_hash, "patch": self.patch.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CaseVariantManifest":
        data = require_fields(value, required={
            "schema_version", "variant_id", "base_case_id", "base_case_hash",
            "content_hash", "patch",
        }, name="CaseVariantManifest")
        data["patch"] = CasePatchIntent.from_dict(data["patch"])
        return cls(**data)


class CaseVariantCatalog:
    """Append-only variant registry; source cases are references, never write targets."""

    def __init__(self) -> None:
        self._variants: dict[str, CaseVariantManifest] = {}

    def register(self, manifest: CaseVariantManifest) -> CaseVariantManifest:
        if manifest.variant_id in self._variants:
            raise KeyError(f"case variant already registered: {manifest.variant_id}")
        self._variants[manifest.variant_id] = manifest
        return manifest

    def get(self, variant_id: str) -> CaseVariantManifest:
        try:
            return self._variants[variant_id]
        except KeyError as error:
            raise KeyError(f"unknown case variant: {variant_id}") from error

    def variants(self) -> tuple[CaseVariantManifest, ...]:
        return tuple(self._variants[key] for key in sorted(self._variants))


def register_case_variant(
    catalog: CaseVariantCatalog,
    *,
    variant_id: str,
    base_case_id: str,
    patch: CasePatchIntent,
    grant: PermissionGrant,
    approved: bool,
) -> CaseVariantManifest:
    """Authorize and register a derived manifest without writing the source case."""
    decision = authorize_case_patch(
        patch, grant, approved=approved, writes_original=False,
    )
    if not decision.allowed:
        raise PermissionError(decision.reason)
    return catalog.register(CaseVariantManifest.create(
        variant_id, base_case_id, patch,
    ))


__all__ = [
    "CASE_VARIANT_SCHEMA_VERSION", "CaseVariantCatalog", "CaseVariantManifest",
    "register_case_variant",
]
