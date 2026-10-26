"""Policies requiring immutable inputs and reviewable model patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .._serialization import frozen_json_mapping, json_value, require_fields
from ..permissions import AgentPermission, PermissionGrant
from .execution import DecisionCode, PolicyDecision


@dataclass(frozen=True, slots=True)
class CasePatchIntent:
    patch_id: str
    base_case_hash: str
    reason: str
    operations: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        for name in ("patch_id", "reason"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        digest = str(self.base_case_hash).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("base_case_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "base_case_hash", digest)
        operations = tuple(
            frozen_json_mapping(item, f"$.operations[{index}]")
            for index, item in enumerate(self.operations)
        )
        if not operations:
            raise ValueError("a case patch requires at least one operation")
        if any("operation" not in item for item in operations):
            raise ValueError("each case-patch operation requires an operation name")
        object.__setattr__(self, "operations", operations)

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "patch_id": self.patch_id, "base_case_hash": self.base_case_hash,
            "reason": self.reason, "operations": self.operations,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CasePatchIntent":
        data = require_fields(value, required={
            "patch_id", "base_case_hash", "reason", "operations",
        }, name="CasePatchIntent")
        data["operations"] = tuple(data["operations"])
        return cls(**data)


def authorize_case_patch(
    patch: CasePatchIntent,
    grant: PermissionGrant,
    *,
    approved: bool,
    writes_original: bool = False,
) -> PolicyDecision:
    if writes_original:
        return PolicyDecision(
            False, DecisionCode.DENY_MUTATION,
            "original cases are immutable; materialize a derived variant",
        )
    if grant.maximum < AgentPermission.CREATE_VARIANT:
        return PolicyDecision(
            False, DecisionCode.DENY_PERMISSION,
            "creating a case variant requires create_variant permission",
        )
    if not approved:
        return PolicyDecision(
            False, DecisionCode.REQUIRE_APPROVAL,
            f"case patch {patch.patch_id} requires explicit approval",
        )
    return PolicyDecision(
        True, DecisionCode.ALLOW,
        "patch may be materialized as a new immutable case variant",
    )


__all__ = ["CasePatchIntent", "authorize_case_patch"]
