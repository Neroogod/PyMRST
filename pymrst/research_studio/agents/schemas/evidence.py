"""Stable references from engineering claims to inspectable evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .._serialization import frozen_json_mapping, json_value, require_fields


class EvidenceKind(StrEnum):
    INPUT = "input"
    DECK_RECORD = "deck_record"
    GRID_OBJECT = "grid_object"
    RUN = "run"
    STATE = "state"
    CONVERGENCE = "convergence"
    CONSERVATION = "conservation"
    TABLE = "table"
    PLOT = "plot"
    VALIDATION = "validation"
    EXTERNAL_REFERENCE = "external_reference"


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """A portable pointer, not a live simulator or filesystem object.

    ``uri`` uses an application-owned scheme such as ``run://run_1`` or
    ``deck://case_1/PROPS.INC``. ``locator`` carries kind-specific coordinates
    such as line, keyword, cell, well, report step, or dataset path.
    """

    kind: EvidenceKind
    uri: str
    label: str
    locator: Mapping[str, Any] = field(default_factory=dict)
    checksum: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        uri = str(self.uri).strip()
        label = str(self.label).strip()
        if not uri or "://" not in uri:
            raise ValueError("evidence uri must be a nonempty application URI")
        if not label:
            raise ValueError("evidence label cannot be empty")
        checksum = None if self.checksum is None else str(self.checksum).strip()
        if checksum == "":
            raise ValueError("evidence checksum cannot be empty")
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "checksum", checksum)
        object.__setattr__(
            self, "locator", frozen_json_mapping(self.locator, "$.locator")
        )

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "kind": self.kind.value,
            "uri": self.uri,
            "label": self.label,
            "locator": self.locator,
            "checksum": self.checksum,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceReference":
        data = require_fields(
            value,
            required={"kind", "uri", "label", "locator", "checksum"},
            name="EvidenceReference",
        )
        return cls(**data)


__all__ = ["EvidenceKind", "EvidenceReference"]
