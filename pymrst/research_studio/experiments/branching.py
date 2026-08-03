"""Checkpoint branches with explicit ancestry and compatibility checks (M3.6)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .definition import canonical_value, content_digest
from .execution import MemberPayload, MemberResultStore, resolve_runner
from .variants import VariantCoordinate


BRANCH_SCHEMA_VERSION = 1


class IncompatibleBranch(ValueError):
    """A parent checkpoint cannot initialize the requested child model."""


@dataclass(frozen=True, slots=True)
class BranchDefinition:
    name: str
    parent_member_id: str
    parent_time_index: int = -1
    parameters: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, str] = field(default_factory=dict)
    controls: Mapping[str, Any] = field(default_factory=dict)
    properties: Mapping[str, Any] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()
    comparison_group: str = "branch"
    schema_version: int = BRANCH_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != BRANCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported branch schema {self.schema_version}")
        if not self.name or not self.parent_member_id or not self.comparison_group:
            raise ValueError("branch name, parent and comparison group are required")
        for name in ("parameters", "components", "controls", "properties"):
            object.__setattr__(self, name, canonical_value(getattr(self, name)))
        object.__setattr__(self, "required_fields",
                           tuple(str(value) for value in self.required_fields))

    @property
    def branch_id(self):
        return f"branch_{content_digest(self.to_dict(include_id=False))}"

    def to_dict(self, *, include_id=True):
        value = {
            "schema_version": self.schema_version, "name": self.name,
            "parent_member_id": self.parent_member_id,
            "parent_time_index": self.parent_time_index,
            "parameters": dict(self.parameters), "components": dict(self.components),
            "controls": dict(self.controls), "properties": dict(self.properties),
            "required_fields": list(self.required_fields),
            "comparison_group": self.comparison_group,
        }
        if include_id:
            value["branch_id"] = self.branch_id
        return value

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        claimed = data.pop("branch_id", None)
        branch = cls(**data)
        if claimed is not None and claimed != branch.branch_id:
            raise ValueError("branch identity mismatch")
        return branch


@dataclass(frozen=True, slots=True)
class BranchExecutionContext:
    definition: BranchDefinition
    parent_coordinate: VariantCoordinate
    parent_time: float
    parent_fields: Mapping[str, np.ndarray]
    seed: int
    output_directory: Path


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    branch_id: str
    member_id: str
    parent_member_id: str
    parent_time: float
    store_path: str
    elapsed_seconds: float
    status: str

    def to_dict(self):
        return {name: getattr(self, name) for name in (
            "branch_id", "member_id", "parent_member_id", "parent_time",
            "store_path", "elapsed_seconds", "status")}


def _parent_snapshot(parent, definition):
    times = parent.times
    index = definition.parent_time_index
    if index < 0:
        index += times.size
    if index < 0 or index >= times.size:
        raise IncompatibleBranch(
            f"parent time index {definition.parent_time_index} is outside {times.size} steps")
    available = set(parent.names("fields"))
    missing = sorted(set(definition.required_fields)-available)
    if missing:
        raise IncompatibleBranch(f"parent checkpoint is missing fields {missing}")
    fields = {name: parent.array("fields", name, time_slice=index)
              for name in sorted(available)}
    shapes = {name: tuple(value.shape) for name, value in fields.items()}
    return index, float(times[index]), fields, shapes


def execute_branch(parent: MemberResultStore, definition: BranchDefinition, *,
                   output: str | Path, runner: str, seed: int = 0) -> BranchOutcome:
    """Execute a child from exactly one persisted parent report state."""
    lazy = parent.lazy()
    parent_coordinate = VariantCoordinate.from_dict(lazy.record["coordinate"])
    if str(parent_coordinate.member_id) != definition.parent_member_id:
        raise IncompatibleBranch(
            "branch parent_member_id does not identify the supplied result store")
    index, parent_time, fields, shapes = _parent_snapshot(lazy, definition)
    member_digest = content_digest({
        "branch_id": definition.branch_id,
        "parent_store_schema": lazy.record["schema_version"],
        "parent_shapes": shapes,
    })
    coordinate = VariantCoordinate(
        parent_coordinate.experiment_id, f"member_{member_digest}", 0,
        definition.name, definition.comparison_group,
        definition.parameters,
        {**parent_coordinate.components, **definition.components},
    )
    store = MemberResultStore(Path(output) / definition.branch_id /
                              str(coordinate.member_id))
    store.begin(coordinate, int(seed), runner)
    started = time.perf_counter()
    try:
        context = BranchExecutionContext(definition, parent_coordinate, parent_time,
                                         fields, int(seed), store.path)
        payload = resolve_runner(runner)(context)
        if not isinstance(payload, MemberPayload):
            raise TypeError("branch runner must return MemberPayload")
        if payload.times[0] <= parent_time:
            raise IncompatibleBranch(
                "branch output times must begin strictly after the parent checkpoint")
        metadata = dict(payload.metadata)
        metadata["branch_lineage"] = {
            "branch_id": definition.branch_id,
            "parent_member_id": definition.parent_member_id,
            "parent_time_index": index, "parent_time": parent_time,
            "changes": {name: getattr(definition, name) for name in
                        ("parameters", "components", "controls", "properties")},
        }
        payload = replace(payload, metadata=metadata)
        elapsed = time.perf_counter()-started
        store.commit(payload, coordinate=coordinate, seed=int(seed), runner=runner,
                     elapsed_seconds=elapsed)
    except BaseException as error:
        store.fail(coordinate=coordinate, seed=int(seed), runner=runner,
                   elapsed_seconds=time.perf_counter()-started, error=error)
        raise
    MemberResultStore._write_json(store.path / "branch.json", definition.to_dict())
    return BranchOutcome(definition.branch_id, str(coordinate.member_id),
                         definition.parent_member_id, parent_time,
                         str(store.path), elapsed, "completed")


def verify_branch_lineage(outcome: BranchOutcome):
    store = MemberResultStore(outcome.store_path)
    branch = BranchDefinition.from_dict(json.loads(
        (store.path / "branch.json").read_text(encoding="utf-8")))
    lineage = store.record()["metadata"]["branch_lineage"]
    if branch.branch_id != outcome.branch_id:
        raise ValueError("branch definition and outcome identities differ")
    if lineage["parent_member_id"] != outcome.parent_member_id:
        raise ValueError("stored branch ancestry differs from the outcome")
    return branch


__all__ = ["BRANCH_SCHEMA_VERSION", "BranchDefinition", "BranchExecutionContext",
           "BranchOutcome", "IncompatibleBranch", "execute_branch",
           "verify_branch_lineage"]
