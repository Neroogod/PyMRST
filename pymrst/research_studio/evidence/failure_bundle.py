"""Portable, checksummed failure bundles without arbitrary object pickles."""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from pymrst.research_studio.evidence.manifest import ReproducibilityManifest
from pymrst.research_studio.evidence.report import SimulationEvidence
from pymrst.research_studio.storage.checkpoint import CheckpointRecord
from pymrst.research_studio.storage.checkpoint import (
    default_model_fingerprint, default_solver_fingerprint, restart_payload,
)

FAILURE_BUNDLE_SCHEMA_VERSION = 1


def _encode(value: Any, arrays: dict[str, bytes], path="root") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return {"nonfinite": repr(value)}
        return value
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays, path)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError(f"object array is not allowed in failure bundle at {path}")
        name = f"arrays/{len(arrays):06d}.npy"
        stream = BytesIO()
        np.save(stream, value, allow_pickle=False)
        arrays[name] = stream.getvalue()
        return {"array": name, "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _encode(item, arrays, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "sequence": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(item, arrays, f"{path}[{index}]") for index, item in enumerate(value)],
        }
    raise TypeError(f"unsupported failure-bundle value {type(value).__name__} at {path}")


def _decode(value: Any, archive: ZipFile) -> Any:
    if isinstance(value, list):
        return [_decode(item, archive) for item in value]
    if not isinstance(value, dict):
        return value
    if "array" in value and set(value) == {"array", "dtype", "shape"}:
        with archive.open(value["array"]) as member:
            array = np.load(BytesIO(member.read()), allow_pickle=False)
        if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
            raise ValueError(f"array metadata mismatch for {value['array']}")
        return array
    if value.get("sequence") in {"list", "tuple"} and set(value) == {"sequence", "items"}:
        items = [_decode(item, archive) for item in value["items"]]
        return tuple(items) if value["sequence"] == "tuple" else items
    return {key: _decode(item, archive) for key, item in value.items()}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_failure_bundle(
    path: str | Path,
    *,
    manifest: ReproducibilityManifest,
    evidence: SimulationEvidence,
    checkpoint: CheckpointRecord,
    failing_step: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, bytes] = {}
    payload = {
        "schema_version": FAILURE_BUNDLE_SCHEMA_VERSION,
        "manifest": manifest.to_dict(),
        "evidence": evidence.to_dict(),
        "checkpoint": _encode(checkpoint.to_dict(), arrays, "checkpoint"),
        "failing_step": _encode(dict(failing_step), arrays, "failing_step"),
    }
    members = {
        "bundle.json": json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8"),
        **arrays,
    }
    index = {
        "schema_version": FAILURE_BUNDLE_SCHEMA_VERSION,
        "members": {name: _digest(data) for name, data in sorted(members.items())},
    }
    members["checksums.json"] = json.dumps(index, sort_keys=True).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_failure_bundle(path: str | Path) -> dict[str, Any]:
    with ZipFile(path, "r") as archive:
        index = json.loads(archive.read("checksums.json"))
        if index.get("schema_version") != FAILURE_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported failure bundle schema")
        for name, expected in index["members"].items():
            try:
                data = archive.read(name)
            except KeyError as error:
                raise ValueError(f"failure bundle member missing: {name}") from error
            if _digest(data) != expected:
                raise ValueError(f"failure bundle checksum mismatch: {name}")
        payload = json.loads(archive.read("bundle.json"))
        if payload.get("schema_version") != FAILURE_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported failure bundle payload schema")
        return {
            "manifest": ReproducibilityManifest.from_dict(payload["manifest"]),
            "evidence": SimulationEvidence.from_dict(payload["evidence"]),
            "checkpoint": CheckpointRecord.from_dict(_decode(payload["checkpoint"], archive)),
            "failing_step": _decode(payload["failing_step"], archive),
        }


def write_store_failure_bundle(
    path: str | Path, *, store, run_id, manifest, evidence, failing_step,
    overwrite: bool = False,
) -> Path:
    if not evidence.checkpoints:
        raise ValueError("failure bundle requires at least one committed checkpoint")
    checkpoint_id = evidence.checkpoints[-1]["checkpoint_id"]
    checkpoint = CheckpointRecord.from_dict(
        store.load_checkpoint(run_id, checkpoint_id)
    )
    return write_failure_bundle(
        path, manifest=manifest, evidence=evidence, checkpoint=checkpoint,
        failing_step=failing_step, overwrite=overwrite,
    )


@dataclass(frozen=True, slots=True)
class FailureReplay:
    """Validated numerical inputs reconstructed from a failure bundle.

    Model, solver and schedule remain caller-owned executable objects; the
    portable bundle never deserializes arbitrary code. Compatibility hashes
    certify that those objects match the accepted checkpoint before ``run``
    transfers control to the native solver.
    """

    model: Any
    nonlinear_solver: Any
    state: Any
    remaining_schedule: Any
    manifest: ReproducibilityManifest
    evidence: SimulationEvidence
    checkpoint: CheckpointRecord
    failing_step: Mapping[str, Any]

    def run(self):
        return self.nonlinear_solver.simulate_schedule(
            self.model, self.state, self.remaining_schedule,
        )


def prepare_failure_replay(
    path: str | Path,
    *,
    model: Any,
    nonlinear_solver: Any,
    full_schedule: Any,
    code_fingerprint: str,
    state_factory=dict,
    verify_inputs: bool = True,
) -> FailureReplay:
    """Load, verify and reconstruct a safe report-boundary failure replay."""
    loaded = load_failure_bundle(path)
    manifest = loaded["manifest"]
    checkpoint = loaded["checkpoint"]
    if manifest.code_fingerprint != code_fingerprint:
        raise ValueError("failure bundle manifest code fingerprint changed")
    if verify_inputs:
        manifest.verify_inputs()
    state, remaining = restart_payload(
        checkpoint,
        schedule=full_schedule,
        model_fingerprint=default_model_fingerprint(model),
        solver_fingerprint=default_solver_fingerprint(nonlinear_solver),
        code_fingerprint=code_fingerprint,
        state_factory=state_factory,
    )
    return FailureReplay(
        model=model, nonlinear_solver=nonlinear_solver, state=state,
        remaining_schedule=remaining, manifest=manifest,
        evidence=loaded["evidence"], checkpoint=checkpoint,
        failing_step=loaded["failing_step"],
    )


__all__ = [
    "FAILURE_BUNDLE_SCHEMA_VERSION", "FailureReplay", "load_failure_bundle",
    "prepare_failure_replay", "write_failure_bundle", "write_store_failure_bundle",
]
