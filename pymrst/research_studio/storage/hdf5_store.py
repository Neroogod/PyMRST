"""Transactional HDF5 ResultStore v1 without pickle-based persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from pymrst.research_studio.evidence.report import SimulationEvidence
from pymrst.research_studio.evidence.events import SimulationEvent
from pymrst.research_studio.runtime.identifiers import CheckpointId, RunId
from pymrst.research_studio.storage.result_store import StoreCorruptionError

STORE_SCHEMA_VERSION = 1


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "HDF5ResultStore requires h5py; install pymrst[studio]"
        ) from error
    return h5py


def _validate_key(key: Any) -> str:
    key = str(key)
    if not key or "/" in key or key in {".", ".."}:
        raise ValueError(f"invalid persistent mapping key {key!r}")
    return key


def _write_value(parent, name: str, value: Any) -> None:
    h5py = _h5py()
    name = _validate_key(name)
    if isinstance(value, Mapping):
        group = parent.create_group(name)
        group.attrs["pymrst_type"] = "mapping"
        for key, item in value.items():
            _write_value(group, _validate_key(key), item)
        return
    if isinstance(value, (list, tuple)):
        group = parent.create_group(name)
        group.attrs["pymrst_type"] = "tuple" if isinstance(value, tuple) else "list"
        group.attrs["length"] = len(value)
        for index, item in enumerate(value):
            _write_value(group, str(index), item)
        return
    if value is None:
        group = parent.create_group(name)
        group.attrs["pymrst_type"] = "none"
        return
    if isinstance(value, str):
        dataset = parent.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
        dataset.attrs["pymrst_type"] = "str"
        return
    if isinstance(value, (bool, int, float, complex, np.generic, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise TypeError(f"object arrays are not persistable at {name!r}")
        if array.dtype.kind == "U":
            dtype = h5py.string_dtype("utf-8")
            dataset = parent.create_dataset(name, data=array.astype(object), dtype=dtype)
        else:
            dataset = parent.create_dataset(name, data=array)
        dataset.attrs["pymrst_type"] = "ndarray" if array.ndim else "scalar"
        return
    raise TypeError(
        f"unsupported persistent state value {type(value).__name__} at {name!r}"
    )


def _read_value(node) -> Any:
    h5py = _h5py()
    kind = node.attrs.get("pymrst_type")
    if isinstance(kind, bytes):
        kind = kind.decode("utf-8")
    if isinstance(node, h5py.Dataset):
        if kind not in {"str", "scalar", "ndarray"}:
            raise StoreCorruptionError(f"unknown dataset type marker {kind!r}")
        value = node[()]
        if kind == "str":
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if kind == "scalar":
            return value.item() if isinstance(value, np.generic) else value
        if getattr(value, "dtype", None) is not None and value.dtype.kind in {"S", "O"}:
            return value.astype(str)
        return value
    if kind == "none":
        return None
    if kind == "mapping":
        return {name: _read_value(node[name]) for name in node}
    if kind in {"list", "tuple"}:
        length = int(node.attrs.get("length", -1))
        if length < 0 or set(node) != {str(index) for index in range(length)}:
            raise StoreCorruptionError("invalid persistent sequence layout")
        values = [_read_value(node[str(index)]) for index in range(length)]
        return tuple(values) if kind == "tuple" else values
    raise StoreCorruptionError(f"unknown group type marker {kind!r}")


class LazyStateSequence(Sequence):
    """Open-on-access state view that does not retain an HDF5 file handle."""

    def __init__(self, store: "HDF5ResultStore", run_id: RunId):
        self.store, self.run_id = store, RunId(run_id)

    def __len__(self) -> int:
        return len(self.store.state_indices(self.run_id))

    def __getitem__(self, item):
        indices = self.store.state_indices(self.run_id)
        if isinstance(item, slice):
            return [self.store.load_state(self.run_id, index) for index in indices[item]]
        return self.store.load_state(self.run_id, indices[item])


class HDF5ResultStore:
    """One-file transactional store with explicit completion markers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _open(self, mode="a"):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return _h5py().File(self.path, mode)

    @staticmethod
    def _run(file, run_id: RunId):
        path = f"runs/{RunId(run_id)}"
        if path not in file:
            raise KeyError(f"run {run_id!s} does not exist")
        run = file[path]
        if int(run.attrs.get("store_schema_version", -1)) != STORE_SCHEMA_VERSION:
            raise StoreCorruptionError("unsupported or missing ResultStore schema version")
        return run

    def begin_run(self, run_id: RunId, manifest: Mapping[str, Any]) -> None:
        run_id = RunId(run_id)
        with self._open() as file:
            runs = file.require_group("runs")
            if str(run_id) in runs:
                raise FileExistsError(f"run {run_id} already exists")
            run = runs.create_group(str(run_id))
            run.attrs.update({
                "store_schema_version": STORE_SCHEMA_VERSION,
                "status": "running",
                "complete": False,
            })
            run.create_group("accepted_states")
            run.create_group("rejected_attempts")
            run.create_group("checkpoints")
            run.create_group("events")
            run.create_group("_pending")
            _write_value(run, "manifest", dict(manifest))
            file.flush()

    def _commit_mapping(self, run, destination: str, value: Mapping[str, Any]) -> None:
        token = uuid4().hex
        pending_name = f"_pending/{token}"
        try:
            _write_value(run["_pending"], token, dict(value))
        except Exception:
            if pending_name in run:
                del run[pending_name]
                run.file.flush()
            raise
        run.file.flush()
        if destination in run:
            del run[pending_name]
            raise FileExistsError(f"artifact {destination!r} already committed")
        run.move(pending_name, destination)
        run.file.flush()

    def commit_state(self, run_id: RunId, index: int, state: Mapping[str, Any], *, accepted: bool = True) -> None:
        if index < 0:
            raise ValueError("state index must be non-negative")
        if not isinstance(state, Mapping):
            raise TypeError("state must be an explicit mapping")
        collection = "accepted_states" if accepted else "rejected_attempts"
        with self._open() as file:
            run = self._run(file, run_id)
            if bool(run.attrs["complete"]):
                raise RuntimeError("cannot commit state to finalized run")
            self._commit_mapping(run, f"{collection}/{index:08d}", state)

    def commit_evidence(self, run_id: RunId, evidence: SimulationEvidence) -> None:
        payload = evidence.to_json()
        with self._open() as file:
            run = self._run(file, run_id)
            pending = f"_pending/{uuid4().hex}"
            dataset = run.create_dataset(pending, data=payload, dtype=_h5py().string_dtype("utf-8"))
            dataset.attrs["pymrst_type"] = "simulation_evidence_json"
            run.file.flush()
            if "evidence" in run:
                del run["evidence"]
            run.move(pending, "evidence")
            run.file.flush()

    def commit_event(self, run_id: RunId, event: SimulationEvent) -> None:
        if RunId(event.run_id) != RunId(run_id):
            raise ValueError("event run_id does not match destination run")
        payload = json.dumps(event.to_dict(), sort_keys=True, allow_nan=False)
        with self._open() as file:
            run = self._run(file, run_id)
            if bool(run.attrs["complete"]):
                raise RuntimeError("cannot commit event to finalized run")
            destination = f"events/{event.sequence:012d}"
            pending = f"_pending/{uuid4().hex}"
            dataset = run.create_dataset(
                pending, data=payload, dtype=_h5py().string_dtype("utf-8"),
            )
            dataset.attrs["pymrst_type"] = "simulation_event_json"
            run.file.flush()
            if destination in run:
                del run[pending]
                raise FileExistsError(f"event sequence {event.sequence} already committed")
            run.move(pending, destination)
            run.file.flush()

    def commit_checkpoint(self, run_id: RunId, checkpoint_id: CheckpointId, data: Mapping[str, Any]) -> None:
        with self._open() as file:
            run = self._run(file, run_id)
            if bool(run.attrs["complete"]):
                raise RuntimeError("cannot commit checkpoint to finalized run")
            self._commit_mapping(run, f"checkpoints/{CheckpointId(checkpoint_id)}", data)

    def load_state(self, run_id: RunId, index: int) -> dict[str, Any]:
        with self._open("r") as file:
            run = self._run(file, run_id)
            name = f"accepted_states/{index:08d}"
            if name not in run:
                raise KeyError(f"accepted state {index} does not exist")
            value = _read_value(run[name])
            if not isinstance(value, dict):
                raise StoreCorruptionError("committed state root is not a mapping")
            return value

    def load_checkpoint(self, run_id: RunId, checkpoint_id: CheckpointId) -> dict[str, Any]:
        with self._open("r") as file:
            run = self._run(file, run_id)
            name = f"checkpoints/{CheckpointId(checkpoint_id)}"
            if name not in run:
                raise KeyError(f"checkpoint {checkpoint_id} does not exist")
            value = _read_value(run[name])
            if not isinstance(value, dict):
                raise StoreCorruptionError("checkpoint root is not a mapping")
            return value

    def load_evidence(self, run_id: RunId) -> SimulationEvidence:
        with self._open("r") as file:
            run = self._run(file, run_id)
            if "evidence" not in run:
                raise KeyError("run has no committed evidence")
            dataset = run["evidence"]
            if dataset.attrs.get("pymrst_type") != "simulation_evidence_json":
                raise StoreCorruptionError("invalid evidence marker")
            value = dataset[()]
            payload = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return SimulationEvidence.from_json(payload)

    def load_event_records(self, run_id: RunId) -> tuple[dict[str, Any], ...]:
        with self._open("r") as file:
            run = self._run(file, run_id)
            names = sorted(run["events"])
            expected = [f"{index:012d}" for index in range(len(names))]
            if names != expected:
                raise StoreCorruptionError("event sequence is not contiguous from zero")
            records = []
            for name in names:
                dataset = run["events"][name]
                if dataset.attrs.get("pymrst_type") != "simulation_event_json":
                    raise StoreCorruptionError("invalid event marker")
                value = dataset[()]
                payload = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                record = json.loads(payload)
                if int(record.get("sequence", -1)) != len(records):
                    raise StoreCorruptionError("event payload sequence mismatch")
                records.append(record)
            return tuple(records)

    def replay_events(self, run_id: RunId, handler) -> int:
        records = self.load_event_records(run_id)
        for record in records:
            handler(dict(record))
        return len(records)

    def state_indices(self, run_id: RunId) -> tuple[int, ...]:
        with self._open("r") as file:
            run = self._run(file, run_id)
            try:
                return tuple(sorted(int(name) for name in run["accepted_states"]))
            except ValueError as error:
                raise StoreCorruptionError("invalid accepted-state name") from error

    def lazy_states(self, run_id: RunId) -> LazyStateSequence:
        return LazyStateSequence(self, run_id)

    def finalize_run(self, run_id: RunId, status: str) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid terminal run status")
        with self._open() as file:
            run = self._run(file, run_id)
            if "evidence" not in run:
                raise RuntimeError("cannot finalize before evidence is committed")
            run.attrs["status"] = status
            run.attrs["complete"] = True
            file.flush()

    def inspect_run(self, run_id: RunId) -> dict[str, Any]:
        with self._open("r") as file:
            run = self._run(file, run_id)
            return {
                "status": str(run.attrs["status"]),
                "complete": bool(run.attrs["complete"]),
                "accepted_states": len(run["accepted_states"]),
                "rejected_attempts": len(run["rejected_attempts"]),
                "checkpoints": len(run["checkpoints"]),
                "events": len(run["events"]),
                "pending_artifacts": len(run["_pending"]),
            }

    def discard_pending_artifacts(self, run_id: RunId) -> int:
        """Recover after interruption by deleting only uncommitted artifacts."""
        with self._open() as file:
            run = self._run(file, run_id)
            pending = run["_pending"]
            names = list(pending)
            for name in names:
                del pending[name]
            file.flush()
            return len(names)


__all__ = ["HDF5ResultStore", "LazyStateSequence", "STORE_SCHEMA_VERSION"]
