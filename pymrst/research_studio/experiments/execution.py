"""Member execution protocol and independent directory result stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .definition import canonical_value
from .variants import VariantCoordinate


MEMBER_STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MemberExecutionContext:
    coordinate: VariantCoordinate
    seed: int
    output_directory: Path
    checkpoint: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MemberPayload:
    times: np.ndarray
    fields: Mapping[str, np.ndarray]
    wells: Mapping[str, np.ndarray] = field(default_factory=dict)
    regions: Mapping[str, np.ndarray] = field(default_factory=dict)
    convergence: Mapping[str, np.ndarray] = field(default_factory=dict)
    conservation: Mapping[str, np.ndarray] = field(default_factory=dict)
    performance: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if times.size == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("member times must be finite, nonempty, and increasing")
        object.__setattr__(self, "times", times)
        for category in ("fields", "wells", "regions", "convergence", "conservation"):
            values = {str(name): np.asarray(value) for name, value in getattr(self, category).items()}
            for name, value in values.items():
                if not name or value.ndim == 0 or value.shape[0] != times.size:
                    raise ValueError(f"{category}.{name} must be time-aligned")
                if value.dtype.kind not in "biufc" or np.any(~np.isfinite(value)):
                    raise ValueError(f"{category}.{name} must be finite numeric data")
            object.__setattr__(self, category, values)
        performance = {str(name): float(value) for name, value in self.performance.items()}
        if any(not np.isfinite(value) for value in performance.values()):
            raise ValueError("performance values must be finite")
        object.__setattr__(self, "performance", performance)
        object.__setattr__(self, "metadata", canonical_value(self.metadata))


@dataclass(frozen=True, slots=True)
class MemberOutcome:
    coordinate: VariantCoordinate
    seed: int
    status: str
    store_path: str
    elapsed_seconds: float
    resumed: bool = False
    error_type: str = ""
    error_message: str = ""

    @property
    def completed(self):
        return self.status == "completed"

    def to_dict(self):
        return {"coordinate": self.coordinate.to_dict(), "seed": self.seed,
                "status": self.status, "store_path": self.store_path,
                "elapsed_seconds": self.elapsed_seconds, "resumed": self.resumed,
                "error_type": self.error_type, "error_message": self.error_message}


class MemberResultStore:
    """One directory per member; arrays and control metadata commit atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.arrays_path = self.path / "arrays.npz"
        self.record_path = self.path / "member.json"
        self.checkpoint_path = self.path / "checkpoint.json"

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def begin(self, coordinate: VariantCoordinate, seed: int, runner: str):
        self.path.mkdir(parents=True, exist_ok=True)
        self._write_json(self.checkpoint_path, {
            "schema_version": MEMBER_STORE_SCHEMA_VERSION, "status": "running",
            "coordinate": coordinate.to_dict(), "seed": seed, "runner": runner,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        })

    def commit(self, payload: MemberPayload, *, coordinate: VariantCoordinate,
               seed: int, runner: str, elapsed_seconds: float):
        arrays = {"time": payload.times}
        layout = {}
        for category in ("fields", "wells", "regions", "convergence", "conservation"):
            layout[category] = {}
            for name, value in getattr(payload, category).items():
                key = f"{category}__{name}"
                arrays[key] = value
                layout[category][name] = {"key": key, "shape": list(value.shape),
                                          "dtype": value.dtype.str}
        temporary = self.arrays_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(self.arrays_path)
        record = {
            "schema_version": MEMBER_STORE_SCHEMA_VERSION, "status": "completed",
            "coordinate": coordinate.to_dict(), "seed": seed, "runner": runner,
            "elapsed_seconds": elapsed_seconds, "performance": dict(payload.performance),
            "metadata": dict(payload.metadata), "layout": layout,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json(self.record_path, record)
        self._write_json(self.checkpoint_path, record)

    def fail(self, *, coordinate: VariantCoordinate, seed: int, runner: str,
             elapsed_seconds: float, error: BaseException):
        record = {
            "schema_version": MEMBER_STORE_SCHEMA_VERSION, "status": "failed",
            "coordinate": coordinate.to_dict(), "seed": seed, "runner": runner,
            "elapsed_seconds": elapsed_seconds, "error_type": type(error).__name__,
            "error_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json(self.record_path, record)
        self._write_json(self.checkpoint_path, record)

    def record(self):
        return json.loads(self.record_path.read_text(encoding="utf-8"))

    def checkpoint(self):
        if not self.checkpoint_path.exists():
            return None
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

    def complete(self, coordinate: VariantCoordinate, runner: str) -> bool:
        if not self.record_path.exists() or not self.arrays_path.exists():
            return False
        record = self.record()
        return (record.get("status") == "completed"
                and record.get("coordinate") == coordinate.to_dict()
                and record.get("runner") == runner)

    def lazy(self):
        return LazyMemberResult(self)


class LazyMemberResult:
    def __init__(self, store: MemberResultStore):
        self.store = store
        self.record = store.record()

    @property
    def times(self):
        with np.load(self.store.arrays_path, allow_pickle=False) as arrays:
            return arrays["time"].copy()

    def names(self, category: str):
        return tuple(self.record["layout"].get(category, {}))

    def array(self, category: str, name: str, *, time_slice=None):
        key = self.record["layout"][category][name]["key"]
        with np.load(self.store.arrays_path, allow_pickle=False) as arrays:
            value = arrays[key]
            return value.copy() if time_slice is None else value[time_slice].copy()


def resolve_runner(owner: str):
    if ":" not in owner:
        raise ValueError("runner must use 'module:function' syntax")
    module, name = owner.split(":", 1)
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError(f"runner {owner!r} is not callable")
    return value


def execute_member(coordinate_dict, seed: int, directory: str, runner_owner: str):
    coordinate = VariantCoordinate.from_dict(coordinate_dict)
    store = MemberResultStore(directory)
    started = time.perf_counter()
    previous_checkpoint = store.checkpoint()
    store.begin(coordinate, seed, runner_owner)
    try:
        context = MemberExecutionContext(
            coordinate, seed, Path(directory), checkpoint=previous_checkpoint)
        payload = resolve_runner(runner_owner)(context)
        if not isinstance(payload, MemberPayload):
            raise TypeError("member runner must return MemberPayload")
        elapsed = time.perf_counter() - started
        store.commit(payload, coordinate=coordinate, seed=seed,
                     runner=runner_owner, elapsed_seconds=elapsed)
        return MemberOutcome(coordinate, seed, "completed", str(store.path), elapsed)
    except BaseException as error:
        elapsed = time.perf_counter() - started
        store.fail(coordinate=coordinate, seed=seed, runner=runner_owner,
                   elapsed_seconds=elapsed, error=error)
        return MemberOutcome(coordinate, seed, "failed", str(store.path), elapsed,
                             error_type=type(error).__name__, error_message=str(error))


__all__ = ["MEMBER_STORE_SCHEMA_VERSION", "LazyMemberResult",
           "MemberExecutionContext", "MemberOutcome", "MemberPayload",
           "MemberResultStore", "execute_member", "resolve_runner"]
