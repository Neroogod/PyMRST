"""Bounded local-process ensemble execution with member isolation and resume."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from .execution import MemberOutcome, MemberResultStore, execute_member
from .variants import MaterializedVariantMatrix, VariantCoordinate


ENSEMBLE_REPORT_SCHEMA_VERSION = 1


def deterministic_member_seed(experiment_seed: int, coordinate: VariantCoordinate) -> int:
    if experiment_seed < 0:
        raise ValueError("experiment seed must be nonnegative")
    payload = f"{experiment_seed}:{coordinate.member_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63)-1)


def available_memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


@dataclass(frozen=True, slots=True)
class EnsembleReport:
    matrix_id: str
    runner: str
    outcomes: tuple[MemberOutcome, ...]
    requested_workers: int
    effective_workers: int
    memory_per_worker_bytes: int
    execution_backend: str = "process_pool"
    schema_version: int = ENSEMBLE_REPORT_SCHEMA_VERSION

    @property
    def completed(self):
        return tuple(value for value in self.outcomes if value.completed)

    @property
    def failed(self):
        return tuple(value for value in self.outcomes if not value.completed)

    def to_dict(self):
        return {"schema_version": self.schema_version, "matrix_id": self.matrix_id,
                "runner": self.runner, "requested_workers": self.requested_workers,
                "effective_workers": self.effective_workers,
                "memory_per_worker_bytes": self.memory_per_worker_bytes,
                "execution_backend": self.execution_backend,
                "outcomes": [outcome.to_dict() for outcome in self.outcomes],
                "summary": {"completed": len(self.completed), "failed": len(self.failed),
                            "resumed": sum(value.resumed for value in self.outcomes)}}


class LocalEnsembleExecutor:
    def __init__(self, root: str | Path, *, max_workers: int | None = None,
                 memory_per_worker_bytes: int = 512*1024*1024,
                 memory_fraction: float = 0.8):
        self.root = Path(root)
        requested = (os.cpu_count() or 1) if max_workers is None else int(max_workers)
        if requested <= 0 or memory_per_worker_bytes <= 0 or not 0 < memory_fraction <= 1:
            raise ValueError("worker and memory limits must be positive")
        self.requested_workers = requested
        self.memory_per_worker_bytes = int(memory_per_worker_bytes)
        available = available_memory_bytes()
        memory_workers = requested if available is None else max(
            1, int(available*memory_fraction)//self.memory_per_worker_bytes)
        self.max_workers = max(1, min(requested, memory_workers))

    def _member_store(self, matrix_id: str, coordinate: VariantCoordinate):
        return MemberResultStore(self.root / matrix_id / str(coordinate.member_id))

    def run(self, materialized: MaterializedVariantMatrix, *, runner: str,
            retry_failed: bool = True) -> EnsembleReport:
        self.root.mkdir(parents=True, exist_ok=True)
        outcomes: dict[int, MemberOutcome] = {}
        pending = []
        for coordinate in materialized.coordinates:
            seed = deterministic_member_seed(materialized.definition.random_seed, coordinate)
            store = self._member_store(materialized.matrix_id, coordinate)
            if store.complete(coordinate, runner):
                record = store.record()
                outcomes[coordinate.ordinal] = MemberOutcome(
                    coordinate, seed, "completed", str(store.path),
                    float(record["elapsed_seconds"]), resumed=True)
                continue
            if store.record_path.exists() and not retry_failed:
                record = store.record()
                outcomes[coordinate.ordinal] = MemberOutcome(
                    coordinate, seed, "failed", str(store.path),
                    float(record.get("elapsed_seconds", 0.0)), resumed=True,
                    error_type=record.get("error_type", "PreviousFailure"),
                    error_message=record.get("error_message", ""))
                continue
            pending.append((coordinate, seed, store))

        backend = "resume_only"
        if pending:
            backend = "process_pool"
            try:
                with ProcessPoolExecutor(max_workers=min(self.max_workers, len(pending))) as pool:
                    futures = {
                        pool.submit(execute_member, coordinate.to_dict(), seed,
                                    str(store.path), runner): coordinate
                        for coordinate, seed, store in pending
                    }
                    for future in as_completed(futures):
                        coordinate = futures[future]
                        try:
                            outcomes[coordinate.ordinal] = future.result()
                        except BaseException as error:
                            self._record_executor_failure(
                                outcomes, materialized, coordinate, runner, error)
            except (NotImplementedError, PermissionError):
                backend = "sequential_fallback"
                for coordinate, seed, store in pending:
                    outcomes[coordinate.ordinal] = execute_member(
                        coordinate.to_dict(), seed, str(store.path), runner)
        report = EnsembleReport(
            materialized.matrix_id, runner,
            tuple(outcomes[index] for index in sorted(outcomes)),
            self.requested_workers, self.max_workers, self.memory_per_worker_bytes,
            backend)
        report_path = self.root / materialized.matrix_id / "ensemble_report.json"
        MemberResultStore._write_json(report_path, report.to_dict())
        return report

    def _record_executor_failure(self, outcomes, materialized, coordinate, runner, error):
        store = self._member_store(materialized.matrix_id, coordinate)
        seed = deterministic_member_seed(materialized.definition.random_seed, coordinate)
        store.fail(coordinate=coordinate, seed=seed, runner=runner,
                   elapsed_seconds=0.0, error=error)
        outcomes[coordinate.ordinal] = MemberOutcome(
            coordinate, seed, "failed", str(store.path), 0.0,
            error_type=type(error).__name__, error_message=str(error))

    @staticmethod
    def aggregate(report: EnsembleReport) -> Mapping[str, float]:
        """Stream scalar records only; field arrays never enter aggregation memory."""
        elapsed = 0.0
        performance: dict[str, list[float]] = {}
        for outcome in report.completed:
            record = MemberResultStore(outcome.store_path).record()
            elapsed += float(record["elapsed_seconds"])
            for name, value in record.get("performance", {}).items():
                performance.setdefault(name, []).append(float(value))
        result = {"completed_members": float(len(report.completed)),
                  "failed_members": float(len(report.failed)),
                  "summed_member_elapsed_seconds": elapsed}
        for name, values in performance.items():
            result[f"{name}_mean"] = sum(values)/len(values)
            result[f"{name}_minimum"] = min(values)
            result[f"{name}_maximum"] = max(values)
        return result


__all__ = ["ENSEMBLE_REPORT_SCHEMA_VERSION", "EnsembleReport",
           "LocalEnsembleExecutor", "available_memory_bytes",
           "deterministic_member_seed"]
