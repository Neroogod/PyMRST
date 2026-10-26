"""Trusted process-isolated adapters for Research Studio ensembles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pymrst.research_studio.experiments import (
    BranchDefinition, LocalEnsembleExecutor, MaterializedVariantMatrix,
    MemberResultStore, execute_branch,
)


MEMBER_RUNNERS = {
    "manufactured-qualification":
        "pymrst.research_studio.experiments.runners:deterministic_manufactured_member",
    "spe1-time-integration":
        "pymrst.research_studio.experiments.runners:spe1_time_integration_member",
}
BRANCH_RUNNERS = {
    "manufactured-branch":
        "pymrst.research_studio.experiments.runners:deterministic_branch_member",
    "spe1-branch":
        "pymrst.research_studio.experiments.runners:spe1_branch_member",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uncertainty_summary(report) -> dict:
    """Aggregate scalar performance and final-state member statistics."""
    values: dict[str, list[float]] = {}
    for outcome in report.completed:
        result = MemberResultStore(outcome.store_path).lazy()
        for name, item in result.record.get("performance", {}).items():
            values.setdefault(f"performance.{name}", []).append(float(item))
        for category in ("fields", "wells", "regions"):
            for name in result.names(category):
                final = np.asarray(result.array(category, name, time_slice=-1), dtype=float)
                values.setdefault(f"{category}.{name}.final_mean", []).append(
                    float(np.mean(final)))
    summary = {}
    for name, items in sorted(values.items()):
        array = np.asarray(items, dtype=float)
        summary[name] = {
            "count": int(array.size), "minimum": float(array.min()),
            "p10": float(np.quantile(array, 0.10)),
            "p50": float(np.quantile(array, 0.50)),
            "p90": float(np.quantile(array, 0.90)),
            "maximum": float(array.max()), "mean": float(array.mean()),
            "standard_deviation": float(array.std()),
        }
    return summary


def execute_ensemble(materialized_path: Path, output: Path, *, runner_kind: str,
                     max_workers: int, memory_per_worker_bytes: int,
                     progress=None) -> dict:
    if runner_kind not in MEMBER_RUNNERS:
        raise ValueError(f"unregistered ensemble member runner: {runner_kind}")
    materialized_path, output = materialized_path.resolve(), output.resolve()
    materialized = MaterializedVariantMatrix.from_dict(
        json.loads(materialized_path.read_text(encoding="utf-8")))
    executor = LocalEnsembleExecutor(
        output, max_workers=max_workers,
        memory_per_worker_bytes=memory_per_worker_bytes,
    )
    report = executor.run(
        materialized, runner=MEMBER_RUNNERS[runner_kind],
        progress_callback=progress,
    )
    return {
        "report": report.to_dict(), "aggregate": dict(executor.aggregate(report)),
        "uncertainty": uncertainty_summary(report),
    }


def run_experiment_ensemble(context, request):
    path = Path(request["materialized_path"])
    if _hash(path) != request["materialized_sha256"]:
        raise RuntimeError("materialized experiment checksum mismatch")
    def update(completed, total, outcome):
        context.raise_if_cancelled()
        context.progress(
            completed/total, f"Ensemble member {completed}/{total}",
            completed_members=completed, total_members=total,
            member_id=str(outcome.coordinate.member_id), member_status=outcome.status,
        )
    return execute_ensemble(
        path, context.run_directory / "ensemble",
        runner_kind=request["member_runner"],
        max_workers=int(request["max_workers"]),
        memory_per_worker_bytes=int(request["memory_per_worker_bytes"]),
        progress=update,
    )


def run_experiment_branch(context, request):
    runner_kind = str(request["runner_kind"])
    if runner_kind not in BRANCH_RUNNERS:
        raise ValueError(f"unregistered branch runner: {runner_kind}")
    parent = Path(request["parent_store"]).resolve()
    record = parent / "member.json"
    if _hash(record) != request["parent_record_sha256"]:
        raise RuntimeError("branch parent checksum mismatch")
    definition = BranchDefinition.from_dict(request["definition"])
    context.progress(0.1, "Verified parent checkpoint",
                     parent_member_id=definition.parent_member_id)
    outcome = execute_branch(
        MemberResultStore(parent), definition,
        output=context.run_directory / "branch",
        runner=BRANCH_RUNNERS[runner_kind], seed=int(request.get("seed", 0)),
    )
    context.progress(0.95, "Branch child committed", member_id=outcome.member_id)
    return {"outcome": outcome.to_dict(), "definition": definition.to_dict()}


__all__ = ["BRANCH_RUNNERS", "MEMBER_RUNNERS", "execute_ensemble",
           "run_experiment_branch", "run_experiment_ensemble", "uncertainty_summary"]
