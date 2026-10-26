"""Controlled experiment submission and bounded ensemble uncertainty analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Protocol

import numpy as np

from .._serialization import frozen_json_mapping, json_value, require_fields
from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission, PermissionGrant
from ..policies.execution import (
    PolicyDecision, ResourceEstimate, authorize_resource_estimate,
)
from ..policies.resource_limits import ExecutionBudget


@dataclass(frozen=True, slots=True)
class ExperimentSubmissionRequest:
    request_id: str
    experiment_id: str
    definition_uri: str
    definition_hash: str
    realization_count: int
    random_seed: int
    budget: ExecutionBudget
    execution_profile: str = "local_process"

    def __post_init__(self) -> None:
        for name in ("request_id", "experiment_id", "execution_profile"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if "://" not in self.definition_uri:
            raise ValueError("experiment definition must use an application URI")
        digest = str(self.definition_hash).lower()
        if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
            raise ValueError("experiment definition hash must be SHA-256")
        object.__setattr__(self, "definition_hash", digest)
        count, seed = int(self.realization_count), int(self.random_seed)
        if count <= 0 or seed < 0:
            raise ValueError("realization count must be positive and seed nonnegative")
        if count > self.budget.max_realizations:
            raise ValueError("realization count exceeds the request budget")
        object.__setattr__(self, "realization_count", count)
        object.__setattr__(self, "random_seed", seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "experiment_id": self.experiment_id,
            "definition_uri": self.definition_uri,
            "definition_hash": self.definition_hash,
            "realization_count": self.realization_count,
            "random_seed": self.random_seed, "budget": self.budget.to_dict(),
            "execution_profile": self.execution_profile,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentSubmissionRequest":
        data = require_fields(value, required={
            "request_id", "experiment_id", "definition_uri", "definition_hash",
            "realization_count", "random_seed", "budget", "execution_profile",
        }, name="ExperimentSubmissionRequest")
        data["budget"] = ExecutionBudget.from_dict(data["budget"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ExperimentSubmissionResult:
    request_id: str
    experiment_id: str
    status: str
    store_uri: str

    def __post_init__(self) -> None:
        for name in ("request_id", "experiment_id", "status"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if self.status not in {"queued", "running", "completed", "failed", "cancelled"}:
            raise ValueError("invalid experiment submission status")
        if "://" not in self.store_uri:
            raise ValueError("experiment result store must use an application URI")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "experiment_id": self.experiment_id,
            "status": self.status, "store_uri": self.store_uri,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentSubmissionResult":
        return cls(**require_fields(value, required={
            "request_id", "experiment_id", "status", "store_uri",
        }, name="ExperimentSubmissionResult"))


class ExperimentExecutionAdapter(Protocol):
    def submit_experiment(
        self, request: ExperimentSubmissionRequest,
    ) -> ExperimentSubmissionResult: ...


class ExperimentRequestCatalog:
    def __init__(self) -> None:
        self._items: dict[str, tuple[ExperimentSubmissionRequest, ResourceEstimate]] = {}

    def register(self, request, estimate):
        if not isinstance(request, ExperimentSubmissionRequest):
            raise TypeError("experiment catalog requires ExperimentSubmissionRequest")
        if not isinstance(estimate, ResourceEstimate):
            raise TypeError("experiment catalog requires ResourceEstimate")
        if request.request_id in self._items:
            raise KeyError(f"experiment request already registered: {request.request_id}")
        if estimate.realizations != request.realization_count:
            raise ValueError("resource estimate and experiment realization counts differ")
        self._items[request.request_id] = (request, estimate)
        return request

    def get(self, request_id):
        try:
            return self._items[request_id]
        except KeyError as error:
            raise KeyError(f"unknown experiment request: {request_id}") from error


class SubmitExperimentTool:
    descriptor = ToolDescriptor(
        "submit_experiment", "Submit one registered bounded experiment",
        AgentPermission.RUN_ENSEMBLE_OR_REMOTE, starts_workload=True,
    )

    def __init__(self, catalog: ExperimentRequestCatalog, adapter: ExperimentExecutionAdapter):
        self.catalog, self.adapter = catalog, adapter

    def authorize(self, call: ToolCall, grant: PermissionGrant, *, approved: bool) -> PolicyDecision:
        try:
            request, estimate = self.catalog.get(str(call.arguments.get("request_id", "")))
        except KeyError as error:
            from ..policies.execution import DecisionCode
            return PolicyDecision(False, DecisionCode.DENY_TOOL, str(error))
        return authorize_resource_estimate(
            self.descriptor, grant, request.budget, estimate, approved=approved,
        )

    def invoke(self, call: ToolCall) -> ToolResult:
        request, _ = self.catalog.get(str(call.arguments["request_id"]))
        result = self.adapter.submit_experiment(request)
        if not isinstance(result, ExperimentSubmissionResult):
            raise TypeError("experiment adapter must return ExperimentSubmissionResult")
        if result.request_id != request.request_id or result.experiment_id != request.experiment_id:
            raise ValueError("experiment adapter changed request lineage")
        return ToolResult(call.call_id, True, {"experiment_result": result.to_dict()})


@dataclass(frozen=True, slots=True)
class EnsembleDataset:
    dataset_id: str
    member_ids: tuple[str, ...]
    statuses: tuple[str, ...]
    quantities: Mapping[str, np.ndarray]
    checksum: str


class EnsembleAnalysisCatalog:
    def __init__(self, *, maximum_values: int = 10_000_000) -> None:
        self.maximum_values = int(maximum_values)
        if self.maximum_values <= 0:
            raise ValueError("maximum ensemble values must be positive")
        self._items: dict[str, EnsembleDataset] = {}

    def register(self, dataset_id, member_ids, statuses, quantities):
        dataset_id = str(dataset_id).strip()
        members = tuple(str(item).strip() for item in member_ids)
        statuses = tuple(str(item) for item in statuses)
        if not dataset_id or dataset_id in self._items:
            raise KeyError(f"ensemble dataset is empty or registered: {dataset_id}")
        if not members or len(set(members)) != len(members) or len(statuses) != len(members):
            raise ValueError("ensemble members must be unique and status-aligned")
        if any(item not in {"completed", "failed", "cancelled"} for item in statuses):
            raise ValueError("ensemble member status is invalid")
        completed = np.asarray([item == "completed" for item in statuses])
        copied, digest, count = {}, hashlib.sha256(), 0
        digest.update(json.dumps({"members": members, "statuses": statuses}).encode())
        for name, value in sorted(quantities.items()):
            name = str(name).strip()
            array = np.array(value, dtype=float, copy=True)
            if not name or array.ndim < 1 or array.shape[0] != len(members):
                raise ValueError("ensemble quantities must have a member-leading axis")
            if np.any(~np.isfinite(array[completed])):
                raise ValueError("completed ensemble member values must be finite")
            array.setflags(write=False)
            copied[name] = array
            count += array.size
            digest.update(name.encode()); digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, np.int64).tobytes()); digest.update(array.tobytes())
        if not copied or count > self.maximum_values:
            raise ValueError("ensemble dataset is empty or exceeds the value limit")
        result = EnsembleDataset(dataset_id, members, statuses, copied, digest.hexdigest())
        self._items[dataset_id] = result
        return result

    def get(self, dataset_id):
        try:
            return self._items[dataset_id]
        except KeyError as error:
            raise KeyError(f"unknown ensemble dataset: {dataset_id}") from error


class AnalyzeEnsembleTool:
    descriptor = ToolDescriptor(
        "analyze_ensemble", "Compute bounded uncertainty summaries over stored members",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: EnsembleAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        dataset = self.catalog.get(str(args["dataset_id"]))
        quantity = str(args["quantity"])
        if quantity not in dataset.quantities:
            raise KeyError(f"unknown ensemble quantity: {quantity}")
        percentiles = tuple(float(item) for item in args.get("percentiles", (10, 50, 90)))
        if not percentiles or any(not 0 <= item <= 100 for item in percentiles):
            raise ValueError("ensemble percentiles must lie in [0, 100]")
        completed = np.asarray([item == "completed" for item in dataset.statuses])
        if not np.any(completed):
            raise ValueError("ensemble has no completed members")
        values = dataset.quantities[quantity][completed]
        failed = int((~completed).sum())
        failure_fraction = failed / len(completed)
        allowed_failure_fraction = float(args.get("maximum_failure_fraction", 0.0))
        if not 0 <= allowed_failure_fraction <= 1:
            raise ValueError("maximum failure fraction must lie in [0, 1]")
        summary = {
            "mean": np.mean(values, axis=0).tolist(),
            "standard_deviation": np.std(values, axis=0, ddof=0).tolist(),
            "minimum": np.min(values, axis=0).tolist(),
            "maximum": np.max(values, axis=0).tolist(),
            "percentiles": {
                f"p{item:g}": np.percentile(values, item, axis=0).tolist()
                for item in percentiles
            },
            "completed_members": int(completed.sum()), "failed_members": failed,
            "failure_fraction": failure_fraction,
        }
        findings = []
        if failure_fraction > allowed_failure_fraction:
            findings.append({
                "severity": "error", "category": "experiment.member_failures",
                "statement": (
                    f"Ensemble failure fraction {failure_fraction:.6g} exceeds "
                    f"the allowed {allowed_failure_fraction:.6g}."
                ),
                "affected_objects": tuple(
                    f"member:{item}" for item, ok in zip(dataset.member_ids, completed) if not ok
                ),
            })
        return ToolResult(call.call_id, True, {
            "analysis_kind": "analyze_ensemble", "quantity": quantity,
            "summary": summary, "findings": tuple(findings),
            "evidence": {
                "kind": "validation", "uri": f"ensemble://{dataset.dataset_id}/{quantity}",
                "label": f"Stored ensemble quantity {quantity}",
                "locator": {"members": len(dataset.member_ids)},
                "checksum": dataset.checksum,
            },
        })


__all__ = [
    "AnalyzeEnsembleTool", "EnsembleAnalysisCatalog", "EnsembleDataset",
    "ExperimentExecutionAdapter", "ExperimentRequestCatalog",
    "ExperimentSubmissionRequest", "ExperimentSubmissionResult",
    "SubmitExperimentTool",
]
