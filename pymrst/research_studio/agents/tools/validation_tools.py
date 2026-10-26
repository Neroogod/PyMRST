"""Deterministic normalized comparison and qualification tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from pymrst.research_studio.validation import (
    ComparisonSeries,
    TolerancePolicy,
    compare_series,
)
from pymrst.research_studio.evidence.conservation import conservation_report
from pymrst.research_studio.evidence.invariants import (
    InvariantSuite, bounded_field, finite_field,
)

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


class ComparisonLimitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidationDataset:
    dataset_id: str
    fields: Mapping[str, np.ndarray]
    checksum: str


class ValidationDataCatalog:
    """In-memory numerical datasets addressed by JSON-safe identifiers."""

    def __init__(self, *, maximum_values=5_000_000):
        self.maximum_values = int(maximum_values)
        if self.maximum_values <= 0:
            raise ValueError("maximum validation values must be positive")
        self._datasets = {}

    def register(self, dataset_id: str, fields: Mapping[str, Any]):
        dataset_id = str(dataset_id).strip()
        if not dataset_id or dataset_id in self._datasets:
            raise KeyError(f"dataset id is empty or already registered: {dataset_id}")
        copied, digest = {}, hashlib.sha256()
        for name, value in sorted(fields.items()):
            name = str(name).strip()
            if not name:
                raise ValueError("validation field names cannot be empty")
            array = np.array(value, dtype=float, copy=True)
            if array.ndim == 0:
                raise ValueError("validation fields must be arrays")
            array.setflags(write=False)
            copied[name] = array
            digest.update(name.encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        if not copied or sum(value.size for value in copied.values()) > self.maximum_values:
            raise ComparisonLimitError("validation dataset is empty or exceeds value limit")
        dataset = ValidationDataset(dataset_id, copied, digest.hexdigest())
        self._datasets[dataset_id] = dataset
        return dataset

    def get(self, dataset_id: str) -> ValidationDataset:
        try:
            return self._datasets[dataset_id]
        except KeyError as error:
            raise KeyError(f"unknown validation dataset: {dataset_id}") from error


def _series(value: Mapping[str, Any], name: str, maximum_values: int):
    values = np.asarray(value["values"], dtype=float)
    if values.size > maximum_values:
        raise ComparisonLimitError(
            f"{name} contains {values.size} values; limit is {maximum_values}"
        )
    return ComparisonSeries(
        values=values,
        times=np.asarray(value["times"], dtype=float),
        entity_names=tuple(value.get("entity_names", ())),
        unit=str(value.get("unit", "dimensionless")),
        sign_convention=str(value.get("sign_convention", "native")),
    )


class CompareSeriesTool:
    descriptor = ToolDescriptor(
        "compare_series", "Normalize and compare two time-by-entity histories",
        AgentPermission.ANALYZE,
    )

    def __init__(self, *, maximum_values=1_000_000):
        self.maximum_values = int(maximum_values)
        if self.maximum_values <= 0:
            raise ValueError("maximum comparison values must be positive")

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        reference = _series(args["reference"], "reference", self.maximum_values)
        candidate = _series(args["candidate"], "candidate", self.maximum_values)
        tolerance_value = args.get("tolerance", {})
        tolerance = TolerancePolicy(
            absolute=float(tolerance_value.get("absolute", 0.0)),
            relative=float(tolerance_value.get("relative", 0.0)),
            scale=float(tolerance_value.get("scale", 1.0)),
        )
        sign_scale = args.get("sign_scale")
        result = compare_series(
            reference, candidate, tolerance=tolerance,
            unit_scale=float(args.get("unit_scale", 1.0)),
            sign_scale=None if sign_scale is None else float(sign_scale),
        )
        return ToolResult(call.call_id, True, result.to_dict())


class _InventoryAdapter:
    def __init__(self, names, units):
        self.component_names = tuple(names)
        self.units = str(units)
        self.sign_convention = "positive_into_reservoir"

    def inventory(self, state):
        return state["inventory"]

    def well_rates(self, state):
        return state["well_rates"]


class ConservationTool:
    descriptor = ToolDescriptor(
        "analyze_conservation", "Compute a component conservation balance",
        AgentPermission.ANALYZE,
    )

    def __init__(self, *, maximum_values=5_000_000):
        self.maximum_values = int(maximum_values)
        if self.maximum_values <= 0:
            raise ValueError("maximum conservation values must be positive")

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        names = tuple(str(item) for item in args["component_names"])
        if not names or len(set(names)) != len(names):
            raise ValueError("component names must be nonempty and unique")
        initial = np.asarray(args["initial_inventory"], dtype=float)
        final = np.asarray(args["final_inventory"], dtype=float)
        if initial.size + final.size > self.maximum_values:
            raise ComparisonLimitError("conservation inventory exceeds value limit")
        if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(final)):
            raise ValueError("conservation inventories must be finite")
        rates = np.asarray(args.get("well_rates", np.zeros(len(names))), dtype=float)
        adapter = _InventoryAdapter(names, args.get("units", "dimensionless"))
        report = conservation_report(
            adapter,
            {"inventory": initial, "well_rates": np.zeros(len(names))},
            {"inventory": final, "well_rates": rates},
            float(args["dt"]),
            boundary_rates=args.get("boundary_rates"),
            source_rates=args.get("source_rates"),
            regions=args.get("regions"),
        )
        tolerance = float(args.get("normalized_tolerance", 1e-8))
        if tolerance < 0:
            raise ValueError("normalized conservation tolerance must be nonnegative")
        value = report.to_dict()
        value.update({
            "analysis_kind": self.descriptor.name,
            "passed": report.maximum_normalized_imbalance <= tolerance,
            "tolerance": tolerance,
            "findings": [] if report.maximum_normalized_imbalance <= tolerance else [{
                "severity": "error", "category": "conservation",
                "statement": (
                    "maximum normalized component imbalance "
                    f"{report.maximum_normalized_imbalance:.6g} exceeds {tolerance:.6g}"
                ),
                "affected_objects": [
                    f"component:{names[index]}" for index in np.flatnonzero(
                        np.abs(report.normalized_imbalance) > tolerance
                    )
                ],
                "details": {
                    "normalized_imbalance": report.normalized_imbalance.tolist(),
                },
            }],
            "evidence": {
                "kind": "conservation", "uri": f"calculation://{call.call_id}/conservation",
                "label": "Deterministic component conservation calculation",
                "locator": {"component_names": list(names)}, "checksum": None,
            },
        })
        return ToolResult(call.call_id, True, value)


class InvariantTool:
    descriptor = ToolDescriptor(
        "analyze_invariants", "Evaluate finite and bounded-field invariants",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ValidationDataCatalog | None = None, *, maximum_values=5_000_000):
        self.catalog = catalog
        self.maximum_values = int(maximum_values)
        if self.maximum_values <= 0:
            raise ValueError("maximum invariant values must be positive")

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        dataset = None
        if "dataset_id" in args:
            if self.catalog is None:
                raise ValueError("dataset_id requires a ValidationDataCatalog")
            dataset = self.catalog.get(str(args["dataset_id"]))
            fields = dataset.fields
        else:
            fields = {
                str(name): np.asarray(value, dtype=float)
                for name, value in args["fields"].items()
            }
        if sum(value.size for value in fields.values()) > self.maximum_values:
            raise ComparisonLimitError("invariant fields exceed value limit")
        providers = []
        for spec in args["checks"]:
            kind, name = str(spec["kind"]), str(spec["field"])
            if name not in fields:
                raise KeyError(f"invariant field is missing: {name}")
            if kind == "finite":
                providers.append(finite_field(name))
            elif kind == "bounded":
                providers.append(bounded_field(
                    name, float(spec["lower"]), float(spec["upper"]),
                    float(spec.get("tolerance", 1e-10)),
                ))
            else:
                raise ValueError(f"unknown invariant kind: {kind}")
        results = InvariantSuite(providers).evaluate(fields, "observe")
        failed = tuple(item for item in results if not item.passed)
        value = {
            "analysis_kind": self.descriptor.name, "passed": not failed,
            "results": [item.to_dict() for item in results],
            "findings": [{
                "severity": "error", "category": "invariant",
                "statement": item.message,
                "affected_objects": [f"index:{index}" for index in item.locations],
                "details": item.to_dict(),
            } for item in failed],
            "evidence": {
                "kind": "validation",
                "uri": (
                    f"dataset://{dataset.dataset_id}" if dataset is not None
                    else f"calculation://{call.call_id}/invariants"
                ),
                "label": "Deterministic physical-invariant calculation",
                "locator": {"checks": len(results)}, "checksum": None,
            },
        }
        if dataset is not None:
            value["evidence"]["checksum"] = dataset.checksum
        return ToolResult(call.call_id, True, value)


@dataclass(frozen=True, slots=True)
class ReferenceQuantity:
    values: np.ndarray
    times: np.ndarray
    entity_names: tuple[str, ...]
    unit: str
    sign_convention: str


@dataclass(frozen=True, slots=True)
class ReferenceRun:
    run_id: str
    simulator: str
    quantities: Mapping[str, ReferenceQuantity]
    checksum: str


class ReferenceRunCatalog:
    """Normalized, immutable histories from PyMRST, MRST, OPM, or other references."""

    def __init__(self, *, maximum_values=10_000_000):
        self.maximum_values = int(maximum_values)
        if self.maximum_values <= 0:
            raise ValueError("maximum reference values must be positive")
        self._runs = {}

    def register(self, run_id: str, simulator: str, quantities: Mapping[str, Any]):
        run_id, simulator = str(run_id).strip(), str(simulator).strip().lower()
        if not run_id or not simulator or run_id in self._runs:
            raise KeyError(f"reference run identity is invalid or registered: {run_id}")
        copied, digest, count = {}, hashlib.sha256(), 0
        digest.update(run_id.encode()); digest.update(simulator.encode())
        for name, spec in sorted(quantities.items()):
            name = str(name).strip()
            values = np.array(spec["values"], dtype=float, copy=True)
            times = np.array(spec["times"], dtype=float, copy=True).reshape(-1)
            entities = tuple(str(item) for item in spec.get("entity_names", ()))
            if values.ndim == 1:
                values = values[:, None]
            if (
                not name or values.ndim != 2 or values.shape[0] != times.size
                or times.size == 0 or np.any(~np.isfinite(values))
                or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0)
            ):
                raise ValueError(f"invalid normalized reference quantity: {name}")
            if entities and (len(entities) != values.shape[1] or len(set(entities)) != len(entities)):
                raise ValueError("reference entity names must be unique and column-aligned")
            unit = str(spec.get("unit", "dimensionless"))
            sign = str(spec.get("sign_convention", "native"))
            values.setflags(write=False); times.setflags(write=False)
            copied[name] = ReferenceQuantity(values, times, entities, unit, sign)
            count += values.size
            digest.update(name.encode()); digest.update(values.dtype.str.encode())
            digest.update(np.asarray(values.shape, np.int64).tobytes()); digest.update(values.tobytes())
            digest.update(times.tobytes()); digest.update("|".join(entities).encode())
            digest.update(unit.encode()); digest.update(sign.encode())
        if not copied or count > self.maximum_values:
            raise ComparisonLimitError("reference run is empty or exceeds value limit")
        result = ReferenceRun(run_id, simulator, copied, digest.hexdigest())
        self._runs[run_id] = result
        return result

    def get(self, run_id: str):
        try:
            return self._runs[run_id]
        except KeyError as error:
            raise KeyError(f"unknown reference run: {run_id}") from error


class CompareReferenceRunsTool:
    descriptor = ToolDescriptor(
        "compare_reference_runs",
        "Compare normalized PyMRST, MRST, OPM, or analytical histories",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ReferenceRunCatalog):
        self.catalog = catalog

    @staticmethod
    def _comparison_series(quantity: ReferenceQuantity):
        return ComparisonSeries(
            quantity.values, quantity.times, quantity.entity_names,
            quantity.unit, quantity.sign_convention,
        )

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        reference = self.catalog.get(str(args["reference_run_id"]))
        candidate = self.catalog.get(str(args["candidate_run_id"]))
        policies = args["quantities"]
        if not policies:
            raise ValueError("reference comparison requires quantity policies")
        comparisons, findings = {}, []
        for name, spec in policies.items():
            name = str(name)
            if name not in reference.quantities or name not in candidate.quantities:
                raise KeyError(f"comparison quantity is missing: {name}")
            tolerance = TolerancePolicy(
                absolute=float(spec.get("absolute", 0.0)),
                relative=float(spec.get("relative", 0.0)),
                scale=float(spec.get("scale", 1.0)),
            )
            sign_scale = spec.get("sign_scale")
            result = compare_series(
                self._comparison_series(reference.quantities[name]),
                self._comparison_series(candidate.quantities[name]),
                tolerance=tolerance,
                unit_scale=float(spec.get("unit_scale", 1.0)),
                sign_scale=None if sign_scale is None else float(sign_scale),
            )
            comparisons[name] = result.to_dict()
            if not result.passed:
                findings.append({
                    "severity": "error", "category": "validation.simulator_disagreement",
                    "statement": (
                        f"{candidate.simulator} and {reference.simulator} disagree for "
                        f"{name}: maximum absolute error {result.maximum_absolute_error:.6g}, "
                        f"maximum relative error {result.maximum_relative_error:.6g}."
                    ),
                    "affected_objects": (f"quantity:{name}", f"entity:{result.worst_entity}"),
                })
        return ToolResult(call.call_id, True, {
            "analysis_kind": "compare_reference_runs",
            "reference": {"run_id": reference.run_id, "simulator": reference.simulator},
            "candidate": {"run_id": candidate.run_id, "simulator": candidate.simulator},
            "passed": not findings, "comparisons": comparisons,
            "findings": tuple(findings),
            "evidence": ({
                "kind": "validation", "uri": f"run://{reference.run_id}/normalized",
                "label": f"Normalized {reference.simulator} reference run",
                "locator": {"simulator": reference.simulator},
                "checksum": reference.checksum,
            }, {
                "kind": "validation", "uri": f"run://{candidate.run_id}/normalized",
                "label": f"Normalized {candidate.simulator} candidate run",
                "locator": {"simulator": candidate.simulator},
                "checksum": candidate.checksum,
            }),
        })


__all__ = [
    "CompareReferenceRunsTool", "CompareSeriesTool", "ComparisonLimitError",
    "ConservationTool", "InvariantTool", "ReferenceQuantity", "ReferenceRun",
    "ReferenceRunCatalog", "ValidationDataCatalog", "ValidationDataset",
]
