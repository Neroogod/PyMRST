"""Bounded, lazy stored-result inspection tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.research_studio.experiments import MemberResultStore
from pymrst.research_studio.studio.results import ResultCatalog

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


class ResultAnalysisError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _store_fingerprint(path: str | Path) -> str:
    store = Path(path)
    digest = hashlib.sha256()
    for name in ("member.json", "arrays.npz"):
        source = store / name
        if not source.is_file():
            raise ResultAnalysisError(f"completed result file is missing: {source}")
        digest.update(name.encode())
        digest.update(_sha256_file(source).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RegisteredResultAnalysis:
    result_id: str
    store_path: str
    fingerprint: str
    file_signature: tuple[tuple[str, int, int], ...]


def _file_signature(path: str | Path):
    store = Path(path)
    values = []
    for name in ("member.json", "arrays.npz"):
        source = store / name
        try:
            stat = source.stat()
        except FileNotFoundError as error:
            raise ResultAnalysisError(
                f"completed result file is missing: {source}"
            ) from error
        values.append((name, stat.st_size, stat.st_mtime_ns))
    return tuple(values)


class ResultAnalysisCatalog:
    """Freeze completed result identity before deterministic interpretation."""

    def __init__(self, results: ResultCatalog, *, maximum_values=20_000_000):
        self.results = results
        self.maximum_values = int(maximum_values)
        if self.maximum_values <= 0:
            raise ValueError("maximum analysis values must be positive")
        self._registered: dict[str, RegisteredResultAnalysis] = {}

    def register(self, result_id: str) -> RegisteredResultAnalysis:
        result_id = str(result_id).strip()
        if not result_id or result_id in self._registered:
            raise KeyError(f"result analysis id is empty or already registered: {result_id}")
        entry = self.results.get(result_id)
        value = RegisteredResultAnalysis(
            result_id, entry.store_path, _store_fingerprint(entry.store_path),
            _file_signature(entry.store_path),
        )
        self._registered[result_id] = value
        return value

    def get(self, result_id: str):
        try:
            registered = self._registered[result_id]
        except KeyError as error:
            raise KeyError(f"unknown result analysis: {result_id}") from error
        signature = _file_signature(registered.store_path)
        if signature != registered.file_signature:
            if _store_fingerprint(registered.store_path) != registered.fingerprint:
                raise ResultAnalysisError(
                    f"completed result changed after analysis registration: {result_id}"
                )
            registered = RegisteredResultAnalysis(
                registered.result_id, registered.store_path,
                registered.fingerprint, signature,
            )
            self._registered[result_id] = registered
        lazy = MemberResultStore(registered.store_path).lazy()
        count = lazy.times.size
        for category in ("fields", "wells", "regions", "convergence", "conservation"):
            for name in lazy.names(category):
                shape = lazy.record["layout"][category][name]["shape"]
                count += int(np.prod(shape, dtype=np.int64))
        if count > self.maximum_values:
            raise ResultAnalysisError(
                f"result contains {count} values; analysis limit is {self.maximum_values}"
            )
        return registered, lazy

    def verify(self, result_id: str) -> bool:
        registered = self._registered.get(result_id)
        if registered is None:
            raise KeyError(f"unknown result analysis: {result_id}")
        if _store_fingerprint(registered.store_path) != registered.fingerprint:
            raise ResultAnalysisError(
                f"completed result content failed verification: {result_id}"
            )
        return True


def _evidence(registered: RegisteredResultAnalysis, subject: str):
    return {
        "kind": "run", "uri": f"result://{registered.result_id}/{subject}",
        "label": f"{subject.replace('_', ' ').title()} for {registered.result_id}",
        "locator": {"subject": subject}, "checksum": registered.fingerprint,
    }


def _finding(severity, category, statement, *, affected=(), details=None):
    return {
        "severity": severity, "category": category, "statement": statement,
        "affected_objects": list(affected), "details": details or {},
    }


def _array(lazy, category, *names):
    available = set(lazy.names(category))
    for name in names:
        if name in available:
            return name, np.asarray(lazy.array(category, name), dtype=float)
    return None, None


def _integrate(values, times, axis=0):
    """NumPy 1.26-compatible trapezoidal integration."""
    function = getattr(np, "trapezoid", None)
    if function is None:
        function = np.trapz
    return function(values, times, axis=axis)


class InspectResultTool:
    descriptor = ToolDescriptor(
        "inspect_result", "Inspect metadata and available quantities in a result store",
        AgentPermission.INSPECT,
    )

    def __init__(self, catalog: ResultCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        result_id = str(call.arguments["result_id"])
        value = self.catalog.metadata(result_id)
        value["evidence"] = {
            "kind": "run", "uri": f"result://{result_id}",
            "label": f"Registered result {result_id}",
            "locator": {}, "checksum": None,
        }
        return ToolResult(call.call_id, True, value)


class GetResultArrayTool:
    descriptor = ToolDescriptor(
        "get_result_array", "Read a bounded slice of a registered result quantity",
        AgentPermission.INSPECT,
    )

    def __init__(self, catalog: ResultCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        time_index = args.get("time_index")
        value = self.catalog.array(
            str(args["result_id"]), str(args["category"]), str(args["name"]),
            time_index=None if time_index is None else int(time_index),
            start=int(args.get("start", 0)),
            stop=None if args.get("stop") is None else int(args["stop"]),
        )
        value["evidence"] = {
            "kind": "state", "uri": (
                f"result://{args['result_id']}/{args['category']}/{args['name']}"
            ),
            "label": f"{args['category']}.{args['name']}",
            "locator": {
                "time_index": time_index,
                "start": value["start"], "stop": value["stop"],
            },
            "checksum": None,
        }
        return ToolResult(call.call_id, True, value)


class NumericalDiagnosticsTool:
    descriptor = ToolDescriptor(
        "analyze_numerical_diagnostics",
        "Analyze timesteps, nonlinear iterations, cuts, and convergence histories",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ResultAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        registered, lazy = self.catalog.get(str(args["result_id"]))
        times = np.asarray(lazy.times, dtype=float)
        intervals = np.diff(times)
        dt = (
            np.concatenate(([times[0]], intervals)) if times[0] > 0
            else intervals
        )
        findings = []
        bad_intervals = np.flatnonzero(intervals <= 0) + 1
        if np.any(~np.isfinite(times)) or times[0] < 0 or bad_intervals.size:
            findings.append(_finding(
                "critical", "numerical.time",
                "saved simulation times are non-finite or non-increasing",
                affected=(f"report_step:{int(index)}" for index in bad_intervals),
            ))

        iteration_name, iterations = _array(
            lazy, "convergence", "iterations", "newton_iterations",
            "nonlinear_iterations",
        )
        cut_name, cuts = _array(
            lazy, "convergence", "cuts", "timestep_cuts", "step_cuts",
        )
        warning_threshold = int(args.get("iteration_warning_threshold", 8))
        critical_threshold = int(args.get("iteration_critical_threshold", 15))
        if warning_threshold <= 0 or critical_threshold < warning_threshold:
            raise ValueError("iteration thresholds must be positive and ordered")

        iteration_summary = None
        if iterations is not None:
            flattened = np.asarray(iterations).reshape(times.size, -1)
            per_step = np.max(flattened, axis=1)
            if np.any(~np.isfinite(per_step)) or np.any(per_step < 0):
                bad = np.flatnonzero(~np.isfinite(per_step) | (per_step < 0))
                findings.append(_finding(
                    "critical", "numerical.iterations",
                    "nonlinear iteration history contains invalid values",
                    affected=(f"report_step:{int(index)}" for index in bad),
                ))
            critical = np.flatnonzero(per_step >= critical_threshold)
            warning = np.flatnonzero(
                (per_step >= warning_threshold) & (per_step < critical_threshold)
            )
            if critical.size:
                findings.append(_finding(
                    "error", "numerical.iterations",
                    f"{critical.size} report step(s) reached at least "
                    f"{critical_threshold} nonlinear iterations",
                    affected=(f"report_step:{int(index)}" for index in critical),
                    details={"maximum_iterations": float(np.nanmax(per_step))},
                ))
            if warning.size:
                findings.append(_finding(
                    "warning", "numerical.iterations",
                    f"{warning.size} report step(s) required at least "
                    f"{warning_threshold} nonlinear iterations",
                    affected=(f"report_step:{int(index)}" for index in warning),
                ))
            finite = per_step[np.isfinite(per_step)]
            iteration_summary = {
                "quantity": iteration_name,
                "total": float(np.sum(finite)),
                "mean": float(np.mean(finite)) if finite.size else None,
                "maximum": float(np.max(finite)) if finite.size else None,
                "p95": float(np.percentile(finite, 95)) if finite.size else None,
            }

        cut_summary = None
        if cuts is not None:
            flattened = np.asarray(cuts).reshape(times.size, -1)
            per_step = np.sum(flattened, axis=1)
            invalid = np.flatnonzero(~np.isfinite(per_step) | (per_step < 0))
            if invalid.size:
                findings.append(_finding(
                    "critical", "numerical.timestep_cuts",
                    "timestep-cut history contains invalid values",
                    affected=(f"report_step:{int(index)}" for index in invalid),
                ))
            cut_steps = np.flatnonzero(per_step > 0)
            if cut_steps.size:
                findings.append(_finding(
                    "warning", "numerical.timestep_cuts",
                    f"{cut_steps.size} report step(s) required timestep cuts",
                    affected=(f"report_step:{int(index)}" for index in cut_steps),
                    details={"total_cuts": float(np.nansum(per_step))},
                ))
            cut_summary = {
                "quantity": cut_name, "total": float(np.nansum(per_step)),
                "steps_with_cuts": int(cut_steps.size),
                "maximum_per_step": float(np.nanmax(per_step)) if per_step.size else 0.0,
            }

        convergence_fields = {}
        limits = {
            str(name): float(limit)
            for name, limit in args.get("convergence_limits", {}).items()
        }
        if any(limit < 0 or not np.isfinite(limit) for limit in limits.values()):
            raise ValueError("convergence limits must be finite and nonnegative")
        for name in lazy.names("convergence"):
            array = np.asarray(lazy.array("convergence", name), dtype=float)
            convergence_fields[name] = {
                "shape": list(array.shape),
                "finite": bool(np.all(np.isfinite(array))),
                "minimum": float(np.nanmin(array)) if array.size else None,
                "maximum": float(np.nanmax(array)) if array.size else None,
            }
            if np.any(~np.isfinite(array)):
                findings.append(_finding(
                    "critical", "numerical.nonfinite",
                    f"convergence quantity {name} contains non-finite values",
                    affected=(f"convergence:{name}",),
                ))
            if name in limits and array.size:
                maximum = float(np.max(np.abs(array)))
                if maximum > limits[name]:
                    findings.append(_finding(
                        "warning", "numerical.convergence_limit",
                        f"convergence quantity {name} reaches {maximum:.6g}, "
                        f"above the declared limit {limits[name]:.6g}",
                        affected=(f"convergence:{name}",),
                        details={"maximum_absolute": maximum, "limit": limits[name]},
                    ))

        value = {
            "analysis_kind": self.descriptor.name,
            "result_id": registered.result_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "report_steps": int(times.size), "simulated_time": float(times[-1]),
            "timestep": {
                "minimum": float(np.min(dt)) if dt.size else None,
                "maximum": float(np.max(dt)) if dt.size else None,
                "median": float(np.median(dt)) if dt.size else None,
            },
            "iterations": iteration_summary, "cuts": cut_summary,
            "convergence_fields": convergence_fields,
            "performance": dict(lazy.record.get("performance", {})),
            "findings": findings,
            "evidence": _evidence(registered, "numerical_diagnostics"),
        }
        return ToolResult(call.call_id, True, value)


class ReservoirPerformanceTool:
    descriptor = ToolDescriptor(
        "analyze_reservoir_performance",
        "Compute pressure, state, well, and field KPIs over saved result history",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ResultAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        registered, lazy = self.catalog.get(str(args["result_id"]))
        times = np.asarray(lazy.times, dtype=float)
        findings, fields, wells, regions = [], {}, {}, {}

        for name in lazy.names("fields"):
            array = np.asarray(lazy.array("fields", name), dtype=float)
            flattened = array.reshape(times.size, -1)
            if np.any(~np.isfinite(flattened)):
                findings.append(_finding(
                    "critical", "performance.nonfinite",
                    f"state field {name} contains non-finite values",
                    affected=(f"field:{name}",),
                ))
            fields[name] = {
                "shape": list(array.shape),
                "initial_mean": float(np.nanmean(flattened[0])),
                "final_mean": float(np.nanmean(flattened[-1])),
                "final_minimum": float(np.nanmin(flattened[-1])),
                "final_maximum": float(np.nanmax(flattened[-1])),
                "mean_change": float(
                    np.nanmean(flattened[-1]) - np.nanmean(flattened[0])
                ),
            }
        pressure = fields.get("pressure")
        if pressure is not None:
            pressure["mean_depletion"] = -pressure["mean_change"]

        metadata = lazy.record.get("metadata", {})
        sign = str(args.get(
            "rate_sign_convention", metadata.get("rate_sign_convention", "native")
        ))
        if sign not in {"native", "positive_into_reservoir"}:
            raise ValueError(
                "rate_sign_convention must be native or positive_into_reservoir"
            )
        rate_arrays = {}
        well_names = tuple(str(item) for item in metadata.get("well_names", ()))
        for name in lazy.names("wells"):
            array = np.asarray(lazy.array("wells", name), dtype=float)
            flattened = array.reshape(times.size, -1)
            if well_names and len(well_names) != flattened.shape[1]:
                raise ResultAnalysisError(
                    f"metadata well_names does not align with wells.{name}"
                )
            entity_names = well_names or tuple(
                f"well_{index}" for index in range(flattened.shape[1])
            )
            wells[name] = {
                "shape": list(array.shape), "entity_names": list(entity_names),
                "initial": flattened[0].tolist(), "final": flattened[-1].tolist(),
                "minimum": np.nanmin(flattened, axis=0).tolist(),
                "maximum": np.nanmax(flattened, axis=0).tolist(),
            }
            if np.any(~np.isfinite(flattened)):
                findings.append(_finding(
                    "critical", "performance.nonfinite",
                    f"well quantity {name} contains non-finite values",
                    affected=(f"well_quantity:{name}",),
                ))
            if name.lower().startswith("q") or "rate" in name.lower():
                rate_arrays[name] = flattened
                wells[name]["signed_integral_over_saved_window"] = _integrate(
                    flattened, times, axis=0).tolist()

        for name in lazy.names("regions"):
            array = np.asarray(lazy.array("regions", name), dtype=float)
            flattened = array.reshape(times.size, -1)
            regions[name] = {
                "shape": list(array.shape),
                "initial": flattened[0].tolist(), "final": flattened[-1].tolist(),
                "change": (flattened[-1] - flattened[0]).tolist(),
                "minimum": np.min(flattened, axis=0).tolist(),
                "maximum": np.max(flattened, axis=0).tolist(),
            }

        production = None
        if sign == "positive_into_reservoir" and rate_arrays:
            def phase(*names):
                for name in names:
                    if name in rate_arrays:
                        return rate_arrays[name]
                return None

            qo = phase("qOs", "oil_rate")
            qw = phase("qWs", "water_rate")
            qg = phase("qGs", "gas_rate")
            production = {
                "sign_convention": sign,
                "observation_window": [float(times[0]), float(times[-1])],
            }
            for label, rates in (("oil", qo), ("water", qw), ("gas", qg)):
                if rates is not None:
                    produced = np.maximum(-rates, 0.0)
                    injected = np.maximum(rates, 0.0)
                    production[f"{label}_produced"] = float(
                        np.sum(_integrate(produced, times, axis=0))
                    )
                    production[f"{label}_injected"] = float(
                        np.sum(_integrate(injected, times, axis=0))
                    )
            if qo is not None and qw is not None:
                oil = np.sum(np.maximum(-qo, 0.0), axis=1)
                water = np.sum(np.maximum(-qw, 0.0), axis=1)
                total = oil + water
                water_cut = np.divide(
                    water, total, out=np.zeros_like(total), where=total > 0,
                )
                production["final_field_water_cut"] = float(water_cut[-1])
                production["maximum_field_water_cut"] = float(np.max(water_cut))
            if qo is not None and qg is not None:
                oil = np.sum(np.maximum(-qo, 0.0), axis=1)
                gas = np.sum(np.maximum(-qg, 0.0), axis=1)
                gor = np.divide(gas, oil, out=np.full_like(gas, np.nan), where=oil > 0)
                production["final_field_gor"] = (
                    None if not np.isfinite(gor[-1]) else float(gor[-1])
                )

        value = {
            "analysis_kind": self.descriptor.name,
            "result_id": registered.result_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "report_steps": int(times.size),
            "observation_window": [float(times[0]), float(times[-1])],
            "fields": fields, "wells": wells, "production": production,
            "regions": regions, "findings": findings,
            "evidence": _evidence(registered, "reservoir_performance"),
        }
        return ToolResult(call.call_id, True, value)


_ECONOMIC_PHASES = ("oil", "water", "gas")


class EconomicsTool:
    descriptor = ToolDescriptor(
        "analyze_economics",
        "Compute discounted cash flow and net present value from stored "
        "production and injection rates",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ResultAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        registered, lazy = self.catalog.get(str(args["result_id"]))
        times = np.asarray(lazy.times, dtype=float)

        def _rates(mapping_name):
            values = {
                str(name): float(value)
                for name, value in args.get(mapping_name, {}).items()
            }
            if not set(values).issubset(_ECONOMIC_PHASES):
                raise ValueError(
                    f"{mapping_name} keys must be among {_ECONOMIC_PHASES}"
                )
            return values

        prices = _rates("prices")
        opex = _rates("opex")
        injection_cost = _rates("injection_cost")
        discount_rate = float(args.get("discount_rate", 0.0))
        capital_cost = float(args.get("capital_cost", 0.0))
        if discount_rate < 0 or not np.isfinite(discount_rate):
            raise ValueError("discount rate must be finite and nonnegative")
        if capital_cost < 0 or not np.isfinite(capital_cost):
            raise ValueError("capital cost must be finite and nonnegative")

        metadata = lazy.record.get("metadata", {})
        sign = str(args.get(
            "rate_sign_convention", metadata.get("rate_sign_convention", "native")
        ))
        if sign not in {"native", "positive_into_reservoir"}:
            raise ValueError(
                "rate_sign_convention must be native or positive_into_reservoir"
            )

        findings = []
        economics = None
        if sign == "positive_into_reservoir":
            qo_name, qo = _array(lazy, "wells", "qOs", "oil_rate")
            qw_name, qw = _array(lazy, "wells", "qWs", "water_rate")
            qg_name, qg = _array(lazy, "wells", "qGs", "gas_rate")
            phase_rates = {"oil": qo, "water": qw, "gas": qg}
            missing_price = []
            cash_flow_rate = np.zeros_like(times)
            revenue, produced_cost, injected_cost, produced_volume = {}, {}, {}, {}
            for phase, rates in phase_rates.items():
                if rates is None:
                    continue
                if np.any(~np.isfinite(rates)):
                    findings.append(_finding(
                        "critical", "economics.nonfinite",
                        f"well rate quantity for phase {phase} contains "
                        f"non-finite values; excluded from cash flow",
                        affected=(f"phase:{phase}",),
                    ))
                    rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
                produced = np.sum(np.maximum(-rates, 0.0), axis=1)
                injected = np.sum(np.maximum(rates, 0.0), axis=1)
                price = prices.get(phase)
                if price is None and np.any(produced > 0):
                    missing_price.append(phase)
                    price = 0.0
                price = price or 0.0
                revenue_rate = produced*price
                cost_rate = produced*opex.get(phase, 0.0) + injected*injection_cost.get(phase, 0.0)
                cash_flow_rate = cash_flow_rate + revenue_rate - cost_rate
                revenue[phase] = float(np.sum(_integrate(revenue_rate, times)))
                produced_cost[phase] = float(np.sum(_integrate(
                    produced*opex.get(phase, 0.0), times)))
                injected_cost[phase] = float(np.sum(_integrate(
                    injected*injection_cost.get(phase, 0.0), times)))
                produced_volume[phase] = float(np.sum(_integrate(produced, times)))
            if missing_price:
                findings.append(_finding(
                    "warning", "economics.missing_price",
                    f"production of {', '.join(sorted(missing_price))} has no "
                    f"assigned price; revenue excludes it",
                    affected=(f"phase:{phase}" for phase in missing_price),
                ))
            discounted_rate = cash_flow_rate*np.exp(-discount_rate*times)
            net_present_value = float(np.sum(_integrate(discounted_rate, times))) - capital_cost
            undiscounted_cash_flow = float(np.sum(_integrate(cash_flow_rate, times))) - capital_cost
            economics = {
                "revenue": revenue, "production_opex": produced_cost,
                "injection_cost": injected_cost, "produced_volume": produced_volume,
                "capital_cost": capital_cost, "discount_rate": discount_rate,
                "net_present_value": net_present_value,
                "undiscounted_net_cash_flow": undiscounted_cash_flow,
            }
            if net_present_value < 0:
                findings.append(_finding(
                    "warning", "economics.npv",
                    f"net present value over the saved observation window is "
                    f"negative ({net_present_value:.6g}) under the supplied "
                    f"price and cost assumptions",
                    details={"net_present_value": net_present_value},
                ))

        value = {
            "analysis_kind": self.descriptor.name,
            "result_id": registered.result_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "observation_window": [float(times[0]), float(times[-1])],
            "assumptions": {
                "prices": prices, "opex": opex, "injection_cost": injection_cost,
                "discount_rate": discount_rate, "capital_cost": capital_cost,
                "rate_sign_convention": sign,
            },
            "economics": economics, "findings": findings,
            "evidence": _evidence(registered, "economics"),
        }
        return ToolResult(call.call_id, True, value)


class HistoryMatchTool:
    descriptor = ToolDescriptor(
        "analyze_history_match",
        "Compare stored simulation results against supplied historical observations",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ResultAnalysisCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        args = call.arguments
        registered, lazy = self.catalog.get(str(args["result_id"]))
        times = np.asarray(lazy.times, dtype=float)
        observations = args.get("observations", {})
        if not observations:
            raise ValueError("history matching requires at least one observation")
        threshold = float(args.get("normalized_rmse_warning_threshold", 0.1))
        if threshold <= 0 or not np.isfinite(threshold):
            raise ValueError("normalized RMSE warning threshold must be positive")

        findings, summaries = [], {}
        for label, spec in observations.items():
            label = str(label)
            category = str(spec["category"])
            name = str(spec["name"])
            if category not in {"wells", "fields", "regions"}:
                raise ValueError("observation category must be wells, fields, or regions")
            if name not in set(lazy.names(category)):
                findings.append(_finding(
                    "critical", "history_matching.missing_quantity",
                    f"observation {label} requests {category}.{name}, which is "
                    f"not present in the stored result",
                    affected=(f"{category}:{name}",),
                ))
                continue
            array = np.asarray(lazy.array(category, name), dtype=float)
            flattened = array.reshape(times.size, -1)
            entity_index = int(spec.get("entity_index", 0))
            if not 0 <= entity_index < flattened.shape[1]:
                raise ValueError(f"observation {label} entity_index is out of range")
            simulated = flattened[:, entity_index]
            obs_times = np.asarray(spec["times"], dtype=float)
            obs_values = np.asarray(spec["values"], dtype=float)
            if (obs_times.ndim != 1 or obs_times.shape != obs_values.shape
                    or obs_times.size == 0):
                raise ValueError(f"observation {label} times and values must be "
                                  f"aligned, nonempty, one-dimensional")
            if np.any(~np.isfinite(obs_times)) or np.any(~np.isfinite(obs_values)):
                raise ValueError(f"observation {label} times and values must be finite")
            if np.any(np.diff(obs_times) <= 0):
                raise ValueError(f"observation {label} times must be strictly increasing")
            if obs_times[0] < times[0] or obs_times[-1] > times[-1]:
                raise ResultAnalysisError(
                    f"observation {label} times lie outside the saved simulated "
                    f"time window; extrapolation is not permitted"
                )
            weight = float(spec.get("weight", 1.0))
            if weight <= 0 or not np.isfinite(weight):
                raise ValueError(f"observation {label} weight must be positive")
            predicted = np.interp(obs_times, times, simulated)
            residual = predicted - obs_values
            rmse = float(np.sqrt(np.mean(residual**2)))
            scale = float(np.max(np.abs(obs_values)))
            normalized_rmse = rmse/scale if scale > 0 else rmse
            summaries[label] = {
                "category": category, "name": name, "entity_index": entity_index,
                "count": int(obs_times.size), "weight": weight, "rmse": rmse,
                "normalized_rmse": normalized_rmse,
                "maximum_absolute_residual": float(np.max(np.abs(residual))),
            }
            if normalized_rmse > threshold:
                findings.append(_finding(
                    "warning", "history_matching.misfit",
                    f"{label} ({category}.{name}) has normalized RMSE "
                    f"{normalized_rmse:.6g}, above the declared threshold "
                    f"{threshold:.6g}",
                    affected=(f"{category}:{name}",), details=summaries[label],
                ))

        aggregate_normalized_rmse = float(np.average(
            [item["normalized_rmse"] for item in summaries.values()],
            weights=[item["weight"] for item in summaries.values()],
        )) if summaries else None

        value = {
            "analysis_kind": self.descriptor.name,
            "result_id": registered.result_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "observation_window": [float(times[0]), float(times[-1])],
            "observations": summaries,
            "aggregate_normalized_rmse": aggregate_normalized_rmse,
            "findings": findings,
            "evidence": _evidence(registered, "history_matching"),
        }
        return ToolResult(call.call_id, True, value)


__all__ = [
    "EconomicsTool", "GetResultArrayTool", "HistoryMatchTool",
    "InspectResultTool", "NumericalDiagnosticsTool",
    "RegisteredResultAnalysis", "ReservoirPerformanceTool",
    "ResultAnalysisCatalog", "ResultAnalysisError",
]
