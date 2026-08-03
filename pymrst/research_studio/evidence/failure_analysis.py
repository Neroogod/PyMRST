"""Deterministic classification of residual and control histories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Any, Iterable, Mapping

import numpy as np


class FailureKind(StrEnum):
    LINEAR_FAILURE = "linear_failure"
    NONFINITE = "nonfinite"
    DIVERGENCE = "divergence"
    STALL = "stall"
    RESIDUAL_FLOOR = "residual_floor"
    OSCILLATION = "oscillation"
    CONTROL_CHATTERING = "control_chattering"
    MAX_ITERATIONS = "max_iterations"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailureAnalysis:
    classifications: tuple[FailureKind, ...]
    worst_residual: str | None
    worst_iteration: int | None
    worst_value: float | None
    initial_norm: float | None
    final_norm: float | None
    median_convergence_ratio: float | None
    estimated_log10_rate: float | None
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifications": [kind.value for kind in self.classifications],
            "worst_residual": self.worst_residual,
            "worst_iteration": self.worst_iteration,
            "worst_value": self.worst_value,
            "initial_norm": self.initial_norm,
            "final_norm": self.final_norm,
            "median_convergence_ratio": self.median_convergence_ratio,
            "estimated_log10_rate": self.estimated_log10_rate,
            "details": dict(self.details),
        }


def _number(value):
    if isinstance(value, Mapping) and "nonfinite" in value:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def analyze_residual_history(
    residual_history: Iterable[Mapping[str, Any]],
    *,
    converged: bool = False,
    control_history: Iterable[Any] = (),
    stall_window: int = 4,
    stall_relative_improvement: float = 0.05,
    divergence_factor: float = 10.0,
    floor_relative_span: float = 0.02,
) -> FailureAnalysis:
    history = [dict(item) for item in residual_history]
    kinds: list[FailureKind] = []
    norms, worst_name, worst_iteration, worst_value = [], None, None, -1.0
    has_nonfinite = has_linear_failure = False
    for iteration, residuals in enumerate(history):
        finite_values = []
        for name, raw in residuals.items():
            number = _number(raw)
            if name == "linear_failure":
                has_linear_failure = True
            if not math.isfinite(number):
                has_nonfinite = True
                continue
            magnitude = abs(number)
            finite_values.append(magnitude)
            if magnitude > worst_value:
                worst_name, worst_iteration, worst_value = str(name), iteration, magnitude
        norms.append(max(finite_values, default=float("nan")))
    finite_norms = np.asarray([value for value in norms if np.isfinite(value)], dtype=float)
    if has_linear_failure:
        kinds.append(FailureKind.LINEAR_FAILURE)
    if has_nonfinite:
        kinds.append(FailureKind.NONFINITE)

    ratios = np.array([], dtype=float)
    if finite_norms.size >= 2:
        previous = finite_norms[:-1]
        ratios = finite_norms[1:]/np.maximum(previous, np.finfo(float).tiny)
        if finite_norms[-1] > divergence_factor*max(finite_norms[0], np.finfo(float).tiny):
            kinds.append(FailureKind.DIVERGENCE)
        changes = np.diff(finite_norms)
        alternating = np.count_nonzero(changes[1:]*changes[:-1] < 0)
        if changes.size >= 4 and alternating >= changes.size-2:
            kinds.append(FailureKind.OSCILLATION)
        if finite_norms.size >= stall_window:
            tail = finite_norms[-stall_window:]
            improvement = (tail[0]-tail[-1])/max(tail[0], np.finfo(float).tiny)
            if improvement < stall_relative_improvement and tail[-1] > 0:
                kinds.append(FailureKind.STALL)
            span = (tail.max()-tail.min())/max(tail.max(), np.finfo(float).tiny)
            if span <= floor_relative_span and tail[-1] > 0:
                kinds.append(FailureKind.RESIDUAL_FLOOR)

    controls = list(control_history)
    switches = sum(first != second for first, second in zip(controls[:-1], controls[1:]))
    reversals = sum(
        controls[index] == controls[index-2] != controls[index-1]
        for index in range(2, len(controls))
    )
    if len(controls) >= 4 and switches >= len(controls)-2 and reversals >= 2:
        kinds.append(FailureKind.CONTROL_CHATTERING)
    if not converged and history and not kinds:
        kinds.append(FailureKind.MAX_ITERATIONS)
    if not converged and not history and not kinds:
        kinds.append(FailureKind.UNKNOWN)

    log_rate = None
    if finite_norms.size >= 2 and np.all(finite_norms > 0):
        log_rate = float(np.polyfit(
            np.arange(finite_norms.size), np.log10(finite_norms), 1,
        )[0])
    return FailureAnalysis(
        classifications=tuple(dict.fromkeys(kinds)),
        worst_residual=worst_name,
        worst_iteration=worst_iteration,
        worst_value=(worst_value if worst_value >= 0 else None),
        initial_norm=(float(finite_norms[0]) if finite_norms.size else None),
        final_norm=(float(finite_norms[-1]) if finite_norms.size else None),
        median_convergence_ratio=(float(np.median(ratios)) if ratios.size else None),
        estimated_log10_rate=log_rate,
        details={
            "iterations_observed": len(history),
            "control_switches": switches,
            "control_reversals": reversals,
        },
    )


def analyze_simulation_evidence(evidence, *, converged: bool) -> FailureAnalysis:
    residuals = []
    for step in evidence.timestep_reports:
        residuals.extend(step.get("residuals", ()))
    controls = [
        step.get("control_index") for step in evidence.timestep_reports
        if step.get("control_index") is not None
    ]
    return analyze_residual_history(
        residuals, converged=converged, control_history=controls,
    )


__all__ = [
    "FailureAnalysis", "FailureKind", "analyze_residual_history",
    "analyze_simulation_evidence",
]
