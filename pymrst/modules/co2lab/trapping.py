"""Backward-compatible import path for spillpoint trapping helpers."""

from pymrst.modules.co2lab_spillpoint.trapping import (
    TrapAnalysis,
    TrapTree,
    downstream_closure,
    downstream_traps,
    find_optimal_injection_point,
    maximize_trapping,
    trap_analysis,
    volumes_of_traps,
)

__all__ = [
    "TrapAnalysis",
    "TrapTree",
    "downstream_closure",
    "downstream_traps",
    "find_optimal_injection_point",
    "maximize_trapping",
    "trap_analysis",
    "volumes_of_traps",
]
