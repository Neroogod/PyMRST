"""MRST-style CO2Lab spillpoint trapping helpers."""

from pymrst.modules.co2lab_spillpoint.node_trapping import (
    NodeTrapAnalysis,
    node_trap_analysis,
    nodes_to_cells,
)
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
    "NodeTrapAnalysis",
    "TrapAnalysis",
    "TrapTree",
    "downstream_closure",
    "downstream_traps",
    "find_optimal_injection_point",
    "maximize_trapping",
    "node_trap_analysis",
    "nodes_to_cells",
    "trap_analysis",
    "volumes_of_traps",
]
