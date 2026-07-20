"""Backward-compatible import path for node-based spillpoint trapping."""

from pymrst.modules.co2lab_spillpoint.node_trapping import (
    NodeTrapAnalysis,
    node_trap_analysis,
    nodes_to_cells,
)

__all__ = [
    "NodeTrapAnalysis",
    "node_trap_analysis",
    "nodes_to_cells",
]
