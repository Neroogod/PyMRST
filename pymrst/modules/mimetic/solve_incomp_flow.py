"""Deprecated MRST entry point ``solveIncompFlow.m``."""

from pymrst.modules.mimetic.incomp_mimetic import incompMimetic


def solve_incomp_flow(state, g, system, fluid=None, **options):
    """Call the directly ported :func:`incompMimetic` implementation."""
    return incompMimetic(state, g, system, fluid, **options)


solveIncompFlow = solve_incomp_flow

__all__ = ["solve_incomp_flow", "solveIncompFlow"]
