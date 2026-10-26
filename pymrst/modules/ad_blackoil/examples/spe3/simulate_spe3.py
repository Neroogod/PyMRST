"""MRST ``ad-blackoil/examples/spe3/simulateSPE3.m``: a minimal example that runs SPE3.

This is a BLACKOIL version of the SPE3 case.

The deck becomes a packed problem (`initEclipsePackedProblemAD`), which is
simulated from its first step with every report step written to disk
(`simulatePackedProblem`), then plotted (`plotPackedProblem`).
"""

from __future__ import annotations

from pymrst.modules.ad_blackoil.examples.utils.local_dataset_path import local_dataset_path
from pymrst.modules.ad_core.simulators.sim_runner.init_eclipse_packed_problem_ad import (
    init_eclipse_packed_problem_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.plot_packed_problem import (
    plot_packed_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem,
)


def simulate_spe3(plot: bool = True):
    """MRST `simulateSPE3`. Returns the packed problem."""
    pth = local_dataset_path("spe3", "BENCH_SPE3.DATA")
    fn = pth/"BENCH_SPE3.DATA"
    problem = init_eclipse_packed_problem_ad(fn)
    # Simulate
    simulate_packed_problem(problem, restartStep=1)
    # Plot
    if plot:
        plot_packed_problem(problem)
    return problem


simulateSPE3 = simulate_spe3

__all__ = ["simulateSPE3", "simulate_spe3"]
