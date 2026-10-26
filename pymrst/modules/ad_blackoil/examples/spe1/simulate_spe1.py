"""MRST ``ad-blackoil/examples/spe1/simulateSPE1.m``: a minimal example that runs SPE1.

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


def simulate_spe1(plot: bool = True):
    """MRST `simulateSPE1`. Returns the packed problem."""
    pth = local_dataset_path("spe1", "BENCH_SPE1.DATA")
    fn = pth/"BENCH_SPE1.DATA"
    problem = init_eclipse_packed_problem_ad(fn)
    # Simulate
    simulate_packed_problem(problem, restartStep=1)
    # Plot
    if plot:
        plot_packed_problem(problem)
    return problem


simulateSPE1 = simulate_spe1

__all__ = ["simulateSPE1", "simulate_spe1"]
