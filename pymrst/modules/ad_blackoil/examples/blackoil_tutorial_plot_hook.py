"""Port of MRST ``blackoilTutorialPlotHook.m`` using Python callbacks."""

from __future__ import annotations

from typing import Callable

import numpy as np

from pymrst.modules.ad_blackoil.examples.simulator_workflow_example import (
    build_simulator_workflow_example,
)
from pymrst.modules.ad_core import NonLinearSolver


def run_blackoil_tutorial_plot_hook(callback: Callable[[int, object], None] | None = None):
    model, state, timesteps = build_simulator_workflow_example()
    solver = NonLinearSolver()
    states = []
    st = state.copy()
    for step, dt in enumerate(np.asarray(timesteps, dtype=float)):
        st, report = solver.solve_timestep(model, st, float(dt))
        if not report.converged:
            raise RuntimeError(f"step {step} failed: {report}")
        states.append(st.copy())
        if callback is not None:
            callback(step, st)
    return states


if __name__ == "__main__":
    states = run_blackoil_tutorial_plot_hook(
        lambda i, st: print(f"step {i}: max sW={np.max(st['sW']):.3f}")
    )
    print(f"callback example produced {len(states)} states")
