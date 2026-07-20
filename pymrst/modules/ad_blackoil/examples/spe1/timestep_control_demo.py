"""Port of MRST ``spe1/timestepControlDemo.m``."""

from __future__ import annotations

from pymrst.modules.ad_blackoil.examples.spe1.simulate_spe1 import build_spe1
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, collect_history
from pymrst.modules.ad_core import IterationCountTimeStepSelector, NonLinearSolver, SimulationReport


def run_timestep_control_demo(total_steps: int = 10) -> ExampleResult:
    deck, model, state = build_spe1()
    selector = IterationCountTimeStepSelector(target_iteration_count=5)
    solver = NonLinearSolver(max_iterations=30, timestep_selector=selector)
    report = SimulationReport()
    states = []
    st = state.copy()
    for dt in deck["tsteps"][:total_steps]:
        st, step = solver.solve_timestep(model, st, float(dt))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(f"SPE1 adaptive step failed: {step}")
        states.append(st.copy())
    hist = collect_history(states, ("pressure", "sW", "sG", "rs", "bhp"))
    return ExampleResult(states, report, hist)


if __name__ == "__main__":
    out = run_timestep_control_demo(5)
    print(f"adaptive SPE1: {sum(len(s.ministeps) for s in out.report.steps)} ministeps")
