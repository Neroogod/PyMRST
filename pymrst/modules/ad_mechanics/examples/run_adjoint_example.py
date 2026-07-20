"""MRST ``runAdjointExample.m`` compatibility example."""

from pymrst.modules.ad_mechanics.adjoint_impl import (
    compute_gradient_adjoint,
    evaluate_objective,
    simulate_forward,
    uplift_objective,
)


def run_adjoint_example(model, state0, dts, controls, node: int,
                        direction: int = 2):
    """Run the PyMRST surface-uplift adjoint workflow.

    Parameters are explicit rather than hidden in a plotting script: provide a
    coupled geomechanics model, initial state, timesteps, well controls and
    objective node, and this returns forward states plus the adjoint result.
    """
    objective = uplift_objective(model, node, direction=direction)
    states = simulate_forward(model, state0, dts, controls)
    value = evaluate_objective(objective, states, dts)
    adjoint = compute_gradient_adjoint(model, state0, dts, controls,
                                       objective, states=states)
    return {"states": states, "objective": value, "adjoint": adjoint}


__all__ = [
    "run_adjoint_example",
    "compute_gradient_adjoint",
    "evaluate_objective",
    "simulate_forward",
    "uplift_objective",
]
