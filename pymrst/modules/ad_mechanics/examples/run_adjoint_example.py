"""Self-contained MRST ``runAdjointExample.m`` workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_mechanics.adjoint_impl import (
    AdjointResult,
    compute_gradient_adjoint,
    evaluate_objective,
    simulate_forward,
    uplift_objective,
)


@dataclass
class AdjointExampleResult:
    """Forward and sensitivity results from the complete example."""

    model: Any
    state0: State
    states: list[State]
    dts: np.ndarray
    controls: np.ndarray
    objective: float
    adjoint: AdjointResult
    node: int
    finite_difference_gradient: np.ndarray | None = None
    gradient_relative_error: float | None = None
    exponent_results: dict[float, AdjointResult] | None = None
    updated_controls: dict[float, np.ndarray] | None = None
    updated_objectives: dict[float, float] | None = None
    updated_states: dict[float, list[State]] | None = None
    figures: list[Any] | None = None

    # Preserve the earlier dictionary-like result surface.
    def __getitem__(self, key: str):
        return getattr(self, key)


def _default_case(nx: int = 4, ny: int = 4, nz: int = 2):
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import (
        MechBlackOilModel,
    )
    from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    physical = [40.0, 40.0, 10.0]
    grid = create_augmented_grid(
        compute_geometry(cart_grid([nx, ny, nz], physical))
    )
    rock = make_rock(grid, 500 * MILLI * DARCY, 0.3)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        rho=[1000.0, 800.0], n=[2.0, 2.0],
        c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=100 * BARSA,
    )

    points, tolerance = grid.nodes.coords, 1e-9
    xnodes = np.flatnonzero(
        (points[:, 0] < tolerance) | (points[:, 0] > physical[0] - tolerance)
    )
    ynodes = np.flatnonzero(
        (points[:, 1] < tolerance) | (points[:, 1] > physical[1] - tolerance)
    )
    base = np.flatnonzero(points[:, 2] > physical[2] - tolerance)
    nodes = np.concatenate([xnodes, ynodes, base])
    masks = np.vstack([
        np.tile([True, False, False], (xnodes.size, 1)),
        np.tile([False, True, False], (ynodes.size, 1)),
        np.tile([False, False, True], (base.size, 1)),
    ])
    elastic_bc = ElasticBC(
        disp_nodes=nodes, disp_mask=masks,
        disp_vals=np.zeros((nodes.size, 3)),
    )
    stiffness = enu2c(
        np.full(grid.cells.num, 1.0e7), np.full(grid.cells.num, 0.3)
    )
    mechanics = MechanicModel(
        grid, stiffness, alpha=1.0, el_bc=elastic_bc
    )

    injection_rate = 2e-3
    wells = add_well(
        None, grid, rock, np.array([0]), well_type="rate", val=2e-3,
        compi=np.array([1.0, 0.0]), sign=1, name="Injector",
    )
    wells = add_well(
        wells, grid, rock, np.array([nx - 1]), well_type="rate",
        val=injection_rate, compi=np.array([1.0, 0.0]), sign=1,
        name="Injector-2",
    )
    wells = add_well(
        wells, grid, rock, np.array([grid.cells.num - 1]),
        well_type="bhp", val=90 * BARSA,
        compi=np.array([0.0, 1.0]), sign=-1, name="Producer",
    )
    model = MechBlackOilModel(grid, rock, fluid, wells, mechanics)
    state = State(
        pressure=np.full(grid.cells.num, 100.0 * BARSA),
        sW=np.full(grid.cells.num, 0.2),
        qWs=np.array([injection_rate, injection_rate, 0.0]),
        qOs=np.array([0.0, 0.0, -2.0 * injection_rate]),
        bhp=np.array([100.0 * BARSA, 100.0 * BARSA, 90.0 * BARSA]),
    )
    state = model.seed_state(state)
    surface = np.flatnonzero(
        (points[:, 2] < tolerance)
        & (points[:, 0] > tolerance)
        & (points[:, 0] < physical[0] - tolerance)
        & (points[:, 1] > tolerance)
        & (points[:, 1] < physical[1] - tolerance)
    )
    if surface.size == 0:
        raise ValueError("default adjoint grid has no free interior surface node")
    return model, state, int(surface[0])


def _mrst_2d_case(nx: int = 31, ny: int = 30):
    """MRST's 30-by-10 metre plane-strain adjoint geometry."""
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_mechanics.models.fully_coupled.mech_oil_water_model import (
        MechOilWaterModel,
    )
    from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c_2d
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, POISE

    physical = (30.0, 10.0)
    grid = create_augmented_grid(
        compute_geometry(cart_grid([nx, ny], physical))
    )
    rock = make_rock(grid, DARCY, 0.3)
    p_ref = 100.0 * BARSA
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 100.0]) * CENTI * POISE,
        rho=[1000.0, 700.0], n=[1.0, 1.0],
        c=[1e-10, 1e-10], p_ref=p_ref,
    )

    points = grid.nodes.coords[:, :2]
    tolerance = 1e-9
    constrained = np.flatnonzero(
        (points[:, 0] < tolerance)
        | (points[:, 0] > physical[0] - tolerance)
        | (points[:, 1] < tolerance)
    )
    boundary = np.any(grid.faces.neighbors == -1, axis=1)
    top_faces = np.flatnonzero(
        boundary & (grid.faces.centroids[:, 1] > physical[1] - tolerance)
    )
    sign = (
        (grid.faces.neighbors[top_faces, 0] == -1).astype(float)
        - (grid.faces.neighbors[top_faces, 1] == -1).astype(float)
    )
    normals = grid.faces.normals[top_faces, :2] * (
        sign / grid.faces.areas[top_faces]
    )[:, None]
    elastic_bc = ElasticBC(
        disp_nodes=constrained,
        disp_mask=np.ones((constrained.size, 2), dtype=bool),
        disp_vals=np.zeros((constrained.size, 2)),
        force_faces=top_faces,
        force=normals * p_ref,
    )
    stiffness = enu2c_2d(
        np.full(grid.cells.num, 1.0e7), np.full(grid.cells.num, 0.3)
    )
    mechanics = MechanicModel(grid, stiffness, alpha=1.0, el_bc=elastic_bc)

    layer = int(np.floor(ny / 4.0))
    injector_1 = int(round(nx / 4.0)) - 1 + layer * nx
    injector_2 = nx - int(round(nx / 4.0)) + layer * nx
    producer = int(round(nx / 2.0)) - 1 + layer * nx
    rate = 1.0 / 86400.0
    wells = add_well(
        None, grid, rock, np.array([injector_1]), well_type="rate",
        val=rate, compi=np.array([1.0, 0.0]), sign=1, name="Injector-1",
    )
    wells = add_well(
        wells, grid, rock, np.array([injector_2]), well_type="rate",
        val=rate, compi=np.array([1.0, 0.0]), sign=1, name="Injector-2",
    )
    wells = add_well(
        wells, grid, rock, np.array([producer]), well_type="bhp",
        val=p_ref, compi=np.array([0.0, 1.0]), sign=-1, name="Producer",
    )
    model = MechOilWaterModel(grid, rock, fluid, wells, mechanics)
    state = State(
        pressure=np.full(grid.cells.num, p_ref),
        sW=np.zeros(grid.cells.num), sO=np.ones(grid.cells.num),
        qWs=np.array([rate, rate, 0.0]), qOs=np.array([0.0, 0.0, -2*rate]),
        bhp=np.full(3, p_ref),
    )
    state = model.seed_state(state)

    top_x = physical[0] / 2.0
    candidates = np.flatnonzero(
        (points[:, 1] > physical[1] - tolerance)
        & (np.abs(points[:, 0] - top_x) <= physical[0] / nx)
    )
    free = candidates[~mechanics.operators.is_dir.reshape(-1, 2)[candidates, 1]]
    if free.size == 0:
        raise ValueError("MRST adjoint grid has no free central top node")
    return model, state, int(free[0])


def _central_gradient(model, state0, dts, controls, objective,
                      relative_step: float) -> np.ndarray:
    gradient = np.zeros_like(controls, dtype=float)
    for step in range(controls.shape[0]):
        for well in range(controls.shape[1]):
            h = relative_step * max(abs(float(controls[step, well])), 1.0)
            upper, lower = controls.copy(), controls.copy()
            upper[step, well] += h
            lower[step, well] -= h
            ju = evaluate_objective(
                objective, simulate_forward(model, state0, dts, upper), dts
            )
            jl = evaluate_objective(
                objective, simulate_forward(model, state0, dts, lower), dts
            )
            gradient[step, well] = (ju - jl) / (2.0 * h)
    return gradient


def mrst_adjoint_schedule(
    *, valmax: float | None = None, valmin: float | None = None,
    producer_bhp: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return MRST's one-day + thirty ten-day adjoint schedule."""
    from pymrst.utils.units import BARSA, DAY

    valmax = (1.0 / DAY) if valmax is None else float(valmax)
    valmin = (0.1 / DAY) if valmin is None else float(valmin)
    producer_bhp = 100.0 * BARSA if producer_bhp is None else float(producer_bhp)
    dts = np.concatenate([[1.0 * DAY], np.full(30, 10.0 * DAY)])
    elapsed = np.cumsum(dts)
    flatten_time = 150.0 * DAY
    rates = np.where(
        elapsed < flatten_time,
        valmin * elapsed / flatten_time
        + valmax * (flatten_time - elapsed) / flatten_time,
        valmin,
    )
    controls = np.column_stack([rates, rates, np.full(rates.size, producer_bhp)])
    return dts, controls


def powered_uplift_objective(
    model, node: int, *, exponent: float = 1.0,
    normalization_constant: float = 1.0, direction: int = 2,
):
    """MRST ``objUpliftAver`` objective and state derivative."""
    if exponent <= 0.0 or normalization_constant <= 0.0:
        raise ValueError("exponent and normalization_constant must be positive")
    signed = uplift_objective(model, node, direction=direction)
    eps = np.finfo(float).eps

    def evaluate(state: State, dt: float):
        signed_value, signed_derivative = signed(state, dt)
        uplift = -signed_value / dt  # depth-positive coordinates
        positive = max(uplift, 0.0)
        scaled = eps + normalization_constant * positive
        value = dt * scaled**exponent
        derivative = np.zeros_like(signed_derivative)
        if uplift > 0.0:
            derivative = (
                -exponent * normalization_constant
                * scaled**(exponent - 1.0) * signed_derivative
            )
        return float(value), derivative

    return evaluate


def _updated_injection_schedule(model, controls, dts, gradient,
                                reduction_fraction: float) -> np.ndarray:
    """Apply MRST's equal-total-volume sensitivity update."""
    updated = controls.copy()
    injectors = [i for i, well in enumerate(model.wells)
                 if well.type == "rate" and well.sign > 0]
    if not injectors or reduction_fraction <= 0.0:
        return updated
    for index in injectors:
        initial_volume = float(np.dot(np.maximum(controls[:, index], 0.0), dts))
        target_reduction = reduction_fraction * initial_volume
        sensitivity = np.asarray(gradient[:, index], dtype=float)
        # A common sign cancels in this normalization, exactly as in MRST.
        denominator = float(np.dot(sensitivity, dts))
        if abs(denominator) <= np.finfo(float).eps:
            continue
        decrement = target_reduction * sensitivity / denominator
        updated[:, index] = np.maximum(0.0, controls[:, index] - decrement)
    return updated


def run_adjoint_example(
    model=None,
    state0=None,
    dts: Sequence[float] | None = None,
    controls=None,
    node: int | None = None,
    direction: int | None = None,
    *,
    check_finite_difference: bool | None = None,
    finite_difference_step: float = 1e-6,
    grid_shape: tuple[int, int, int] = (4, 4, 2),
    exponents: Sequence[float] | None = None,
    normalization_constant: float = 1.0e3,
    update_reduction_fraction: float = 0.03,
    perform_schedule_updates: bool = True,
    use_mrst_schedule: bool = False,
    use_mrst_geometry: bool = False,
    make_plots: bool = False,
    show_plots: bool = False,
) -> AdjointExampleResult:
    """Run forward, uplift objective, adjoint and optional FD verification.

    With no model arguments this constructs a deterministic two-phase
    poromechanical reservoir. Explicit arguments retain the former API.
    """
    from pymrst.utils.units import BARSA, DAY

    supplied = (model, state0, node)
    self_contained = all(item is None for item in supplied)
    if self_contained:
        if use_mrst_geometry:
            model, state0, node = _mrst_2d_case()
        else:
            model, state0, node = _default_case(*grid_shape)
    elif any(item is None for item in supplied):
        raise TypeError("model, state0 and node must be supplied together")

    schedule_controls = None
    if dts is None and use_mrst_schedule:
        dts, schedule_controls = mrst_adjoint_schedule(
            producer_bhp=90.0 * BARSA
        )
    elif dts is None:
        dts = np.array([10.0 * DAY, 10.0 * DAY])
    dts = np.asarray(dts, dtype=float).ravel()
    if controls is None:
        controls = schedule_controls
        if controls is None:
            targets = [well.val for well in model.wells]
            controls = np.tile(targets, (dts.size, 1))
    controls = np.atleast_2d(np.asarray(controls, dtype=float))
    n_wells = getattr(getattr(model, "facility", None), "n_wells", controls.shape[1])
    if controls.shape != (dts.size, n_wells):
        raise ValueError("controls must have one row per timestep and well")
    if check_finite_difference is None:
        check_finite_difference = self_contained
    if exponents is None:
        exponents = (1.0, 10.0) if self_contained else ()

    objective_function = uplift_objective(model, int(node), direction=direction)
    states = simulate_forward(model, state0, dts, controls)
    value = evaluate_objective(objective_function, states, dts)
    adjoint = compute_gradient_adjoint(
        model, state0, dts, controls, objective_function, states=states
    )

    fd_gradient = None
    relative_error = None
    if check_finite_difference:
        fd_gradient = _central_gradient(
            model, state0, dts, controls, objective_function,
            finite_difference_step,
        )
        scale = max(float(np.max(np.abs(fd_gradient))), 1e-30)
        relative_error = float(
            np.max(np.abs(adjoint.gradient - fd_gradient)) / scale
        )

    exponent_results: dict[float, AdjointResult] = {}
    updated_controls: dict[float, np.ndarray] = {}
    updated_objectives: dict[float, float] = {}
    updated_state_sets: dict[float, list[State]] = {}
    for exponent in exponents:
        exponent = float(exponent)
        powered = powered_uplift_objective(
            model, int(node), exponent=exponent,
            normalization_constant=normalization_constant,
            direction=direction,
        )
        sensitivity = compute_gradient_adjoint(
            model, state0, dts, controls, powered, states=states
        )
        exponent_results[exponent] = sensitivity
        if perform_schedule_updates:
            updated = _updated_injection_schedule(
                model, controls, dts, sensitivity.gradient,
                update_reduction_fraction,
            )
            updated_states = simulate_forward(model, state0, dts, updated)
            updated_controls[exponent] = updated
            updated_state_sets[exponent] = list(updated_states)
            updated_objectives[exponent] = evaluate_objective(
                powered, updated_states, dts
            )

    result = AdjointExampleResult(
        model=model, state0=state0, states=list(states), dts=dts,
        controls=controls, objective=value, adjoint=adjoint, node=int(node),
        finite_difference_gradient=fd_gradient,
        gradient_relative_error=relative_error,
        exponent_results=exponent_results,
        updated_controls=updated_controls,
        updated_objectives=updated_objectives,
        updated_states=updated_state_sets,
    )
    if not self_contained and not check_finite_difference:
        # Exact backward compatibility for the former explicit-input helper.
        return {"states": states, "objective": value, "adjoint": adjoint}
    if make_plots:
        result.figures = [plot_adjoint_example(result, show=show_plots)[0]]
    return result


def _uplift_history(result: AdjointExampleResult,
                    states: Sequence[State]) -> np.ndarray:
    dimension = int(result.model.G.griddim)
    direction = dimension - 1
    values = []
    for state in states[1:]:
        full = state.get("u")
        if full is None:
            full = result.model.mech.full_displacement(state["xd"])
        values.append(-np.asarray(full).reshape(-1, dimension)[result.node, direction])
    return np.asarray(values, dtype=float)


def plot_adjoint_example(result: AdjointExampleResult, *, show: bool = False):
    """Render the MRST adjoint schedule/uplift/sensitivity dashboard."""
    import matplotlib.pyplot as plt

    time_days = np.cumsum(result.dts) / 86400.0
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    # Reservoir map and well placement.
    ax = axes[0, 0]
    centroids = result.model.G.cells.centroids
    pressure = np.asarray(result.states[-1]["pressure"], dtype=float) / 1.0e5
    image = ax.scatter(centroids[:, 0], centroids[:, 1], c=pressure,
                       s=18, cmap="viridis")
    for well in result.model.wells:
        cells = np.asarray(well.cells, dtype=int).ravel()
        xy = centroids[cells, :2].mean(axis=0)
        marker = "^" if well.sign > 0 else "v"
        ax.scatter(*xy, marker=marker, s=90, edgecolor="black",
                   facecolor="white", zorder=3)
        ax.annotate(well.name or well.type, xy, xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    figure.colorbar(image, ax=ax, label="Pressure [bar]")
    ax.set(title="Final pressure and wells", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")

    # Original and gradient-updated schedules.
    ax = axes[0, 1]
    injectors = [i for i, well in enumerate(result.model.wells)
                 if well.type == "rate" and well.sign > 0]
    for index in injectors:
        ax.step(time_days, result.controls[:, index] * 86400.0, where="post",
                linewidth=2, label=f"Original {result.model.wells[index].name}")
    for exponent, controls in (result.updated_controls or {}).items():
        if injectors:
            ax.step(time_days, controls[:, injectors[0]] * 86400.0,
                    where="post", linestyle="--", label=f"Updated p={exponent:g}")
    ax.set(title="Injection schedules", xlabel="Time [day]",
           ylabel="Rate [m³/day]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Uplift before and after schedule update.
    ax = axes[1, 0]
    ax.plot(time_days, _uplift_history(result, result.states), linewidth=2,
            label="Original")
    for exponent, states in (result.updated_states or {}).items():
        ax.plot(time_days, _uplift_history(result, states), linestyle="--",
                label=f"Updated p={exponent:g}")
    ax.set(title="Surface uplift", xlabel="Time [day]", ylabel="Uplift [m]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Normalized sensitivities, matching MRST's comparison plot.
    ax = axes[1, 1]
    for exponent, sensitivity in (result.exponent_results or {}).items():
        if not injectors:
            continue
        gradient = np.asarray(sensitivity.gradient[:, injectors[0]], dtype=float)
        scale = max(float(np.max(np.abs(gradient))), np.finfo(float).eps)
        ax.plot(time_days, gradient / scale, marker="o", markersize=3,
                label=f"p={exponent:g}")
    ax.set(title="Normalized uplift sensitivity", xlabel="Time [day]",
           ylabel="Normalized dJ/dq")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    if show:
        plt.show()
    return figure, axes


__all__ = [
    "AdjointExampleResult",
    "run_adjoint_example",
    "compute_gradient_adjoint",
    "evaluate_objective",
    "simulate_forward",
    "uplift_objective",
    "powered_uplift_objective",
    "mrst_adjoint_schedule",
    "plot_adjoint_example",
]


if __name__ == "__main__":
    result = run_adjoint_example()
    print(f"objective: {result.objective:.6e}")
    print("adjoint gradient:\n", result.adjoint.gradient)
    print("finite-difference gradient:\n", result.finite_difference_gradient)
    print(f"maximum relative error: {result.gradient_relative_error:.3e}")
    for exponent, sensitivity in result.exponent_results.items():
        print(f"p={exponent:g}: objective={sensitivity.value:.6e}, "
              f"updated={result.updated_objectives[exponent]:.6e}")
