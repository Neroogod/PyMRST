"""Port of MRST `book-ii/compositionalMiscibilityExample.m`.

The MRST example is intentionally expensive: seven producer pressures, 1000
cells, and 200 timesteps. This port keeps the same case generator and pressure
sweep, while the default run is scaled down for tests. Pass `full_size=True` to
request the MRST dimensions.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.examples._shared import (
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    run_schedule,
    simple_schedule,
    simulate_packed_problem,
)
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI


MRST_MISCIBILITY_PRESSURES_BAR = np.array([70.0, 90.0, 100.0, 110.0, 120.0, 130.0, 150.0])


def miscibility_mixture() -> Mixture:
    """Four-component CO2/C1/C5/C10 mixture from MRST."""
    mix = Mixture(
        names=("CO2", "C1", "C5", "C10"),
        Tcrit=np.array([304.7, 190.60, 419.5, 626.0]),
        Pcrit=np.array([73.87, 43.04, 37.47, 24.20]) * BARSA,
        Vcrit=np.array([0.094, 0.098, 0.258, 0.534]) / 1000.0,
        acentric=np.array([0.225, 0.013, 0.1956, 0.385]),
        molarMass=np.array([44.01, 16.04, 58.12, 134.0]) / 1000.0,
        name="MRST miscibility CO2/C1/C5/C10 mixture",
    )
    bic = np.zeros((4, 4))
    bic[0, 1] = 0.1
    bic[3, 1] = 0.041
    mix.set_binary_interaction(bic + bic.T)
    return mix


def setup_miscibility_test(
    bhp: float,
    ncell: int = 40,
    nstep: int = 8,
    use_natural: bool = False,
):
    """Build MRST's `setupMiscibilityTest` case."""
    mixture = miscibility_mixture()
    grid = compute_geometry(cart_grid([ncell, 1, 1], [200.0, 10.0, 1.0]))
    rock = make_rock(grid, 100.0 * MILLI * DARCY, 0.3)

    z0 = np.array([0.33, 0.03, 0.24, 0.40])
    z_inj = np.array([1.0, 0.0, 0.0, 0.0])
    z_inj = np.maximum(z_inj, 1.0e-8)
    z_inj = z_inj / z_inj.sum()
    temperature = 373.0
    p0 = 150.0 * BARSA

    wells = add_well(
        None,
        grid,
        rock,
        np.array([0]),
        radius=0.15,
        well_type="rate",
        val=50.0 / DAY,
        compi=np.array([1.0, 0.0]),
        sign=1,
        name="Injector",
    )
    wells[0].components = z_inj
    wells = add_well(
        wells,
        grid,
        rock,
        np.array([grid.cells.num - 1]),
        radius=0.15,
        well_type="bhp",
        val=float(bhp),
        compi=np.array([1.0, 0.0]),
        sign=-1,
        name="Producer",
    )
    wells[1].components = z_inj

    model_cls = NaturalVariablesCompositionalModel if use_natural else OverallCompositionModel
    model = model_cls(
        grid,
        rock,
        mixture,
        temperature=temperature,
        wells=wells,
        water=False,
        rho_s=(25.0, 25.0),
        n_kr=(2.0, 2.0),
        max_iterations=60,
    )
    state0 = model.initial_state(p0, z0)
    schedule = simple_schedule(np.full(int(nstep), 2000.0 * DAY / float(nstep)), wells=wells)
    return state0, model, schedule


def run_compositional_miscibility_example(
    pressures_bar=(70.0,),
    *,
    ncell: int = 6,
    nstep: int = 10,
    full_size: bool = False,
    compare_formulations: bool = False,
):
    """Run MRST's producer-pressure sweep for the miscible displacement case."""
    if full_size:
        ncell, nstep = 1000, 200
        if np.asarray(pressures_bar, dtype=float).size == 1 and float(np.asarray(pressures_bar, dtype=float)[0]) == 70.0:
            pressures_bar = MRST_MISCIBILITY_PRESSURES_BAR
    cases = []
    problems = []
    for pressure_bar in np.asarray(pressures_bar, dtype=float):
        state0, model, schedule = setup_miscibility_test(
            pressure_bar * BARSA,
            ncell=ncell,
            nstep=nstep,
            use_natural=False,
        )
        problems.append(pack_simulation_problem(
            state0,
            model,
            schedule,
            "MiscibilityDemo",
            name=f"{int(pressure_bar) if pressure_bar.is_integer() else pressure_bar:g}",
        ))
    simulate_packed_problem(problems)
    _, states, reports, names = get_multiple_packed_simulator_outputs(problems)

    for problem, state_history, report, name in zip(problems, states, reports, names):
        result = problem.result
        assert result is not None
        final = result.final_state
        pressure_bar = float(problem.name)
        cases.append(
            {
                "name": name,
                "states": state_history,
                "reports": report,
                "producer_bhp": float(pressure_bar * BARSA),
                "producer_bhp_bar": float(pressure_bar),
                "gas_saturation": final_sg(problem.model, final),
                "co2": final.z[:, 0].copy(),
                "co2_left": float(final.z[0, 0]),
                "co2_middle": float(final.z[final.z.shape[0] // 2, 0]),
                "iterations": result.iterations,
            }
        )

    comparison = None
    if compare_formulations:
        bhp = float(np.asarray(pressures_bar, dtype=float).ravel()[0]) * BARSA
        coarse_nstep = max(10, int(np.ceil(nstep / 5)))
        state_nat, model_nat, schedule_nat = setup_miscibility_test(
            bhp,
            ncell=ncell,
            nstep=coarse_nstep,
            use_natural=True,
        )
        state_over, model_over, schedule_over = setup_miscibility_test(
            bhp,
            ncell=ncell,
            nstep=coarse_nstep,
            use_natural=False,
        )
        result_nat = run_schedule(model_nat, state_nat, schedule_nat)
        result_over = run_schedule(model_over, state_over, schedule_over)
        comparison = {
            "natural_iterations": result_nat.iterations,
            "overall_iterations": result_over.iterations,
            "natural_final": result_nat.final_state,
            "overall_final": result_over.final_state,
        }
        comparison["iteration_table"] = miscibility_formulation_iterations(comparison)
    out = {
        "pressures_bar": np.asarray(pressures_bar, dtype=float),
        "packed_problems": problems,
        "names": names,
        "states": states,
        "reports": reports,
        "cases": cases,
        "comparison": comparison,
        "front_animation": miscibility_front_animation(problems, states),
    }
    out["plot_sections"] = miscibility_plot_sections(out)
    return out


def miscibility_front_animation(problems, states, *, max_steps: int | None = None):
    """Return MRST's gas-saturation front animation data.

    The MATLAB example plots ``states{i}{step}.s(:, 2)`` against the cell
    centroid x-coordinate for every pressure case and every report step.
    """
    items = list(problems)
    histories = list(states)
    nstep = min(len(history) for history in histories) if histories else 0
    if max_steps is not None:
        nstep = min(nstep, int(max_steps))
    frames = []
    x = np.asarray(items[0].model.g.cells.centroids[:, 0], dtype=float).copy() if items else np.array([])
    for ix in range(nstep):
        profiles = {}
        for problem, history in zip(items, histories):
            profiles[problem.name] = final_sg(problem.model, history[ix])
        frames.append({"step": int(ix + 1), "profiles": profiles})
    return {
        "x": x,
        "names": tuple(problem.name for problem in items),
        "steps": np.arange(1, nstep + 1, dtype=int),
        "frames": frames,
    }


def miscibility_formulation_iterations(comparison):
    """Return MRST's natural/overall coarse-timestep iteration bar data."""
    natural = np.asarray(comparison["natural_iterations"], dtype=int).ravel()
    overall = np.asarray(comparison["overall_iterations"], dtype=int).ravel()
    n = min(natural.size, overall.size)
    table = np.column_stack([natural[:n], overall[:n]]) if n else np.empty((0, 2), dtype=int)
    return {
        "names": ("Natural variables", "Overall composition"),
        "iterations": table,
        "cumulative": np.cumsum(table, axis=0),
    }


def miscibility_plot_sections(output: dict) -> dict:
    """Return MRST's animation and iteration-plot metadata sections."""
    front = output["front_animation"]
    names = tuple(output["names"])
    comparison = output.get("comparison")
    sections = {
        "front_animation": {
            "figure": 1,
            "x": front["x"],
            "steps": front["steps"],
            "frames": front["frames"],
            "names": front["names"],
            "colormap": "summer",
            "color_count": len(names),
            "line_width": 2.0,
            "ylabel": "S_g",
            "xlabel": "x",
        }
    }
    if comparison is not None:
        table = comparison["iteration_table"]
        sections["formulation_iterations"] = {
            "iterations": table["iterations"],
            "cumulative": table["cumulative"],
            "legend": table["names"],
            "bar_width": 2.0,
        }
    else:
        sections["formulation_iterations"] = {
            "iterations": None,
            "cumulative": None,
            "legend": ("Natural variables", "Overall composition"),
            "bar_width": 2.0,
            "reason": "comparison branch not run",
        }
    return sections


def final_sg(model: OverallCompositionModel, state) -> np.ndarray:
    """Return gas saturation for a completed state."""
    _, _, _, s_g, _, _ = model._flow_props0(state)
    return np.asarray(s_g, dtype=float)


if __name__ == "__main__":
    def test_miscibility_setup_matches_mrst_constants():
        st, model, schedule = setup_miscibility_test(90.0 * BARSA, ncell=8, nstep=2)
        assert model.mix.names == ("CO2", "C1", "C5", "C10")
        assert model.g.cells.num == 8
        assert len(model.wells) == 2
        assert np.isclose(model.wells[0].val, 50.0 / DAY)
        expected = np.array([1.0, 1.0e-8, 1.0e-8, 1.0e-8])
        expected = expected / expected.sum()
        assert np.allclose(model.wells[0].components, expected)
        assert schedule.step.val.size == 2
        assert np.allclose(st.z[0], [0.33, 0.03, 0.24, 0.40])
        print("ok test_miscibility_setup_matches_mrst_constants")

    def test_miscibility_sweep_returns_mrst_pressure_cases():
        out = run_compositional_miscibility_example(
            pressures_bar=[70.0],
            ncell=6,
            nstep=10,
            compare_formulations=False,
        )
        assert len(out["cases"]) == 1
        assert np.allclose(out["pressures_bar"], [70.0])
        assert all(case["co2_left"] > 0.0 for case in out["cases"])
        assert all(np.all(np.isfinite(case["gas_saturation"])) for case in out["cases"])
        assert out["front_animation"]["frames"]
        assert out["front_animation"]["frames"][0]["profiles"]["70"].shape == (6,)
        assert out["plot_sections"]["front_animation"]["figure"] == 1
        assert out["plot_sections"]["front_animation"]["colormap"] == "summer"
        assert out["plot_sections"]["formulation_iterations"]["reason"] == "comparison branch not run"
        print("ok test_miscibility_sweep_returns_mrst_pressure_cases")

    def test_miscibility_comparison_returns_iteration_table():
        table = miscibility_formulation_iterations({
            "natural_iterations": np.array([4, 5, 4]),
            "overall_iterations": np.array([3, 3, 4]),
        })
        assert table["names"] == ("Natural variables", "Overall composition")
        assert table["iterations"].shape[1] == 2
        assert np.all(table["cumulative"] >= table["iterations"])
        print("ok test_miscibility_comparison_returns_iteration_table")

    def test_miscibility_full_size_expands_pressure_list_without_running():
        assert np.allclose(MRST_MISCIBILITY_PRESSURES_BAR, [70, 90, 100, 110, 120, 130, 150])
        st, model, schedule = setup_miscibility_test(
            150.0 * BARSA,
            ncell=1000,
            nstep=200,
            use_natural=True,
        )
        assert model.g.cells.num == 1000
        assert schedule.step.val.size == 200
        assert np.isclose(st.pressure[0], 150.0 * BARSA)
        print("ok test_miscibility_full_size_expands_pressure_list_without_running")

    test_miscibility_setup_matches_mrst_constants()
    test_miscibility_sweep_returns_mrst_pressure_cases()
    test_miscibility_comparison_returns_iteration_table()
    test_miscibility_full_size_expands_pressure_list_without_running()
    print("\nAll tests passed!")


__all__ = [
    "MRST_MISCIBILITY_PRESSURES_BAR",
    "final_sg",
    "miscibility_formulation_iterations",
    "miscibility_front_animation",
    "miscibility_mixture",
    "miscibility_plot_sections",
    "run_compositional_miscibility_example",
    "setup_miscibility_test",
]
