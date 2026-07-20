"""Port of MRST `compositionalValidationSimple.m`.

MRST's example builds the simple 1D injector/producer case with
``setupSimpleCompositionalExample(useNatural)``, packs it as a simulation
problem, runs the schedule, and compares natural-variable and
overall-composition formulations against external simulator references.

PyMRST does not ship the commercial E300/AD-GPRS reference states, but this
port keeps the MRST setup shape: the same ``use_natural`` switch, an explicit
schedule, state history, and nonlinear iteration report.
"""

import numpy as np

from pymrst.modules.compositional.examples._shared import (
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    run_schedule,
    simple_schedule,
    simulate_packed_problem,
)
from pymrst.modules.compositional.examples.utils import setup_simple_compositional_example
from pymrst.utils.units import DAY


def setup_compositional_validation_simple_problem(
    *,
    use_natural: bool = True,
    full_schedule: bool = False,
    nstep: int = 3,
    dt: float = 20.0 * DAY,
):
    """Build MRST's packed simple-compositional validation problem."""
    setup = setup_simple_compositional_example(use_natural=use_natural)
    if full_schedule:
        timesteps = np.full(500, 200.0 * DAY / 500.0)
    else:
        timesteps = np.full(int(nstep), float(dt))
    schedule = simple_schedule(timesteps, wells=setup.wells)
    name = "Natural" if use_natural else "Overall"
    problem = pack_simulation_problem(
        setup.state,
        setup.model,
        schedule,
        "simple_comp",
        name=name,
    )
    return {
        "setup": setup,
        "schedule": schedule,
        "problem": problem,
        "name": name,
        "external_references": {
            "E300": None,
            "AD-GPRS": None,
        },
    }


def run_compositional_validation_simple(
    nstep: int = 3,
    dt: float = 20.0 * DAY,
    *,
    use_natural: bool = True,
    full_schedule: bool = False,
):
    """Run MRST's simple compositional validation case.

    ``full_schedule=True`` mirrors the long MRST setup more closely: 500 report
    steps over 200 days. The default remains a short smoke test for routine
    validation.
    """
    case = setup_compositional_validation_simple_problem(
        use_natural=use_natural,
        full_schedule=full_schedule,
        nstep=nstep,
        dt=dt,
    )
    setup = case["setup"]
    schedule = case["schedule"]
    result = run_schedule(setup.model, setup.state, schedule)
    return {
        "name": case["name"],
        "setup": setup,
        "schedule": schedule,
        "problem": case["problem"],
        "external_references": case["external_references"],
        "result": result,
        "state": result.final_state,
        "pressure": np.asarray([s.pressure for s in result.states]),
        "composition": np.asarray([s.z for s in result.states]),
        "saturation": np.asarray([s.sT for s in result.states]),
        "qOs": np.asarray([s.qOs for s in result.states]),
        "qGs": np.asarray([s.qGs for s in result.states]),
        "iterations": result.iterations,
    }


def compare_natural_overall(nstep: int = 3, dt: float = 20.0 * DAY):
    """Run both MRST formulations and return their histories side by side."""
    natural = run_compositional_validation_simple(nstep=nstep, dt=dt, use_natural=True)
    overall = run_compositional_validation_simple(nstep=nstep, dt=dt, use_natural=False)
    return {"natural": natural, "overall": overall}


def run_compositional_validation_packed(
    *,
    use_natural: bool = True,
    full_schedule: bool = False,
    nstep: int = 3,
    dt: float = 20.0 * DAY,
):
    """Run the MRST-style packed validation problem and collect outputs."""
    case = setup_compositional_validation_simple_problem(
        use_natural=use_natural,
        full_schedule=full_schedule,
        nstep=nstep,
        dt=dt,
    )
    simulate_packed_problem(case["problem"])
    problems, states, reports, names = get_multiple_packed_simulator_outputs(case["problem"])
    case.update({
        "packed_problems": problems,
        "states": states,
        "reports": reports,
        "names": names,
        "result": case["problem"].result,
    })
    case["plot_sections"] = validation_plot_sections(case)
    return case


def validation_comparison_data(run_output):
    """Return the data tuple MRST plots: MRST, E300, and AD-GPRS slots."""
    result = run_output["result"]
    states = result.states
    return {
        "data": {
            "MRST": states,
            "E300": run_output["external_references"]["E300"],
            "AD-GPRS": run_output["external_references"]["AD-GPRS"],
        },
        "names": ("MRST", "E300", "AD-GPRS"),
        "markers": ("-", "--", "--"),
        "component_names": run_output["setup"].mixture.names,
        "available": ("MRST",),
    }


def ternary_displacement_paths(states, every: int = 20):
    """Map state composition paths to MRST's ternary-diagram coordinates."""
    states = list(states)
    step = max(int(every), 1)
    selected = states[::step] if states else []
    if states and selected[-1] is not states[-1]:
        selected.append(states[-1])
    paths = []
    for state in selected:
        comp = np.asarray(state.z, dtype=float)
        denom = np.maximum(comp[:, 0] + comp[:, 1] + comp[:, 2], 1.0e-300)
        x = 0.5 * (2.0 * comp[:, 1] + comp[:, 2]) / denom
        y = (np.sqrt(3.0) / 2.0) * comp[:, 2] / denom
        paths.append(np.column_stack([x, y]))
    return {
        "paths": paths,
        "vertices": np.array([
            [0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]),
    }


def validation_plot_sections(run_output):
    """Return MRST's plotting-section metadata for validation-simple."""
    comparison = validation_comparison_data(run_output)
    states = comparison["data"]["MRST"]
    ternary = ternary_displacement_paths(states)
    component_legend = tuple(
        f"{name} {component}"
        for name in comparison["names"]
        for component in comparison["component_names"]
    )
    detailed_step = min(180, len(states)) if states else 0
    detailed = None
    if detailed_step:
        state = states[detailed_step - 1]
        pressure = np.asarray(state.pressure, dtype=float)
        pressure_norm = pressure / np.max(pressure) if pressure.size and np.max(pressure) != 0.0 else pressure
        detailed = {
            "step": int(detailed_step),
            "available": ("MRST",),
            "legend": ("sV", "Normalized pressure", *comparison["component_names"]),
            "legend_location": "northoutside",
            "legend_orientation": "horizontal",
            "ylim": (0.0, 1.0),
            "MRST": {
                "gas_saturation": None if state.sT is None else np.asarray(state.sT, dtype=float).copy(),
                "normalized_pressure": pressure_norm.copy(),
                "components": np.asarray(state.z, dtype=float).copy(),
                "marker": "-",
                "linewidth": 1.0,
            },
            "E300": None,
        }
    return {
        "component_comparison": {
            "figure_position_offset": (0, 0, 350, 0),
            "data": comparison["data"],
            "names": comparison["names"],
            "available": comparison["available"],
            "markers": comparison["markers"],
            "component_names": comparison["component_names"],
            "line_widths": (0.5, 2.5, 5.0),
            "legend": component_legend,
            "legend_location": "north",
            "legend_num_columns": 3,
            "ylim": (0.0, 1.0),
            "ylabel": "z",
        },
        "pressure_saturation_comparison": detailed,
        "state_toolbars": tuple(
            {
                "name": name,
                "available": name in comparison["available"],
                "plot1d": True,
                "title": name,
                "states": comparison["data"][name],
            }
            for name in comparison["names"]
        ),
        "ternary_diagram": {
            "vertices": ternary["vertices"],
            "paths": ternary["paths"],
            "axis": "off",
            "colormap": "parula",
            "component_labels": comparison["component_names"],
            "half_labels": (
                {"text": "0.5", "composition": (0.5, 0.5, 0.0)},
                {"text": "0.5", "composition": (0.0, 0.5, 0.5)},
                {"text": "0.5", "composition": (0.5, 0.0, 0.5)},
            ),
        },
    }


if __name__ == "__main__":
    def test_validation_run_enriches_injector():
        out = run_compositional_validation_simple(nstep=2, dt=10.0 * DAY, use_natural=True)
        setup = out["setup"]
        state = out["state"]
        assert out["pressure"].shape[0] == 2
        assert state.z[0, 1] > setup.state.z[0, 1]
        assert state.qGs[0] > 0.0
        assert state.qOs[1] + state.qGs[1] < 0.0
        assert out["schedule"].step.control.tolist() == [1, 1]
        assert out["iterations"].shape == (2,)
        print("ok test_validation_run_enriches_injector")

    def test_natural_overall_comparison_runs():
        out = compare_natural_overall(nstep=2, dt=10.0 * DAY)
        natural = out["natural"]
        overall = out["overall"]
        assert natural["pressure"].shape == overall["pressure"].shape
        assert natural["composition"].shape == overall["composition"].shape
        assert np.all(np.isfinite(natural["iterations"]))
        assert np.all(np.isfinite(overall["iterations"]))
        assert natural["state"].z[0, 1] > natural["setup"].state.z[0, 1]
        assert overall["state"].z[0, 1] > overall["setup"].state.z[0, 1]
        print("ok test_natural_overall_comparison_runs")

    def test_packed_problem_and_plot_data_match_mrst_shape():
        out = run_compositional_validation_packed(use_natural=True, nstep=2, dt=10.0 * DAY)
        comparison = validation_comparison_data(out)
        sections = validation_plot_sections(out)
        ternary = ternary_displacement_paths(out["result"].states, every=1)
        assert out["names"] == ["Natural"]
        assert out["problem"].directory == "simple_comp"
        assert comparison["names"] == ("MRST", "E300", "AD-GPRS")
        assert comparison["available"] == ("MRST",)
        assert comparison["data"]["E300"] is None
        assert len(ternary["paths"]) == 2
        assert ternary["paths"][0].shape[1] == 2
        assert ternary["vertices"].shape == (4, 2)
        assert sections["component_comparison"]["legend_location"] == "north"
        assert sections["state_toolbars"][0]["plot1d"] is True
        assert sections["ternary_diagram"]["axis"] == "off"
        print("ok test_packed_problem_and_plot_data_match_mrst_shape")

    test_validation_run_enriches_injector()
    test_natural_overall_comparison_runs()
    test_packed_problem_and_plot_data_match_mrst_shape()
    print("\nAll tests passed!")


__all__ = [
    "compare_natural_overall",
    "run_compositional_validation_packed",
    "run_compositional_validation_simple",
    "setup_compositional_validation_simple_problem",
    "ternary_displacement_paths",
    "validation_comparison_data",
    "validation_plot_sections",
]
