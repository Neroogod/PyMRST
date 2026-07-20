"""Port of MRST ``runCompositionalSPE1.m`` deck-conversion setup.

MRST's example converts the SPE1 black-oil deck/state into a two
pseudo-component compositional representation (``PseudoOil``/``PseudoGas``)
before running K-value compositional models.  PyMRST now ports the native deck
setup, state/well conversion path, the black-oil-derived K-value/property
backend, and the ``eos_model`` bridge used by the native compositional flow
equations.  This deck-sized example still advances SPE1 with PyMRST's mature
black-oil deck solver and converts every report state into the same
compositional state fields MRST constructs; reduced native K-value
compositional flow solves are validated separately in ``validate_phase112``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    DEFAULT_DECK,
    setup_spe1_tutorial,
)
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport
from pymrst.modules.compositional.fluids.blackoil import (
    convert_black_oil_state_to_compositional,
    convert_black_oil_model_to_compositional_model,
    pseudo_blackoil_mixture,
)
from pymrst.modules.compositional.model import CompositionalState, OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params import NORM_GRAVITY
from pymrst.utils.units import BARSA


@dataclass
class CompositionalSPE1Result:
    """Deck-converted SPE1 result."""

    deck: dict
    blackoil_model: object
    initial_state: object
    compositional_initial_state: object
    compositional_states: list[object]
    time: np.ndarray
    mixture: object
    compositional_model: object
    well_components: dict[str, np.ndarray]
    iterations: np.ndarray
    native: bool = False
    substeps: np.ndarray | None = None


@dataclass
class CompositionalSPE1Comparison:
    """MRST `runCompositionalSPE1` result bundle."""

    natural: CompositionalSPE1Result
    overall: CompositionalSPE1Result
    reference: CompositionalSPE1Result
    names: tuple[str, str, str] = (
        "MRST-NaturalVariables",
        "MRST-MolarVariables",
        "MRST-BlackOil",
    )
    plot_sections: dict | None = None


def setup_compositional_spe1(
    deck_path: str | Path | None = None,
    *,
    interpolation: str = "pchip",
):
    """Build SPE1 deck objects and the MRST-style converted initial state."""
    deck, model, state0 = setup_spe1_tutorial(
        deck_path=deck_path or DEFAULT_DECK,
        well_order=("INJECTOR", "PRODUCER"),
    )
    comp_state0 = convert_black_oil_state_to_compositional(model, state0)
    well_components = {}
    for well in model.wells:
        if well.name.upper() == "PRODUCER":
            # MRST switches schedule.control.W(2) from ORAT to BHP before
            # running the compositional K-value models.
            well.type = "bhp"
            well.val = 200.0 * BARSA
            well.lims = None
        # MRST uses W.components = W.compi(2:end) for the two pseudo components.
        well.components = np.asarray(well.compi[1:], dtype=float).copy()
        well_components[well.name] = well.components.copy()
    comp_model = convert_black_oil_model_to_compositional_model(
        model,
        interpolation=interpolation,
    )
    return deck, model, state0, comp_state0, comp_model, well_components


def _compositional_wells(blackoil_wells):
    """Convert SPE1 black-oil wells to MRST-style compositional wells."""
    wells = []
    for well in blackoil_wells:
        comp = np.asarray(well.compi, dtype=float)
        hydro = comp[1:].copy()
        if hydro.sum() > 0.0:
            hydro = hydro / hydro.sum()
        else:
            hydro[:] = 0.0
        cw = replace(
            well,
            compi=comp.copy(),
            components=hydro,
            lims=None if well.lims is None else dict(well.lims),
        )
        wells.append(cw)
    return wells


def native_compositional_spe1_model(
    deck_path: str | Path | None = None,
    *,
    natural: bool = False,
    max_iterations: int = 30,
    interpolation: str = "pchip",
):
    """Build the native K-value compositional SPE1 model and initial state."""
    deck, blackoil_model, state0, comp_state0, comp_model, well_components = setup_compositional_spe1(
        deck_path,
        interpolation=interpolation,
    )
    model_cls = NaturalVariablesCompositionalModel if natural else OverallCompositionModel
    rho_s = (blackoil_model.fluid.rho_s[1], blackoil_model.fluid.rho_s[2])
    model = model_cls(
        blackoil_model.G,
        blackoil_model.rock,
        comp_model.mixture,
        temperature=273.15,
        wells=_compositional_wells(blackoil_model.wells),
        rho_s=rho_s,
        water=True,
        eos_model=comp_model.eos_model,
        relperm_model=blackoil_model.fluid,
        gravity=NORM_GRAVITY,
        max_iterations=max_iterations,
        tol_cnv=1.0e-7,
        tol_well=1.0e-7,
        ds_max=0.2,
        dz_max=np.inf,
        dp_max_rel=np.inf,
    )
    return deck, blackoil_model, state0, comp_state0, comp_model, well_components, model


def _native_initial_state(model, converted_state, *, natural: bool = False):
    if natural:
        return model.initial_state(
            converted_state.pressure,
            converted_state.z,
            sW=converted_state.sW,
        )
    return CompositionalState(
        pressure=converted_state.pressure.copy(),
        z=converted_state.z.copy(),
        qOs=converted_state.qOs.copy(),
        qGs=converted_state.qGs.copy(),
        bhp=converted_state.bhp.copy(),
        sW=None if converted_state.sW is None else converted_state.sW.copy(),
        qWs=None if converted_state.qWs is None else converted_state.qWs.copy(),
        sT=None if converted_state.sT is None else converted_state.sT.copy(),
    )


def _try_native_step(model, state, dt: float, *, min_dt: float, max_cuts: int):
    """Advance one report step, recursively cutting if Newton fails."""
    stack = [float(dt)]
    accepted = []
    total_iterations = 0
    current = state
    cuts = 0
    while stack:
        local_dt = stack.pop(0)
        trial, ok, iterations = model.step(current, local_dt)
        if ok and np.all(np.isfinite(trial.pressure)) and np.all(np.isfinite(trial.z)):
            current = trial
            accepted.append(local_dt)
            total_iterations += int(iterations)
            continue
        if local_dt <= min_dt or cuts >= max_cuts:
            return current, False, total_iterations, np.asarray(accepted, dtype=float)
        half = 0.5 * local_dt
        stack = [half, half, *stack]
        cuts += 1
    return current, True, total_iterations, np.asarray(accepted, dtype=float)


def run_compositional_spe1(
    nsteps: int = 3,
    deck_path: str | Path | None = None,
    *,
    interpolation: str = "pchip",
) -> CompositionalSPE1Result:
    """Run SPE1 deck report steps and convert states to compositional form."""
    deck, model, state0, comp_state0, comp_model, well_components = setup_compositional_spe1(
        deck_path,
        interpolation=interpolation,
    )
    timesteps = np.asarray(deck["tsteps"], dtype=float)
    if nsteps is not None:
        timesteps = timesteps[:int(nsteps)]

    solver = NonLinearSolver(max_iterations=30, use_line_search=True)
    report = SimulationReport()
    state = state0.copy()
    comp_states = []
    elapsed = []
    total = 0.0
    for step_no, dt in enumerate(timesteps, start=1):
        state, step_report = solver.solve_timestep(model, state, float(dt))
        report.steps.append(step_report)
        if not step_report.converged:
            raise RuntimeError(
                f"SPE1 compositional deck step {step_no} failed: {step_report.residuals[-1]}"
            )
        comp_states.append(convert_black_oil_state_to_compositional(model, state))
        total += float(dt)
        elapsed.append(total)

    iterations = np.asarray([step.iterations for step in report.steps], dtype=int)
    return CompositionalSPE1Result(
        deck=deck,
        blackoil_model=model,
        initial_state=state0,
        compositional_initial_state=comp_state0,
        compositional_states=comp_states,
        time=np.asarray(elapsed, dtype=float),
        mixture=comp_model.mixture,
        compositional_model=comp_model,
        well_components=well_components,
        iterations=iterations,
    )


def run_native_compositional_spe1(
    nsteps: int = 1,
    deck_path: str | Path | None = None,
    *,
    natural: bool = False,
    max_report_step: float | None = None,
    min_dt: float = 3600.0,
    max_cuts: int = 20,
    interpolation: str = "pchip",
) -> CompositionalSPE1Result:
    """Run SPE1 with PyMRST's native K-value compositional model.

    SPE1 is read as a black-oil deck, converted to pseudo oil/gas components,
    and then advanced with the compositional conservation equations using the
    converted K-value EOS/property model. Report steps can be subdivided for
    nonlinear robustness; the returned ``time`` still records the original
    report times.
    """
    (
        deck,
        blackoil_model,
        state0,
        comp_state0,
        comp_model,
        well_components,
        native_model,
    ) = native_compositional_spe1_model(
        deck_path,
        natural=natural,
        interpolation=interpolation,
    )
    timesteps = np.asarray(deck["tsteps"], dtype=float)
    if nsteps is not None:
        timesteps = timesteps[:int(nsteps)]
    if max_report_step is not None:
        timesteps = np.minimum(timesteps, float(max_report_step))

    state = _native_initial_state(native_model, comp_state0, natural=natural)
    comp_states = []
    elapsed = []
    iterations = []
    substep_counts = []
    total = 0.0
    for step_no, dt in enumerate(timesteps, start=1):
        state, ok, iters, accepted = _try_native_step(
            native_model,
            state,
            float(dt),
            min_dt=float(min_dt),
            max_cuts=int(max_cuts),
        )
        if not ok:
            raise RuntimeError(f"native SPE1 compositional step {step_no} failed after timestep cuts")
        comp_states.append(state.copy())
        iterations.append(iters)
        substep_counts.append(accepted.size)
        total += float(dt)
        elapsed.append(total)

    return CompositionalSPE1Result(
        deck=deck,
        blackoil_model=blackoil_model,
        initial_state=state0,
        compositional_initial_state=comp_state0,
        compositional_states=comp_states,
        time=np.asarray(elapsed, dtype=float),
        mixture=comp_model.mixture,
        compositional_model=native_model,
        well_components=well_components,
        iterations=np.asarray(iterations, dtype=int),
        native=True,
        substeps=np.asarray(substep_counts, dtype=int),
    )


def run_compositional_spe1_comparison(
    nsteps: int = 1,
    deck_path: str | Path | None = None,
    *,
    min_dt: float = 6 * 3600.0,
    max_cuts: int = 20,
    interpolation: str = "pchip",
) -> CompositionalSPE1Comparison:
    """Run MRST's natural/overall/reference SPE1 comparison bundle."""
    natural = run_native_compositional_spe1(
        nsteps=nsteps,
        deck_path=deck_path,
        natural=True,
        min_dt=min_dt,
        max_cuts=max_cuts,
        interpolation=interpolation,
    )
    overall = run_native_compositional_spe1(
        nsteps=nsteps,
        deck_path=deck_path,
        natural=False,
        min_dt=min_dt,
        max_cuts=max_cuts,
        interpolation=interpolation,
    )
    reference = run_compositional_spe1(
        nsteps=nsteps,
        deck_path=deck_path,
        interpolation=interpolation,
    )
    comparison = CompositionalSPE1Comparison(natural=natural, overall=overall, reference=reference)
    comparison.plot_sections = spe1_plot_sections(comparison)
    return comparison


def spe1_well_solution_datasets(comparison: CompositionalSPE1Comparison) -> dict:
    """Return the well-solution datasets plotted by MRST's example."""
    datasets = {}
    for name, result in zip(comparison.names, (comparison.natural, comparison.overall, comparison.reference)):
        states = result.compositional_states
        datasets[name] = {
            "time": result.time.copy(),
            "bhp": np.vstack([st.bhp for st in states]),
            "qOs": np.vstack([st.qOs for st in states]),
            "qGs": np.vstack([st.qGs for st in states]),
        }
        if states and states[0].qWs is not None:
            datasets[name]["qWs"] = np.vstack([st.qWs for st in states])
    return {
        "names": comparison.names,
        "datasets": datasets,
    }


def spe1_reference_pseudo_component_rates(reference: CompositionalSPE1Result) -> dict:
    """Return MRST's `PseudoOil`/`PseudoGas` fields for the reference wells."""
    rho_os = float(reference.blackoil_model.fluid.rho_s[1])
    rho_gs = float(reference.blackoil_model.fluid.rho_s[2])
    return {
        "PseudoOil": np.vstack([st.qOs * rho_os for st in reference.compositional_states]),
        "PseudoGas": np.vstack([st.qGs * rho_gs for st in reference.compositional_states]),
        "rhoOS": rho_os,
        "rhoGS": rho_gs,
    }


def spe1_cumulative_iterations(comparison: CompositionalSPE1Comparison) -> dict:
    """Return MRST's cumulative nonlinear-iteration stairs plot data."""
    arrays = [comparison.natural.iterations, comparison.overall.iterations, comparison.reference.iterations]
    n = min(np.asarray(a).size for a in arrays)
    table = np.column_stack([np.asarray(a, dtype=int).ravel()[:n] for a in arrays]) if n else np.empty((0, 3), dtype=int)
    return {
        "names": comparison.names,
        "iterations": table,
        "cumulative": np.cumsum(table, axis=0),
    }


def spe1_plot_sections(comparison: CompositionalSPE1Comparison) -> dict:
    """Return MRST's interactive SPE1 plotting section metadata."""
    wells = spe1_well_solution_datasets(comparison)
    its = spe1_cumulative_iterations(comparison)
    return {
        "state_toolbars": (
            {
                "dataset": comparison.names[0],
                "title": "Compositional Natural Variables",
                "axis": "tight",
                "view": (30, 30),
                "states": comparison.natural.compositional_states,
            },
            {
                "dataset": comparison.names[1],
                "title": "Compositional Molar Variables",
                "axis": "tight",
                "view": (30, 30),
                "states": comparison.overall.compositional_states,
            },
            {
                "dataset": comparison.names[2],
                "title": "Black-oil",
                "axis": "tight",
                "view": (30, 30),
                "states": comparison.reference.compositional_states,
            },
        ),
        "well_solutions": {
            "datasetnames": comparison.names,
            "datasets": wells["datasets"],
        },
        "cumulative_iterations": {
            "names": its["names"],
            "iterations": its["iterations"],
            "cumulative": its["cumulative"],
            "plot": "stairs",
            "legend": its["names"],
            "ylabel": "Cumulative nonlinear iterations",
        },
    }


if __name__ == "__main__":
    def test_spe1_deck_conversion_runs():
        out = run_compositional_spe1(nsteps=2)
        assert out.mixture.names == ("PseudoOil", "PseudoGas")
        assert out.compositional_initial_state.z.shape == (300, 2)
        assert len(out.compositional_states) == 2
        assert np.allclose(out.compositional_initial_state.z.sum(axis=1), 1.0)
        assert np.allclose(out.compositional_states[-1].z.sum(axis=1), 1.0)
        assert set(out.well_components) == {"PRODUCER", "INJECTOR"}
        assert out.time[-1] > 0.0 and out.time[-1] <= out.deck["tsteps"].sum()
        print("ok test_spe1_deck_conversion_runs")

    def test_native_spe1_compositional_runs():
        out = run_native_compositional_spe1(nsteps=1, min_dt=6 * 3600.0)
        assert out.native
        assert out.mixture.names == ("PseudoOil", "PseudoGas")
        assert len(out.compositional_states) == 1
        st = out.compositional_states[-1]
        assert np.allclose(st.z.sum(axis=1), 1.0)
        assert st.sW is not None and np.all((st.sW >= 0.0) & (st.sW <= 1.0))
        assert np.all(np.isfinite(st.pressure))
        print("ok test_native_spe1_compositional_runs")

    def test_spe1_comparison_runs_three_mrst_paths():
        out = run_compositional_spe1_comparison(nsteps=1)
        assert out.names == ("MRST-NaturalVariables", "MRST-MolarVariables", "MRST-BlackOil")
        assert out.natural.native and out.overall.native
        assert not out.reference.native
        assert len(out.natural.compositional_states) == 1
        assert len(out.overall.compositional_states) == 1
        assert len(out.reference.compositional_states) == 1
        wells = spe1_well_solution_datasets(out)
        pseudo = spe1_reference_pseudo_component_rates(out.reference)
        its = spe1_cumulative_iterations(out)
        sections = spe1_plot_sections(out)
        assert wells["names"] == out.names
        assert pseudo["PseudoOil"].shape[0] == 1
        assert its["cumulative"].shape == (1, 3)
        assert out.plot_sections is not None
        assert sections["state_toolbars"][0]["title"] == "Compositional Natural Variables"
        assert sections["cumulative_iterations"]["plot"] == "stairs"
        print("ok test_spe1_comparison_runs_three_mrst_paths")

    test_spe1_deck_conversion_runs()
    test_native_spe1_compositional_runs()
    test_spe1_comparison_runs_three_mrst_paths()
    print("\nAll tests passed!")


__all__ = [
    "CompositionalSPE1Result",
    "CompositionalSPE1Comparison",
    "pseudo_blackoil_mixture",
    "native_compositional_spe1_model",
    "run_compositional_spe1",
    "run_compositional_spe1_comparison",
    "run_native_compositional_spe1",
    "spe1_cumulative_iterations",
    "spe1_plot_sections",
    "spe1_reference_pseudo_component_rates",
    "spe1_well_solution_datasets",
    "setup_compositional_spe1",
]
