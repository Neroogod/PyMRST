"""MRST ``ad-blackoil/examples/spe1/blackoilTutorialSPE1.m``.

The Odeh benchmark (SPE1): gas injected into a 10x10x3 undersaturated
reservoir, producer and injector in opposite corners. `blackoil_tutorial_spe1`
is the tutorial, section by section, on the AD-OO stack:
- `setupSPE1` and `selectModelFromDeck` build the case;
- `convertDeckScheduleToMRST` builds the schedule;
- `simulateScheduleAD` runs it with a line-searching `NonLinearSolver`;
- the solution is plotted (`plotWellSols`, `plotToolbar`) and compared
  with Eclipse 100 (`SPE1_smry.mat`): producer gas/oil ratio and both
  wells' bottom-hole pressures.

The rest of this file is PyMRST's earlier SPE1 tutorial on the legacy
models (`setup_spe1_tutorial`, `run_blackoil_tutorial_spe1`, ...), kept for
the code that imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
    initial_state_from_deck,
)
from pymrst.modules.ad_blackoil.backend_blackoil_fluid import (
    init_backend_blackoil_fluid,
)
from pymrst.modules.ad_blackoil.backend_blackoil_model import (
    BackendBOState,
    BackendBlackOilModel,
)
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport
from pymrst.backends import NewtonReport
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.units import BARSA, DAY, DARCY, MILLI

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_DECK = ROOT / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


@dataclass
class SPE1TutorialResult:
    deck: dict
    model: object
    initial_state: object
    states: list[object]
    report: object
    history: dict[str, np.ndarray]
    time: np.ndarray
    producer_index: int
    injector_index: int


def setup_spe1_tutorial(
    deck_path: str | Path | None = None,
    well_order: Sequence[str] = ("PRODUCER", "INJECTOR"),
):
    """Build the SPE1 objects corresponding to MRST's ``setupSPE1`` section."""
    deck = read_spe1_deck(Path(deck_path) if deck_path is not None else DEFAULT_DECK)
    grid = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(grid, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)

    wells = None
    for name in well_order:
        wd = deck["wells"][name]
        wells = add_well(
            wells,
            grid,
            rock,
            wd["cells"],
            well_type=wd["type"],
            val=wd["val"],
            radius=wd["radius"],
            compi=wd["compi"],
            sign=wd["sign"],
            name=name,
            ref_depth=wd["ref_depth"],
            gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
        )

    model = GenericBlackOilModelAD(
        grid,
        rock,
        fluid,
        wells,
        disgas=True,
        gravity=NORM_GRAVITY,
        tol_well=1e-8,
    )
    state0 = initial_state_from_deck(model, deck)
    return deck, model, state0


def _collect_history(states: Sequence[State]) -> dict[str, np.ndarray]:
    keys = ("pressure", "sW", "sG", "rs", "qWs", "qOs", "qGs", "bhp")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for state in states:
        for key in keys:
            out[key].append(np.asarray(state[key], dtype=float).copy())
    return {key: np.vstack(values) for key, values in out.items()}


def _collect_backend_history(states: Sequence[BackendBOState]) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {
        "pressure": [],
        "sW": [],
        "sG": [],
        "rs": [],
        "qWs": [],
        "qOs": [],
        "qGs": [],
        "bhp": [],
    }
    for state in states:
        out["pressure"].append(state.pressure.copy())
        out["sW"].append(state.s[:, 0].copy())
        out["sG"].append(state.s[:, 2].copy())
        out["rs"].append(state.rs.copy())
        out["qWs"].append(state.qWs.copy())
        out["qOs"].append(state.qOs.copy())
        out["qGs"].append(state.qGs.copy())
        out["bhp"].append(state.bhp.copy())
    return {key: np.vstack(values) for key, values in out.items()}


def _well_indices(model: GenericBlackOilModelAD) -> tuple[int, int]:
    signs = np.array([w.sign for w in model.wells], dtype=int)
    producers = np.flatnonzero(signs < 0)
    injectors = np.flatnonzero(signs > 0)
    if producers.size != 1 or injectors.size != 1:
        raise ValueError("SPE1 tutorial expects one producer and one injector")
    return int(producers[0]), int(injectors[0])


def _format_elapsed_days(seconds: float) -> str:
    days = float(seconds) / DAY
    if np.isclose(days, round(days), rtol=0.0, atol=1e-10):
        value = int(round(days))
        label = "Day" if value == 1 else "Days"
        return f"{value} {label}"
    label = "Day" if np.isclose(days, 1.0, rtol=0.0, atol=1e-10) else "Days"
    return f"{days:g} {label}"


def _print_timestep_progress(step_no: int, n_steps: int, start: float, dt: float) -> None:
    start_label = "" if np.isclose(start, 0.0) else _format_elapsed_days(start)
    end_label = _format_elapsed_days(start + dt)
    print(
        f"Solving timestep {step_no:03d}/{n_steps:03d}: "
        f"{start_label:<30} -> {end_label}",
        flush=True,
    )


def run_blackoil_tutorial_spe1(
    nsteps: int | None = None,
    *,
    deck_path: str | Path | None = None,
    well_order: Sequence[str] = ("PRODUCER", "INJECTOR"),
    use_line_search: bool = True,
    verbose: bool = True,
    progress: str | None = None,
) -> SPE1TutorialResult:
    """Run the PyMRST version of MRST's SPE1 tutorial.

    `nsteps=None` runs the full schedule available in the PyMRST SPE1 deck
    parser. Use a smaller integer for quick presentation checks.
    """
    deck, model, state0 = setup_spe1_tutorial(deck_path, well_order)
    timesteps = np.asarray(deck["tsteps"], dtype=float)
    if nsteps is not None:
        timesteps = timesteps[:int(nsteps)]

    solver = NonLinearSolver(
        max_iterations=30,
        use_line_search=use_line_search,
        verbose=verbose,
        progress=progress,
    )
    report = SimulationReport()
    states: list[State] = []
    state = state0.copy()
    elapsed = []
    total = 0.0

    for step_no, dt in enumerate(timesteps, start=1):
        if verbose:
            _print_timestep_progress(step_no, len(timesteps), total, float(dt))
        state, step_report = solver.solve_timestep(model, state, float(dt))
        report.steps.append(step_report)
        if not step_report.converged:
            raise RuntimeError(
                f"SPE1 tutorial step {step_no} failed: {step_report.residuals[-1]}"
            )
        states.append(state.copy())
        total += float(dt)
        elapsed.append(total)

    producer, injector = _well_indices(model)
    return SPE1TutorialResult(
        deck=deck,
        model=model,
        initial_state=state0,
        states=states,
        report=report,
        history=_collect_history(states),
        time=np.asarray(elapsed, dtype=float),
        producer_index=producer,
        injector_index=injector,
    )


def run_backend_blackoil_tutorial_spe1(
    nsteps: int | None = None,
    *,
    deck_path: str | Path | None = None,
    well_order: Sequence[str] = ("PRODUCER", "INJECTOR"),
    backend: str = "torch",
    device: str | None = None,
    dtype: str = "float64",
    max_iterations: int = 15,
    tolerance: float = 1e-7,
    damping: float = 1.0,
    linear_solver: str = "dense",
    jacobian_drop_tolerance: float = 0.0,
    gmres_tolerance: float = 1e-8,
    gmres_max_iterations: int = 80,
    gmres_restart: int | None = None,
    gmres_scaling: str = "none",
    verbose: bool = False,
) -> SPE1TutorialResult:
    """Run SPE1 through the experimental backend Torch black-oil model.

    This follows the same deck/grid/well setup as the regular tutorial, but
    residual and Jacobian assembly are Torch-native. The current backend Newton
    solve is dense-autograd and is intended for validation/SPE1-scale GPU
    prototyping, not field-scale sparse runs.
    """
    deck, ref_model, state0_graph = setup_spe1_tutorial(deck_path, well_order)
    backend_fluid = init_backend_blackoil_fluid(
        deck,
        backend=backend,
        device=device,
        dtype=dtype,
    )
    model = BackendBlackOilModel(
        ref_model.G,
        ref_model.rock,
        backend_fluid,
        ref_model.wells,
        gravity=NORM_GRAVITY,
        backend=backend,
        device=device,
        dtype=dtype,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
        linear_solver=linear_solver,
        jacobian_drop_tolerance=jacobian_drop_tolerance,
        gmres_tolerance=gmres_tolerance,
        gmres_max_iterations=gmres_max_iterations,
        gmres_restart=gmres_restart,
        gmres_scaling=gmres_scaling,
    )
    state0 = BackendBOState.from_state(state0_graph)
    timesteps = np.asarray(deck["tsteps"], dtype=float)
    if nsteps is not None:
        timesteps = timesteps[:int(nsteps)]

    states: list[BackendBOState] = []
    reports: list[NewtonReport] = []
    state = state0.copy()
    elapsed = []
    total = 0.0
    for step_no, dt in enumerate(timesteps, start=1):
        if verbose:
            _print_timestep_progress(step_no, len(timesteps), total, float(dt))
        state, step_report = model.step(state, float(dt))
        reports.append(step_report)
        if not step_report.converged:
            msg = f"Backend SPE1 step {step_no} failed: {step_report.residual_norms[-1]}"
            if step_report.linear_residuals:
                msg += (
                    f" (last linear residual {step_report.linear_residuals[-1]}, "
                    f"linear iterations {step_report.linear_iterations[-1]})"
                )
            raise RuntimeError(msg)
        states.append(state.copy())
        total += float(dt)
        elapsed.append(total)

    producer, injector = _well_indices(model)
    return SPE1TutorialResult(
        deck=deck,
        model=model,
        initial_state=state0,
        states=states,
        report=reports,
        history=_collect_backend_history(states),
        time=np.asarray(elapsed, dtype=float),
        producer_index=producer,
        injector_index=injector,
    )


def producer_gor(result: SPE1TutorialResult) -> np.ndarray:
    """Producer gas/oil ratio, matching MRST's final tutorial plot quantity."""
    qg = result.history["qGs"][:, result.producer_index]
    qo = result.history["qOs"][:, result.producer_index]
    return qg / qo


def plot_toolbar_fields(result: SPE1TutorialResult) -> dict[str, np.ndarray]:
    """Cell-wise SPE1 fields prepared for PyMRST's interactive plotToolbar."""
    history = result.history
    return {
        "pressure_bar": history["pressure"] / BARSA,
        "water_saturation": history["sW"],
        "gas_saturation": history["sG"],
        "oil_saturation": 1.0 - history["sW"] - history["sG"],
        "rs": history["rs"],
    }


def plot_toolbar_spe1(
    result: SPE1TutorialResult,
    *,
    show: bool = True,
    off_screen: bool = False,
    use_qt: bool = True,
    title: str = "PyMRST SPE1 black-oil tutorial",
):
    """Open PyMRST's interactive plotToolbar for the SPE1 report states."""
    from pymrst.plotting import plot_toolbar

    return plot_toolbar(
        result.model.G,
        plot_toolbar_fields(result),
        show=show,
        times=result.time,
        time_unit=DAY,
        time_label="days",
        off_screen=off_screen,
        use_qt=use_qt,
        title=title,
    )


def plot_browser_spe1(
    result: SPE1TutorialResult,
    *,
    filename: str | Path = "examples/results_spe1/spe1.browser.html",
    style: str = "mesh",
    open_browser: bool = False,
    zscale: float = 1.0,
) -> Path:
    """Write the complete SPE1 history to the interactive web viewer."""
    from pymrst.plotting.extensions.browser import plot_reservoir_browser

    well_series = {
        key: np.asarray(result.history[key], dtype=float)
        for key in ("bhp", "qWs", "qOs", "qGs")
        if key in result.history
    }
    return plot_reservoir_browser(
        result.model.G,
        plot_toolbar_fields(result),
        wells=result.model.wells,
        times=result.time,
        filename=filename,
        title="PyMRST SPE1 black-oil tutorial",
        zscale=zscale,
        style=style,
        well_series=well_series,
        rock=result.model.rock,
        open_browser=open_browser,
    )


def tutorial_summary(result: SPE1TutorialResult) -> dict[str, float]:
    p = result.history["pressure"][-1]
    bhp = result.history["bhp"][-1]
    gor = producer_gor(result)
    if isinstance(result.report, list):
        total_iterations = sum(rep.iterations for rep in result.report)
    else:
        total_iterations = result.report.total_iterations
    return {
        "steps": float(len(result.states)),
        "days": float(result.time[-1] / DAY if result.time.size else 0.0),
        "newton_iterations": float(total_iterations),
        "pressure_min_pa": float(p.min()),
        "pressure_max_pa": float(p.max()),
        "producer_bhp_pa": float(bhp[result.producer_index]),
        "injector_bhp_pa": float(bhp[result.injector_index]),
        "producer_gor_final": float(gor[-1]),
    }


def print_tutorial_summary(result: SPE1TutorialResult) -> None:
    summary = tutorial_summary(result)
    print("PyMRST SPE1 black-oil tutorial")
    print(f"  report steps: {int(summary['steps'])}")
    print(f"  simulated time: {summary['days']:.1f} days")
    print(f"  Newton iterations: {int(summary['newton_iterations'])}")
    print(
        "  final pressure range: "
        f"{summary['pressure_min_pa']:.6e} - {summary['pressure_max_pa']:.6e} Pa"
    )
    print(f"  final producer BHP: {summary['producer_bhp_pa']:.6e} Pa")
    print(f"  final injector BHP: {summary['injector_bhp_pa']:.6e} Pa")
    print(f"  final producer GOR: {summary['producer_gor_final']:.6e}")


def save_tutorial_states(
    result: SPE1TutorialResult,
    filename: str | Path,
) -> Path:
    """Persist MRST-like SPE1 state histories to a compressed ``.npz`` file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    wells = result.model.wells
    g = result.model.G
    rock = result.model.rock
    payload = {
        "time": np.asarray(result.time, dtype=float),
        "producer_index": np.asarray(result.producer_index, dtype=int),
        "injector_index": np.asarray(result.injector_index, dtype=int),
        "well_names": np.asarray([getattr(w, "name", f"W{i + 1}") for i, w in enumerate(wells)]),
        "well_sign": np.asarray([w.sign for w in wells], dtype=int),
        "cell_centroids": np.asarray(g.cells.centroids, dtype=float),
        "grid.griddim": np.asarray(g.griddim, dtype=int),
        "grid.cartDims": np.asarray(g.cartDims if g.cartDims is not None else [], dtype=int),
        "grid.cells.num": np.asarray(g.cells.num, dtype=int),
        "grid.cells.facePos": np.asarray(g.cells.facePos, dtype=int),
        "grid.cells.faces": np.asarray(g.cells.faces, dtype=int),
        "grid.cells.indexMap": np.asarray(g.cells.indexMap, dtype=int),
        "grid.cells.volumes": np.asarray(g.cells.volumes, dtype=float),
        "grid.cells.centroids": np.asarray(g.cells.centroids, dtype=float),
        "grid.faces.num": np.asarray(g.faces.num, dtype=int),
        "grid.faces.nodePos": np.asarray(g.faces.nodePos, dtype=int),
        "grid.faces.nodes": np.asarray(g.faces.nodes, dtype=int),
        "grid.faces.neighbors": np.asarray(g.faces.neighbors, dtype=int),
        "grid.faces.areas": np.asarray(g.faces.areas, dtype=float),
        "grid.faces.normals": np.asarray(g.faces.normals, dtype=float),
        "grid.faces.centroids": np.asarray(g.faces.centroids, dtype=float),
        "grid.nodes.num": np.asarray(g.nodes.num, dtype=int),
        "grid.nodes.coords": np.asarray(g.nodes.coords, dtype=float),
        "rock.perm": np.asarray(rock.perm, dtype=float),
        "rock.poro": np.asarray(rock.poro, dtype=float),
    }
    if getattr(rock, "ntg", None) is not None:
        payload["rock.ntg"] = np.asarray(rock.ntg, dtype=float)
    for i, w in enumerate(wells):
        prefix = f"well.{i}"
        payload[f"{prefix}.cells"] = np.asarray(w.cells, dtype=int)
        payload[f"{prefix}.r"] = np.asarray(w.r, dtype=float)
        payload[f"{prefix}.WI"] = np.asarray(w.WI, dtype=float)
        payload[f"{prefix}.dZ"] = np.asarray(w.dZ, dtype=float)
        payload[f"{prefix}.compi"] = np.asarray(w.compi, dtype=float)
        payload[f"{prefix}.val"] = np.asarray(w.val, dtype=float)
        payload[f"{prefix}.dir"] = np.asarray(w.dir)
        payload[f"{prefix}.type"] = np.asarray(w.type)
        payload[f"{prefix}.name"] = np.asarray(getattr(w, "name", f"W{i + 1}"))
        payload[f"{prefix}.refDepth"] = np.asarray(w.refDepth, dtype=float)
        payload[f"{prefix}.sign"] = np.asarray(w.sign, dtype=int)
    payload.update({key: np.asarray(value, dtype=float) for key, value in result.history.items()})
    np.savez_compressed(path, **payload)
    return path


def plot_permeability_and_wells(
    result: SPE1TutorialResult,
    *,
    filename: str | Path | None = None,
    show: bool = True,
):
    """Plot permeability and wells, matching MRST's first SPE1 figure."""
    plt = _pyplot()
    g = result.model.G
    perm_md = np.asarray(result.model.rock.perm[:, 0], dtype=float) / (MILLI * DARCY)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_cells(ax, g, perm_md, "Permeability (mD)", cmap="viridis", alpha=0.65)
    _plot_well_paths(ax, g, result.model.wells)
    ax.set_title("SPE1 permeability and wells")
    _format_3d_axes(ax)
    if filename is not None:
        fig.savefig(filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return Path(filename)
    if show:
        plt.show()
    return fig


def plot_state_snapshot(
    result: SPE1TutorialResult,
    *,
    step: int = -1,
    field: str = "sG",
    filename: str | Path | None = None,
    show: bool = True,
):
    """Plot a reservoir state field, corresponding to MRST's plotToolbar view."""
    plt = _pyplot()
    g = result.model.G
    values = np.asarray(result.history[field][step], dtype=float)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_cells(ax, g, values, f"{field} step {step}", cmap="viridis", alpha=0.8)
    _plot_well_paths(ax, g, result.model.wells)
    ax.set_title(f"SPE1 {field} at step {step}")
    _format_3d_axes(ax)
    if filename is not None:
        fig.savefig(filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return Path(filename)
    if show:
        plt.show()
    return fig


def plot_well_response_curves(
    result: SPE1TutorialResult,
    *,
    filename: str | Path | None = None,
    show: bool = True,
):
    """Plot producer GOR and producer/injector BHP like the MRST tutorial."""
    plt = _pyplot()

    t_year = result.time / (365.0 * DAY)
    gor = producer_gor(result)
    bhp = result.history["bhp"] / BARSA

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), constrained_layout=True)
    axes[0].plot(t_year, gor, linewidth=2)
    axes[0].set_title("Producer gas/oil ratio")
    axes[0].set_xlabel("Time [years]")
    axes[0].set_ylabel("qGs / qOs")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_year, bhp[:, result.producer_index], linewidth=2)
    axes[1].set_title("Bottom-hole pressure (producer)")
    axes[1].set_xlabel("Time [years]")
    axes[1].set_ylabel("bar")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_year, bhp[:, result.injector_index], linewidth=2)
    axes[2].set_title("Bottom-hole pressure (injector)")
    axes[2].set_xlabel("Time [years]")
    axes[2].set_ylabel("bar")
    axes[2].grid(True, alpha=0.3)

    if filename is not None:
        fig.savefig(filename, dpi=180)
        plt.close(fig)
        return Path(filename)
    if show:
        plt.show()
    return fig


def _pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "pymrst_matplotlib"),
    )
    import matplotlib

    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _scatter_cells(ax, g, values, label, *, cmap="viridis", alpha=0.8):
    c = g.cells.centroids
    z = c[:, 2] if c.shape[1] > 2 else np.zeros(c.shape[0])
    sc = ax.scatter(c[:, 0], c[:, 1], z, c=values, s=34, cmap=cmap, alpha=alpha)
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
    cb.set_label(label)
    return sc


def _plot_well_paths(ax, g, wells):
    for w in wells:
        pts = g.cells.centroids[np.asarray(w.cells, dtype=int)]
        z = pts[:, 2] if pts.shape[1] > 2 else np.zeros(pts.shape[0])
        color = "tab:blue" if w.sign > 0 else "tab:red"
        label = getattr(w, "name", "well")
        ax.plot(pts[:, 0], pts[:, 1], z, "o-", color=color, linewidth=2.0, label=label)
    ax.legend(loc="upper right")


def _format_3d_axes(ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=28, azim=-45)


def plot_blackoil_tutorial_spe1(
    result: SPE1TutorialResult,
    *,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, Path | object]:
    """Create the SPE1 tutorial plots.

    If `output_dir` is provided, PNG files are written there and the returned
    dictionary contains paths. Otherwise interactive plot objects are returned.
    """
    outdir = Path(output_dir) if output_dir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
    files = {
        "permeability_wells": None if outdir is None else outdir / "spe1_permeability_wells.png",
        "well_responses": None if outdir is None else outdir / "spe1_well_responses.png",
        "gas_saturation": None if outdir is None else outdir / "spe1_final_gas_saturation.png",
        "pressure": None if outdir is None else outdir / "spe1_final_pressure.png",
    }
    return {
        "permeability_wells": plot_permeability_and_wells(
            result, filename=files["permeability_wells"], show=show
        ),
        "well_responses": plot_well_response_curves(
            result, filename=files["well_responses"], show=show
        ),
        "gas_saturation": plot_state_snapshot(
            result, field="sG", filename=files["gas_saturation"], show=show
        ),
        "pressure": plot_state_snapshot(
            result, field="pressure", filename=files["pressure"], show=show
        ),
    }



# -- MRST blackoilTutorialSPE1, on the AD-OO stack ---------------------------

SPE1_SUMMARY = ROOT/"mrst-2024a-Geomech/modules/ad-blackoil/examples/spe1/SPE1_smry.mat"


def load_spe1_smry(path=SPE1_SUMMARY):
    """MATLAB's `load SPE1_smry`: the Eclipse summary, with `smry.get`.

    The file stores the `readEclipseSummaryUnFmt` struct; its `get` is a
    MATLAB function handle, so the struct is rebuilt as PyMRST's `Summary`
    (0-based name and keyword indices).
    """
    from scipy.io import loadmat

    from pymrst.modules.ad_core.utils.read_summary_local import Summary
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)["smry"]
    return Summary(WGNAMES=[str(n) for n in raw.WGNAMES],
                   KEYWORDS=[str(k) for k in raw.KEYWORDS],
                   nInx=np.asarray(raw.nInx, dtype=np.int64) - 1,
                   kInx=np.asarray(raw.kInx, dtype=np.int64) - 1,
                   data=np.asarray(raw.data, dtype=float))


def blackoil_tutorial_spe1(plot: bool = True):
    """MRST `blackoilTutorialSPE1`. Returns the run and the comparison data."""
    from types import SimpleNamespace

    from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.ad_core.utils.well_sol_to_vector import well_sol_to_vector
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.utils.units import BARSA, DARCY, FT, MILLI, PSIA, STB, YEAR

    # Set up the problem. The model comes from the deck: three-phase black
    # oil with dissolved gas.
    G, rock, fluid, deck, state = setup_spe1()
    model = select_model_from_deck(G, rock, fluid, deck)
    print(model)
    # Convert the deck schedule into an MRST schedule by parsing the wells.
    schedule = convert_deck_schedule_to_mrst(model, deck)

    # Plot well and permeability: three layers, injector completed in the
    # upper one and producer in the lower.
    if plot:
        import pyvista as pv

        from pymrst.plotting.plot_cell_data import plot_cell_data
        from pymrst.plotting.plot_well import plot_well
        figure = pv.Plotter()
        plot_cell_data(G, np.asarray(rock.perm)[:, 0]/(MILLI*DARCY), plotter=figure,
                       opacity=0.5, edge_color="black", show=False,
                       label="Permeability (mD)")
        plot_well(G, schedule.controls[0].W, figure, radius=.5)
        figure.add_title("Permeability (mD)")
        figure.camera_position = "iso"
        figure.show()

    # Run the entire schedule.
    nls = NonLinearSolver(useLinesearch=True)
    wellSols, states, report = simulate_schedule_ad(state, model, schedule,
                                                    NonLinearSolver=nls)

    # Plot the well solutions and simulator states.
    if plot:
        from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
        from pymrst.plotting.extensions.plot_toolbar import plot_toolbar
        plot_well_sols(wellSols, report["ReservoirTime"])
        toolbar = plot_toolbar(G, states, show=False, off_screen=pv.OFF_SCREEN)
        plot_well(G, schedule.controls[0].W, toolbar.pl)
        if not toolbar.off_screen:
            toolbar.show()

    # Compare with a commercial simulator: load the summary and find the
    # producer and injector. The first summary entry is zero, so skip it.
    smry = load_spe1_smry()
    inj = int(np.flatnonzero([w.sign == 1 for w in wellSols[0]])[0])
    prod = int(np.flatnonzero([w.sign == -1 for w in wellSols[0]])[0])
    ind = np.arange(1, smry.data.shape[1])
    qWs, qOs, qGs, bhp = well_sol_to_vector(wellSols)
    T = np.cumsum(schedule.step_val)/YEAR
    Tcomp = smry.get(":+:+:+:+", "YEARS", ind).ravel()
    out = SimpleNamespace(
        G=G, model=model, schedule=schedule, wellSols=wellSols, states=states,
        report=report, qWs=qWs, qOs=qOs, qGs=qGs, bhp=bhp, T=T, Tcomp=Tcomp,
        inj=inj, prod=prod,
        gor_ecl=smry.get("PRODUCER", "WGOR", ind).ravel()*(1000*FT**3/STB),
        gor_mrst=qGs[:, prod]/qOs[:, prod],
        bhp_prod_ecl=smry.get("PRODUCER", "WBHP", ind).ravel()*PSIA,
        bhp_inj_ecl=smry.get("INJECTOR", "WBHP", ind).ravel()*PSIA)
    if plot:
        import matplotlib.pyplot as plt
        mrstarg = dict(linewidth=2)
        eclarg = dict(linestyle="none", marker="o", markersize=5,
                      markerfacecolor=[.6, .6, .6], color="r")
        # The producer gas/oil ratio is the interesting part of SPE1.
        plt.figure(3)
        plt.clf()
        plt.plot(T, out.gor_mrst, **mrstarg)
        plt.plot(Tcomp, out.gor_ecl, **eclarg)
        plt.legend(["MRST", "Eclipse"])
        plt.xlabel("Time [Years]")
        plt.title("Gas rate / Oil rate")
        # The injector is rate controlled, so its BHP comes out of the solve.
        for fig, well, ecl, title in ((4, prod, out.bhp_prod_ecl,
                                       "Bottom-hole pressure (producer)"),
                                      (5, inj, out.bhp_inj_ecl,
                                       "Bottom-hole pressure (injector)")):
            plt.figure(fig)
            plt.clf()
            plt.plot(T, bhp[:, well]/BARSA, **mrstarg)
            plt.plot(Tcomp, ecl/BARSA, **eclarg)
            plt.legend(["MRST", "Eclipse"])
            plt.xlabel("Time [Years]")
            plt.ylabel("bar")
            plt.title(title)
        plt.show()
    return out


blackoilTutorialSPE1 = blackoil_tutorial_spe1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PyMRST SPE1 black-oil tutorial.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of report steps to run. Omit to run the full SPE1 schedule.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate tutorial plots.")
    parser.add_argument(
        "--backend",
        choices=("adi", "torch"),
        default="adi",
        help="Run the standard ADI tutorial or the experimental Torch backend.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for --backend torch, for example cpu or cuda.",
    )
    parser.add_argument(
        "--linear-solver",
        choices=(
            "dense",
            "scipy_sparse",
            "cupy_sparse",
            "matrix_free_gmres",
            "matrix_free_gmres_fallback",
        ),
        default="dense",
        help=(
            "Newton linear solver for --backend torch. matrix_free_gmres avoids "
            "forming the full Jacobian; matrix_free_gmres_fallback attempts "
            "that path and falls back to dense when unpreconditioned GMRES "
            "stalls; cupy_sparse requires CuPy and CUDA."
        ),
    )
    parser.add_argument(
        "--jacobian-drop-tolerance",
        type=float,
        default=0.0,
        help="Drop Jacobian entries at or below this absolute value for sparse backend solves.",
    )
    parser.add_argument(
        "--gmres-tolerance",
        type=float,
        default=1e-8,
        help="Relative GMRES tolerance for --linear-solver matrix_free_gmres.",
    )
    parser.add_argument(
        "--gmres-max-iterations",
        type=int,
        default=80,
        help="Maximum GMRES iterations per Newton step.",
    )
    parser.add_argument(
        "--gmres-restart",
        type=int,
        default=None,
        help="Restart length for matrix-free GMRES. Defaults to gmres max iterations.",
    )
    parser.add_argument(
        "--gmres-scaling",
        choices=("none", "row", "variable", "block"),
        default="none",
        help="Diagonal scaling for matrix-free GMRES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/figures_spe1"),
        help="Directory for PNG plots when --plot is used.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot windows instead of only writing PNGs.",
    )
    parser.add_argument(
        "--toolbar",
        action="store_true",
        help="Open PyMRST plotToolbar for interactive reservoir-state browsing.",
    )
    parser.add_argument(
        "--browser", action="store_true",
        help="Write the interactive standalone HTML reservoir viewer.",
    )
    parser.add_argument(
        "--open-browser", action="store_true",
        help="Open the HTML viewer after writing it (implies --browser).",
    )
    parser.add_argument(
        "--browser-output", type=Path,
        default=Path("examples/results_spe1/spe1.browser.html"),
        help="Output filename for the browser viewer.",
    )
    parser.add_argument(
        "--browser-style", choices=("mesh", "points"), default="mesh",
        help="Render solid cell faces or centroid points in the web viewer.",
    )
    parser.add_argument(
        "--zscale", type=float, default=1.0,
        help="Vertical exaggeration in the web viewer.",
    )
    parser.add_argument(
        "--save-states",
        nargs="?",
        const=Path("examples/results_spe1/spe1_states.npz"),
        type=Path,
        default=None,
        help=(
            "Save MRST-like state histories to a compressed .npz file. "
            "If no path is given, writes examples/results_spe1/spe1_states.npz."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress MRST-style timestep progress messages.",
    )
    args = parser.parse_args()

    if args.backend == "torch":
        result = run_backend_blackoil_tutorial_spe1(
            nsteps=args.steps,
            device=args.device,
            linear_solver=args.linear_solver,
            jacobian_drop_tolerance=args.jacobian_drop_tolerance,
            gmres_tolerance=args.gmres_tolerance,
            gmres_max_iterations=args.gmres_max_iterations,
            gmres_restart=args.gmres_restart,
            gmres_scaling=args.gmres_scaling,
            verbose=not args.quiet,
        )
    else:
        result = run_blackoil_tutorial_spe1(nsteps=args.steps, verbose=not args.quiet)
    print_tutorial_summary(result)
    if args.save_states is not None:
        saved = save_tutorial_states(result, args.save_states)
        print(f"  saved states: {saved}")
    if args.plot:
        outputs = plot_blackoil_tutorial_spe1(
            result,
            output_dir=None if args.show else args.output_dir,
            show=args.show,
        )
        if not args.show:
            print("  plots:")
            for key, path in outputs.items():
                print(f"    {key}: {path}")
    if args.toolbar:
        plot_toolbar_spe1(result, show=True)
    if args.browser or args.open_browser:
        browser_path = plot_browser_spe1(
            result,
            filename=args.browser_output,
            style=args.browser_style,
            open_browser=args.open_browser,
            zscale=args.zscale,
        )
        print(f"  browser viewer: {browser_path}")


#: MRST's own spelling. MRST files each example as a script; this port
#: gives it a `run_` entry point, and the alias keeps the MATLAB name
#: callable - the convention the rest of the package follows.
blackoilTutorialSPE1 = run_blackoil_tutorial_spe1

__all__ = [
    "DEFAULT_DECK",
    "SPE1TutorialResult",
    "producer_gor",
    "plot_blackoil_tutorial_spe1",
    "plot_permeability_and_wells",
    "plot_state_snapshot",
    "plot_toolbar_fields",
    "plot_toolbar_spe1",
    "plot_browser_spe1",
    "plot_well_response_curves",
    "run_blackoil_tutorial_spe1",
    "run_backend_blackoil_tutorial_spe1",
    "save_tutorial_states",
    "setup_spe1_tutorial",
    "tutorial_summary",
]
