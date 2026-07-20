"""Port of MRST `compositionalLargerProblemTutorial.m`.

The MRST tutorial builds a large 3D SPE5 problem and compares AD backends on a
single short timestep. PyMRST keeps the same switches (`useBC`,
`includeWater`, `useNatural`) and backend matrix. The default grid is scaled
for tests; pass `full_size=True` for MRST's 30-by-30-by-30 dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from unittest.mock import patch

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.backends import DiagonalAutoDiffBackend, SparseAutoDiffBackend
from pymrst.modules.compositional.fluids import get_benchmark_mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.params.rock import log_norm_layers, make_rock, pore_volume
from pymrst.params.wells_and_bc import fluxside, pside, vertical_well
from pymrst.utils.units import DAY, DARCY, MILLI, YEAR


@dataclass
class LargerProblemTiming:
    """One backend row from the larger-problem tutorial."""

    backend: str
    available: bool
    converged: bool | None
    iterations: int | None
    assembly_time: float | None
    linear_solve_time: float | None
    total_time: float | None
    assembly_per_iteration: float | None
    linear_solve_per_iteration: float | None
    total_per_iteration: float | None
    reason: str = ""

    @property
    def elapsed(self) -> float | None:
        """Backward-compatible alias for older callers."""
        return self.total_time


def larger_problem_backends():
    """MRST backend variants from the larger-problem tutorial."""
    return [
        SparseAutoDiffBackend(),
        DiagonalAutoDiffBackend(name="Diagonal", use_mex=False),
        DiagonalAutoDiffBackend(name="Diagonal (MEX+RowMajor)", use_mex=True, row_major=True),
    ]


def larger_problem_tick_labels(has_mex_row: bool) -> tuple[str, ...]:
    """X-axis labels used by MRST's timing plot."""
    if has_mex_row:
        return ("Sparse", "Diagonal", "Diagonal with Mex", "Sparse+Direct solver")
    return ("Sparse", "Diagonal", "Sparse+Direct solver")


def make_larger_problem_case(
    dims: tuple[int, int, int] = (4, 4, 3),
    *,
    full_size: bool = False,
    use_bc: bool = False,
    include_water: bool = False,
    use_natural: bool = True,
    backend=None,
):
    """Build MRST's larger 3D SPE5 compositional benchmark case."""
    if full_size:
        dims = (30, 30, 30)
    dims = tuple(int(x) for x in dims)
    grid = compute_geometry(cart_grid(dims, [1000.0, 1000.0, 100.0]))
    perm_md = log_norm_layers(dims, rng=np.random.default_rng(0))
    rock = make_rock(grid, perm_md[:, None] * MILLI * DARCY, 0.2)
    pv = pore_volume(grid, rock)
    mixture, info = get_benchmark_mixture("spe5")
    min_pressure = 0.5 * info.pressure
    total_time = 10.0 * YEAR

    bc = None
    wells = []
    if use_bc:
        sat = np.array([0.0, 1.0, 0.0]) if include_water else np.array([1.0, 0.0])
        bc = fluxside(None, grid, "xmin", pv.sum() / total_time, sat=sat)
        bc = pside(bc, grid, "xmax", min_pressure, sat=sat)
        bc.components = np.tile(info.injection, (bc.face.size, 1))
    else:
        if include_water:
            inj_compi = np.array([0.0, 1.0, 0.0])
            prod_compi = np.array([0.0, 0.5, 0.5])
        else:
            inj_compi = np.array([1.0, 0.0])
            prod_compi = np.array([0.5, 0.5])
        wells = vertical_well(
            None,
            grid,
            rock,
            0,
            0,
            [0],
            well_type="rate",
            name="Inj",
            val=float(pv.sum() / total_time),
            sign=1,
            compi=inj_compi,
        )
        wells[0].components = info.injection
        wells = vertical_well(
            wells,
            grid,
            rock,
            dims[0] - 1,
            dims[1] - 1,
            [dims[2] - 1],
            well_type="bhp",
            name="Prod",
            val=float(min_pressure),
            sign=-1,
            compi=prod_compi,
        )
        wells[1].components = info.injection

    backend = SparseAutoDiffBackend() if backend is None else backend
    model_cls = NaturalVariablesCompositionalModel if use_natural else OverallCompositionModel
    model = model_cls(
        grid,
        rock,
        mixture,
        temperature=info.T,
        wells=wells,
        bc=bc,
        water=include_water,
        rho_s=(500.0, 500.0),
        water_rho_s=1000.0,
        n_kr=(2.0, 2.0, 2.0) if include_water else (2.0, 2.0),
        auto_diff_backend=backend,
        max_iterations=60,
    )
    if include_water:
        state0 = model.initial_state(info.pressure, info.initial, sW=0.2)
    else:
        state0 = model.initial_state(info.pressure, info.initial)
    return {
        "grid": grid,
        "rock": rock,
        "pv": pv,
        "mixture": mixture,
        "info": info,
        "bc": bc,
        "wells": wells,
        "model": model,
        "initial_state": state0,
        "min_pressure": min_pressure,
        "total_time": total_time,
        "dims": dims,
    }


def run_compositional_larger_problem_tutorial(
    dims: tuple[int, int, int] = (4, 4, 3),
    *,
    full_size: bool = False,
    use_bc: bool = False,
    include_water: bool = False,
    use_natural: bool = True,
    dt: float = 0.1 * DAY,
    run_unavailable: bool = False,
):
    """Run one short larger-problem step for available backend variants."""
    timings: list[LargerProblemTiming] = []
    reference_case = None
    for backend in larger_problem_backends():
        case = make_larger_problem_case(
            dims,
            full_size=full_size,
            use_bc=use_bc,
            include_water=include_water,
            use_natural=use_natural,
            backend=backend,
        )
        if reference_case is None:
            reference_case = case
        if not backend.available and not run_unavailable:
            timings.append(
                LargerProblemTiming(
                    backend=backend.label(),
                    available=False,
                    converged=None,
                    iterations=None,
                    assembly_time=None,
                    linear_solve_time=None,
                    total_time=None,
                    assembly_per_iteration=None,
                    linear_solve_per_iteration=None,
                    total_per_iteration=None,
                    reason=backend.reason,
                )
            )
            continue
        ok, iterations, total, linear = _timed_larger_problem_step(case, float(dt))
        assembly = max(total - linear, 0.0)
        denom = max(iterations, 1)
        timings.append(
            LargerProblemTiming(
                backend=backend.label(),
                available=backend.available,
                converged=ok,
                iterations=iterations,
                assembly_time=assembly,
                linear_solve_time=linear,
                total_time=total,
                assembly_per_iteration=assembly / denom,
                linear_solve_per_iteration=linear / denom,
                total_per_iteration=total / denom,
                reason=backend.reason if not backend.available else "",
            )
        )
    assert reference_case is not None
    direct_case = make_larger_problem_case(
        dims,
        full_size=full_size,
        use_bc=use_bc,
        include_water=include_water,
        use_natural=use_natural,
        backend=SparseAutoDiffBackend(),
    )
    ok, iterations, total, linear = _timed_larger_problem_step(direct_case, float(dt))
    assembly = max(total - linear, 0.0)
    denom = max(iterations, 1)
    timings.append(
        LargerProblemTiming(
            backend="Sparse+Direct solver",
            available=True,
            converged=ok,
            iterations=iterations,
            assembly_time=assembly,
            linear_solve_time=linear,
            total_time=total,
            assembly_per_iteration=assembly / denom,
            linear_solve_per_iteration=linear / denom,
            total_per_iteration=total / denom,
        )
    )
    reference_case["timings"] = timings
    reference_case["time_matrix"] = larger_problem_time_matrix(timings)
    reference_case["tick_labels"] = larger_problem_tick_labels(
        any(row.backend == "Diagonal (MEX+RowMajor)" and row.available for row in timings)
    )
    reference_case["plot_sections"] = larger_problem_plot_sections(timings, use_natural=use_natural)
    return reference_case


def _timed_larger_problem_step(case: dict, dt: float) -> tuple[bool, int, float, float]:
    """Run one step while separating total and sparse linear-solve time."""
    linear_time = 0.0
    original_spsolve = spla.spsolve

    def timed_spsolve(*args, **kwargs):
        nonlocal linear_time
        tic = time.perf_counter()
        out = original_spsolve(*args, **kwargs)
        linear_time += time.perf_counter() - tic
        return out

    tic = time.perf_counter()
    with patch("scipy.sparse.linalg.spsolve", side_effect=timed_spsolve):
        _, ok, iterations = case["model"].step(case["initial_state"], dt, bc=case["bc"])
    total = time.perf_counter() - tic
    return ok, iterations, total, linear_time


def larger_problem_time_matrix(rows: list[LargerProblemTiming]) -> np.ndarray:
    """Return MRST's bar-plot table: assembly, linear solver, total."""
    available = [row for row in rows if row.available and row.total_time is not None]
    return np.asarray(
        [
            [row.assembly_time, row.linear_solve_time, row.total_time]
            for row in available
        ],
        dtype=float,
    )


def larger_problem_plot_sections(rows: list[LargerProblemTiming], *, use_natural: bool = True) -> dict:
    """Return the two plotting sections at the end of the MRST tutorial.

    MRST first plots assembly, linear-solver, and total time for each available
    backend row, then zooms the y-axis to the assembly-time column. This helper
    carries those plotting arrays and labels without requiring a plotting
    backend.
    """
    has_mex_row = any(row.backend == "Diagonal (MEX+RowMajor)" and row.available for row in rows)
    time = larger_problem_time_matrix(rows)
    assembly_time = time[:, 0] if time.size else np.asarray([], dtype=float)
    zoom_max = float(1.2 * np.max(assembly_time)) if assembly_time.size else 0.0
    return {
        "timing_bar": {
            "figure": 1 + int(bool(use_natural)),
            "time": time,
            "x_tick_labels": larger_problem_tick_labels(has_mex_row),
            "legend": ("Equation assembly", "Linear solver", "Total time"),
            "legend_location": "NorthWest",
        },
        "assembly_zoom": {
            "assembly_time": assembly_time,
            "ylim": (0.0, zoom_max),
        },
    }


if __name__ == "__main__":
    def test_larger_problem_case_matches_mrst_well_setup():
        out = make_larger_problem_case((3, 3, 2), use_bc=False, include_water=False, use_natural=True)
        assert out["dims"] == (3, 3, 2)
        assert out["mixture"].n == 6
        assert len(out["wells"]) == 2
        assert out["wells"][0].name == "Inj"
        assert out["wells"][1].name == "Prod"
        assert np.isclose(out["wells"][0].val, out["pv"].sum() / out["total_time"])
        print("ok test_larger_problem_case_matches_mrst_well_setup")

    def test_larger_problem_case_matches_mrst_bc_setup():
        out = make_larger_problem_case((3, 3, 2), use_bc=True, include_water=False, use_natural=False)
        assert out["bc"] is not None
        assert len(out["wells"]) == 0
        assert out["bc"].components.shape == (out["bc"].face.size, out["mixture"].n)
        assert np.isclose(out["min_pressure"], 0.5 * out["info"].pressure)
        print("ok test_larger_problem_case_matches_mrst_bc_setup")

    def test_larger_problem_reports_backend_rows():
        out = run_compositional_larger_problem_tutorial(
            dims=(2, 2, 2),
            use_bc=False,
            include_water=False,
            use_natural=False,
            dt=0.01 * DAY,
        )
        labels = [row.backend for row in out["timings"]]
        assert labels == ["Sparse", "Diagonal", "Diagonal (MEX+RowMajor)", "Sparse+Direct solver"]
        assert out["timings"][0].available
        assert out["timings"][0].elapsed is not None
        assert out["time_matrix"].shape[1] == 3
        assert out["tick_labels"] in {
            ("Sparse", "Diagonal", "Sparse+Direct solver"),
            ("Sparse", "Diagonal", "Diagonal with Mex", "Sparse+Direct solver"),
        }
        assert out["plot_sections"]["timing_bar"]["legend"] == (
            "Equation assembly", "Linear solver", "Total time"
        )
        assert out["plot_sections"]["timing_bar"]["figure"] == 1
        assert out["plot_sections"]["assembly_zoom"]["ylim"][1] >= 0.0
        print("ok test_larger_problem_reports_backend_rows")

    test_larger_problem_case_matches_mrst_well_setup()
    test_larger_problem_case_matches_mrst_bc_setup()
    test_larger_problem_reports_backend_rows()
    print("\nAll tests passed!")


__all__ = [
    "LargerProblemTiming",
    "larger_problem_backends",
    "larger_problem_plot_sections",
    "larger_problem_tick_labels",
    "larger_problem_time_matrix",
    "make_larger_problem_case",
    "run_compositional_larger_problem_tutorial",
]
