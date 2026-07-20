"""Port of MRST `book-ii/compositionalPerformanceExample.m`.

The MRST example compares automatic-differentiation backends on a large SPE5
compositional problem. PyMRST now runs both sparse and pure-Python diagonal
backends through the nonlinear compositional step. If Numba is installed, the
MEX-labelled rows use compiled row-major diagonal kernels. Diagonal Jacobian
blocks are converted to sparse only at the linear-solve handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from unittest.mock import patch

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.backends import DiagonalAutoDiffBackend, SparseAutoDiffBackend
from pymrst.modules.compositional.fluids import get_compositional_fluid_case
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import log_norm_layers, make_rock, pore_volume
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, DAY, DARCY, MILLI, YEAR


@dataclass
class PerformanceTiming:
    """Timing row analogous to MRST's report timing table."""

    backend: str
    available: bool
    cells: int
    degrees_of_freedom: int
    assembly_time: float | None
    linear_solve_time: float | None
    total_time: float | None
    assembly_per_iteration: float | None
    linear_solve_per_iteration: float | None
    total_per_iteration: float | None
    iterations: int | None
    converged: bool | None
    reason: str = ""


def _corner_index(ix: int, iy: int, iz: int, dims: tuple[int, int, int]) -> int:
    """Zero-based Cartesian cell index in MRST/PyMRST column-major ordering."""
    nx, ny, _ = dims
    return ix + iy * nx + iz * nx * ny


def make_performance_case(
    dims: tuple[int, int, int] = (6, 6, 4),
    *,
    backend=None,
) -> tuple:
    """Build the SPE5 injector-producer benchmark case.

    MRST's published example uses `(50, 50, 50)`. The default here is scaled so
    the module self-test remains quick, while callers can pass larger `dims`.
    """
    dims = tuple(int(x) for x in dims)
    backend = SparseAutoDiffBackend() if backend is None else backend
    pdims = [1000.0, 1000.0, 100.0]
    grid = compute_geometry(cart_grid(dims, pdims))
    rng = np.random.default_rng(0)
    perm_md = log_norm_layers(dims, rng=rng)
    rock = make_rock(grid, perm_md[:, None] * MILLI * DARCY, 0.2)
    pv = pore_volume(grid, rock)

    mixture, info = get_compositional_fluid_case("spe5")
    min_p = 0.5 * info.pressure
    total_time = 10.0 * YEAR
    inj_cell = _corner_index(0, 0, 0, dims)
    prod_cell = _corner_index(dims[0] - 1, dims[1] - 1, dims[2] - 1, dims)

    wells = add_well(
        None,
        grid,
        rock,
        np.array([inj_cell]),
        well_type="rate",
        val=float(pv.sum() / total_time),
        compi=np.array([1.0, 0.0]),
        sign=1,
        name="Inj",
    )
    wells[0].components = info.injection
    wells = add_well(
        wells,
        grid,
        rock,
        np.array([prod_cell]),
        well_type="bhp",
        val=float(min_p),
        compi=np.array([0.5, 0.5]),
        sign=-1,
        name="Prod",
    )
    wells[1].components = info.injection

    model = OverallCompositionModel(
        grid,
        rock,
        mixture,
        temperature=info.T,
        wells=wells,
        rho_s=(500.0, 500.0),
        water=False,
        n_kr=(2.0, 2.0),
        auto_diff_backend=backend,
    )
    state0 = model.initial_state(info.pressure, info.initial)
    return grid, rock, mixture, info, wells, model, state0


def performance_backends():
    """Return the MRST backend variants from the performance example."""
    return [
        SparseAutoDiffBackend(),
        DiagonalAutoDiffBackend(),
        DiagonalAutoDiffBackend(name="Diagonal (RowMajor)", use_mex=False, row_major=True),
        DiagonalAutoDiffBackend(name="Diagonal (MEX)", use_mex=True),
        DiagonalAutoDiffBackend(
            name="Diagonal (MEX+RowMajor)",
            use_mex=True,
            row_major=True,
            deferred_assembly=True,
        ),
    ]


def mrst_performance_plot_table(rows: list[PerformanceTiming]):
    """Return the four-row timing matrix plotted by MRST's example.

    PyMRST keeps an extra pure-Python ``Diagonal (RowMajor)`` row because it is
    useful without compiled kernels. The MATLAB script plots only Sparse,
    Diagonal, Diagonal (MEX), and Diagonal (MEX+RowMajor), with columns
    ``Assembly / each``, ``Linear solve / it``, and ``Total / it``.
    Unavailable backend timings are represented by NaN.
    """
    by_label = {row.backend: row for row in rows}
    labels = ["Sparse", "Diagonal", "Diagonal (MEX)", "Diagonal (MEX+RowMajor)"]
    time_matrix = np.full((len(labels), 3), np.nan, dtype=float)
    available = np.zeros(len(labels), dtype=bool)
    reasons = []
    for i, label in enumerate(labels):
        row = by_label[label]
        available[i] = bool(row.available)
        reasons.append(row.reason)
        if row.available:
            time_matrix[i, :] = [
                row.assembly_per_iteration,
                row.linear_solve_per_iteration,
                row.total_per_iteration,
            ]
    return {
        "labels": labels,
        "legend": ["Assembly / each", "Linear solve / it", "Total / it"],
        "ylabel": "Time [s]",
        "time": time_matrix,
        "available": available,
        "reasons": reasons,
    }


def _timed_step(model: OverallCompositionModel, state0, dt: float) -> tuple[bool, int, float, float]:
    """Run one step while measuring total and sparse linear-solve time."""
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
        _, ok, iterations = model.step(state0, dt)
    total = time.perf_counter() - tic
    return ok, iterations, total, linear_time


def run_compositional_performance_example(
    dims: tuple[int, int, int] = (6, 6, 4),
    *,
    run_unavailable: bool = False,
    dt: float = 0.1 * DAY,
) -> list[PerformanceTiming]:
    """Benchmark the available PyMRST backend and report MRST backend gaps."""
    rows: list[PerformanceTiming] = []
    for backend in performance_backends():
        grid, _, _, _, _, model, state0 = make_performance_case(dims, backend=backend)
        dof = grid.cells.num * model.mix.n + 3 * len(model.wells)
        if not backend.available and not run_unavailable:
            rows.append(
                PerformanceTiming(
                    backend=backend.label(),
                    available=False,
                    cells=grid.cells.num,
                    degrees_of_freedom=dof,
                    assembly_time=None,
                    linear_solve_time=None,
                    total_time=None,
                    assembly_per_iteration=None,
                    linear_solve_per_iteration=None,
                    total_per_iteration=None,
                    iterations=None,
                    converged=None,
                    reason=backend.reason,
                )
            )
            continue

        model.max_iterations = 60
        ok, iterations, total, linear = _timed_step(model, state0, float(dt))
        assembly = max(total - linear, 0.0)
        denom = max(iterations, 1)
        rows.append(
            PerformanceTiming(
                backend=backend.label(),
                available=backend.available,
                cells=grid.cells.num,
                degrees_of_freedom=dof,
                assembly_time=assembly,
                linear_solve_time=linear,
                total_time=total,
                assembly_per_iteration=assembly / denom,
                linear_solve_per_iteration=linear / denom,
                total_per_iteration=total / denom,
                iterations=iterations,
                converged=ok,
                reason=backend.reason if not backend.available else "",
            )
        )
    return rows


if __name__ == "__main__":
    def test_performance_example_reports_mrst_backend_matrix():
        out = run_compositional_performance_example((3, 3, 2))
        labels = [row.backend for row in out]
        assert labels == [
            "Sparse",
            "Diagonal",
            "Diagonal (RowMajor)",
            "Diagonal (MEX)",
            "Diagonal (MEX+RowMajor)",
        ]
        assert out[0].available and out[0].total_time is not None
        assert out[0].degrees_of_freedom == 3 * 3 * 2 * 6 + 6
        assert out[1].available and out[1].total_time is not None
        assert out[1].converged is True
        assert out[2].available and out[2].converged is True
        if out[3].available:
            assert out[3].converged is True
            assert out[4].converged is True
        else:
            assert out[3].reason
        table = mrst_performance_plot_table(out)
        assert table["labels"] == ["Sparse", "Diagonal", "Diagonal (MEX)", "Diagonal (MEX+RowMajor)"]
        assert table["legend"] == ["Assembly / each", "Linear solve / it", "Total / it"]
        assert table["time"].shape == (4, 3)
        assert np.all(np.isfinite(table["time"][table["available"]]))
        print("ok test_performance_example_reports_mrst_backend_matrix")

    def test_performance_case_matches_spe5_setup():
        grid, _, mixture, info, wells, model, state0 = make_performance_case((3, 3, 2))
        assert grid.cells.num == 18
        assert mixture.names == ("C1", "C3", "C6", "C10", "C15", "C20")
        assert len(wells) == 2
        assert np.allclose(wells[0].components, info.injection)
        assert state0.z.shape == (18, 6)
        assert model.auto_diff_backend.label() == "Sparse"
        print("ok test_performance_case_matches_spe5_setup")

    def test_diagonal_backend_matches_sparse_step():
        sparse_case = make_performance_case((3, 3, 2), backend=SparseAutoDiffBackend())
        diagonal_case = make_performance_case((3, 3, 2), backend=DiagonalAutoDiffBackend())
        row_major_case = make_performance_case(
            (3, 3, 2),
            backend=DiagonalAutoDiffBackend(name="Diagonal (RowMajor)", use_mex=False, row_major=True),
        )
        sparse_model, sparse_state = sparse_case[-2], sparse_case[-1]
        diagonal_model, diagonal_state = diagonal_case[-2], diagonal_case[-1]
        row_major_model, row_major_state = row_major_case[-2], row_major_case[-1]
        sparse_model.max_iterations = diagonal_model.max_iterations = row_major_model.max_iterations = 8
        st_s, ok_s, it_s = sparse_model.step(sparse_state, 0.1 * DAY)
        st_d, ok_d, it_d = diagonal_model.step(diagonal_state, 0.1 * DAY)
        st_r, ok_r, it_r = row_major_model.step(row_major_state, 0.1 * DAY)
        assert ok_s and ok_d and ok_r
        assert it_s == it_d == it_r
        for st_other in (st_d, st_r):
            assert np.allclose(st_other.pressure, st_s.pressure, rtol=1.0e-9, atol=1.0e-5)
            assert np.allclose(st_other.z, st_s.z, rtol=1.0e-9, atol=1.0e-10)
            assert np.allclose(st_other.qOs, st_s.qOs, rtol=1.0e-9, atol=1.0e-10)
            assert np.allclose(st_other.qGs, st_s.qGs, rtol=1.0e-9, atol=1.0e-10)
        print("ok test_diagonal_backend_matches_sparse_step")

    test_performance_example_reports_mrst_backend_matrix()
    test_performance_case_matches_spe5_setup()
    test_diagonal_backend_matches_sparse_step()
    print("\nAll tests passed!")


__all__ = [
    "PerformanceTiming",
    "make_performance_case",
    "mrst_performance_plot_table",
    "performance_backends",
    "run_compositional_performance_example",
]
