"""Shared native solver setup for SPE-201243-PA Figures 10--13."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.water_model import WaterModelAD
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.flow_discretization import FlowDiscretization
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.modules.shale.data import FracturePlane, ShaleNNC
from pymrst.modules.shale.hfm_fns import (
    edfm_shale_grid,
    fracture_fracture_shale_nncs_3d,
    setup_shale_edfm_operators_tpfa,
)
from pymrst.modules.shale.other_mrst_fns import make_shale_rock
from pymrst.modules.shale.pedfm_nnc import (
    projected_matrix_fracture_nncs_3d,
    setup_pedfm_operators_tpfa,
)
from pymrst.params.wells_and_bc import pside
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, POISE


@dataclass
class TeneCrossVariant:
    name: str
    grid: object
    rock: object
    model: WaterModelAD
    boundary: object
    initial_state: object
    states: list
    reports: list


@dataclass
class TeneCrossCase:
    fracture_permeability: float
    matrix_grid: object
    fracture_planes: tuple[FracturePlane, ...]
    global_grid: object
    pedfm: TeneCrossVariant
    edfm: TeneCrossVariant
    explicit: TeneCrossVariant
    explicit_fracture_cells: np.ndarray
    timesteps: np.ndarray


def _merge_nnc(first: ShaleNNC, second: ShaleNNC) -> ShaleNNC:
    return ShaleNNC(
        np.vstack([first.cells, second.cells]), np.r_[first.T, second.T],
        first.type + second.type, np.r_[first.area, second.area],
        np.vstack([first.normal, second.normal]),
        np.vstack([first.plane_point, second.plane_point]),
    )


def _water_model(grid, rock, operators, *, boundary_grid=None):
    fluid = init_simple_adi_fluid(
        mu=[CENTI*POISE, CENTI*POISE], rho=[1000., 1000.], n=[1., 1.],
        c=[1e-4/BARSA, 0.], p_ref=5*BARSA,
    )
    boundary_grid = grid if boundary_grid is None else boundary_grid
    bc = pside(None, boundary_grid, "left", 10*BARSA, sat=[1.])
    bc = pside(bc, boundary_grid, "right", 0., sat=[1.])
    # AD-core's boundary owner uses the modern lower-case alias while the
    # shale operators preserve MRST's ``T_all`` spelling.
    if not hasattr(operators, "t_all"):
        operators.t_all = operators.T_all
    model = WaterModelAD(grid, rock, fluid, bc=bc)
    model.operators = operators
    model.pv_ref = np.asarray(operators.pv, dtype=float)
    model.gdz = np.zeros(operators.N.shape[0])
    model.FlowDiscretization = FlowDiscretization(model)
    return model, bc, model.initial_state(5*BARSA)


def _variant(name, grid, rock, operators, *, boundary_grid=None):
    model, bc, state = _water_model(
        grid, rock, operators, boundary_grid=boundary_grid,
    )
    return TeneCrossVariant(name, grid, rock, model, bc, state, [], [])


def setup_tene_cross_case(*, high_conductivity: bool, reduced_explicit=False):
    """Build pEDFM, EDFM and resolved-grid versions of the Tene cross case."""
    physical = np.array([9., 9., 3.])
    axes = [np.linspace(0., length, count+1) for length, count in zip(physical, (27, 27, 3))]
    matrix = compute_geometry(tensor_grid(*axes))
    matrix_rock = make_shale_rock(matrix, DARCY, .3)
    kf = (1e3 if high_conductivity else 1e-8)*DARCY
    points = (
        np.array([[2,4.5,0], [7,4.5,0], [7,4.5,3], [2,4.5,3]], float),
        np.array([[4.5,2,0], [4.5,7,0], [4.5,7,3], [4.5,2,3]], float),
    )
    planes = tuple(
        FracturePlane(p, aperture=1/25, porosity=.3, permeability=kf)
        for p in points
    )
    global_grid, processed = edfm_shale_grid(matrix, planes, matrix_rock, tolerance=1e-5)
    ff = fracture_fracture_shale_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        aperture=[1/25, 1/25], tolerance=1e-5,
    )
    global_grid.nnc = _merge_nnc(global_grid.nnc, ff)
    half_t = get_face_transmissibility(global_grid.grid, global_grid.rock)
    edfm_ops = setup_shale_edfm_operators_tpfa(
        global_grid.grid, global_grid.rock, half_t, global_grid.nnc,
        cell_volumes=global_grid.grid.cells.volumes,
    )
    projected = projected_matrix_fracture_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        tolerance=1e-5,
    )
    pedfm_ops = setup_pedfm_operators_tpfa(
        global_grid.grid, global_grid.rock, half_t, global_grid.nnc,
        projected=projected, tolerance=1e-5,
        cell_volumes=global_grid.grid.cells.volumes,
    )
    pedfm = _variant(
        "pedfm", global_grid.grid, global_grid.rock, pedfm_ops,
        boundary_grid=matrix,
    )
    edfm = _variant(
        "edfm", global_grid.grid, global_grid.rock, edfm_ops,
        boundary_grid=matrix,
    )

    dims = (45, 45, 2) if reduced_explicit else (225, 225, 2)
    explicit_grid = compute_geometry(tensor_grid(*[
        np.linspace(0., length, count+1) for length, count in zip(physical, dims)
    ]))
    explicit_rock = make_shale_rock(explicit_grid, DARCY, .3)
    centers = explicit_grid.cells.centroids
    dx, dy = physical[0]/dims[0], physical[1]/dims[1]
    horizontal = (
        (np.abs(centers[:, 1]-4.5) <= dy/2 + 1e-12)
        & (centers[:, 0] >= 2) & (centers[:, 0] <= 7)
    )
    vertical = (
        (np.abs(centers[:, 0]-4.5) <= dx/2 + 1e-12)
        & (centers[:, 1] >= 2) & (centers[:, 1] <= 7)
    )
    fracture_cells = np.flatnonzero(horizontal | vertical)
    explicit_rock.perm[fracture_cells, :] = kf
    explicit_model = WaterModelAD(
        explicit_grid, explicit_rock,
        init_simple_adi_fluid(
            mu=[CENTI*POISE, CENTI*POISE], rho=[1000., 1000.], n=[1.,1.],
            c=[1e-4/BARSA, 0.], p_ref=5*BARSA,
        ),
    )
    explicit_bc = pside(None, explicit_grid, "left", 10*BARSA, sat=[1.])
    explicit_bc = pside(explicit_bc, explicit_grid, "right", 0., sat=[1.])
    explicit_model.bc = explicit_bc
    explicit = TeneCrossVariant(
        "explicit", explicit_grid, explicit_rock, explicit_model, explicit_bc,
        explicit_model.initial_state(5*BARSA), [], [],
    )
    return TeneCrossCase(
        kf, matrix, tuple(processed), global_grid, pedfm, edfm, explicit,
        fracture_cells, rampup_timesteps(60*DAY, 30*DAY, 10),
    )


def run_tene_cross_case(*, nstep=None, timesteps=None, **kwargs):
    """Run all three discretizations and return aligned state histories."""
    case = setup_tene_cross_case(**kwargs)
    steps = case.timesteps if timesteps is None else np.asarray(timesteps, float)
    steps = steps if nstep is None else steps[:int(nstep)]
    case.timesteps = steps
    solver = NonLinearSolver(max_iterations=20, max_timestep_cuts=8)
    for variant in (case.pedfm, case.edfm, case.explicit):
        current = variant.initial_state
        for dt in steps:
            current, report = solver.solve_timestep(variant.model, current, float(dt))
            if not report.converged:
                raise RuntimeError(f"{variant.name} failed to converge")
            variant.states.append(current)
            variant.reports.append(report)
    return case


__all__ = ["TeneCrossCase", "TeneCrossVariant", "setup_tene_cross_case", "run_tene_cross_case"]
