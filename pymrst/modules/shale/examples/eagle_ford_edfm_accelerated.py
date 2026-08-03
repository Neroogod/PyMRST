"""Native compiled-diagonal EDFM workflow (MRST ``eagleFordEDFM_MEX.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_core.backends.diagonal import DiagonalAutoDiffBackend
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.examples._shared import run_schedule, simple_schedule
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.modules.compositional.utils.init_compositional_state import init_compositional_state
from pymrst.modules.shale.examples.eagle_ford import (
    EagleFordCase,
    _merge_nnc,
    _natural_fractures,
)
from pymrst.modules.shale.examples.eagle_ford_edfm import _completion_side_lengths
from pymrst.modules.shale.hfm_fns import (
    edfm_shale_grid,
    fracture_fracture_shale_nncs_3d,
    setup_shale_edfm_operators_tpfa,
)
from pymrst.modules.shale.other_mrst_fns import (
    add_well_shale,
    make_shale_rock,
    well_fracture_nncs_3d,
)
from pymrst.modules.shale.utils import (
    create_multistage_hfs,
    get_shale_comp_fluid_case,
    mesh_hf_system,
)
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import DARCY, DAY, FT, MILLI, NANO, PSIA, YEAR


def eagle_ford_edfm_accelerated_backend():
    """Return the compiled row-major backend corresponding to MRST MEX AD."""
    return DiagonalAutoDiffBackend(
        name="Eagle Ford EDFM compiled diagonal",
        use_mex=True,
        row_major=True,
    )


def setup_eagle_ford_edfm_accelerated(
    *,
    natural_fracture_limit: int | None = None,
    build_edfm: bool = False,
    auto_diff_backend=None,
) -> EagleFordCase:
    """Build the complete short-field accelerated MRST EDFM case."""
    tolerance = 1e-5
    physical = np.array([150.0, 300.0, 38.1])
    fracture_half_length = 180.0
    fracture_height = 75.0 * FT
    fracture_spacing = 150.0
    fractures_per_stage = 3
    number_of_stages = 1
    cluster_spacing = 10.0
    fracture_aperture = 3.0 * MILLI
    fracture_porosity = 0.5
    fracture_permeability = DARCY
    matrix_porosity = 0.12
    matrix_permeability = 900.0 * NANO * DARCY
    well_length = 150.0
    well_radius = 0.328084 * FT
    heel = np.array([0.0, 0.0, physical[2] - tolerance])
    toe = np.array([well_length, 0.0, physical[2] - tolerance])
    bottom_hole_pressure = 1000.0 * PSIA
    first_centroid = np.array([
        65.0, fracture_half_length / 2.0,
        physical[2] - fracture_height / 2.0,
    ])

    hydraulic, centroids = create_multistage_hfs(
        num_stages=number_of_stages,
        frac_spacing=fracture_spacing,
        num_fracs_per_stage=fractures_per_stage,
        frac_half_length=fracture_half_length,
        frac_height=fracture_height,
        cluster_spacing=cluster_spacing,
        frac_centroid1=first_centroid,
        is_stencil=True,
    )
    for index, plane in enumerate(hydraulic):
        plane.aperture = fracture_aperture
        plane.porosity = fracture_porosity
        plane.permeability = fracture_permeability
        plane.metadata.update(kind="hydraulic", source_index=index)
    matrix_grid = compute_geometry(mesh_hf_system(
        physical,
        centroids,
        num_stages=number_of_stages,
        num_fracs_per_stage=fractures_per_stage,
        ny=20,
        nz=5,
        nx_refine_small=4,
        nx_refine=9,
    ))
    matrix_rock = make_shale_rock(
        matrix_grid, matrix_permeability, matrix_porosity,
    )
    natural, rejected, areas = _natural_fractures(
        full_field=False, limit=natural_fracture_limit,
    )
    case = EagleFordCase(
        "edfm", False, matrix_grid, matrix_rock, tuple(hydraulic), natural,
        rejected, areas, np.vstack([heel, toe]), well_radius,
        bottom_hole_pressure,
    )
    if not build_edfm:
        return case

    fracture_planes = case.fracture_planes
    global_grid, processed = edfm_shale_grid(
        matrix_grid, fracture_planes, matrix_rock, tolerance=tolerance,
    )
    fracture_nnc = fracture_fracture_shale_nncs_3d(
        global_grid.edfm,
        global_grid.fractures,
        global_grid.projections,
        aperture=[plane.aperture for plane in fracture_planes],
        tolerance=tolerance,
    )
    global_grid.nnc = _merge_nnc(global_grid.nnc, fracture_nnc)
    connections = well_fracture_nncs_3d(
        [case.well_path],
        global_grid.fractures,
        global_grid.projections,
        matrix_cell_count=matrix_grid.cells.num,
        tolerance=tolerance,
    )
    half_transmissibility = get_face_transmissibility(
        global_grid.grid, global_grid.rock,
    )
    operators = setup_shale_edfm_operators_tpfa(
        global_grid.grid,
        global_grid.rock,
        half_transmissibility,
        global_grid.nnc,
        cell_volumes=global_grid.grid.cells.volumes,
    )

    mixture, info = get_shale_comp_fluid_case("eagleford")
    wells = []
    if connections.fracture_cells.size:
        completions = np.column_stack([
            connections.fracture_cells, connections.fracture_index,
        ])
        wells = add_well_shale(
            wells,
            global_grid,
            completions,
            fracture_aperture=[plane.aperture for plane in fracture_planes],
            fracture_cell_side_lengths=_completion_side_lengths(
                global_grid, completions,
            ),
            radius=well_radius,
            direction="x",
            well_type="bhp",
            val=bottom_hole_pressure,
            compi=[0.17, 0.74, 0.09],
            sign=-1,
            name="Prod1",
            hydraulic_plane_count=len(hydraulic),
            components=info.initial,
        )

    backend = (
        eagle_ford_edfm_accelerated_backend()
        if auto_diff_backend is None else auto_diff_backend
    )
    model = NaturalVariablesCompositionalModel(
        global_grid.grid,
        global_grid.rock,
        mixture,
        temperature=info.T,
        wells=wells,
        water=True,
        n_kr=(2, 2, 2),
        max_iterations=40,
        auto_diff_backend=backend,
    )
    model.ops = operators
    model.pv = operators.pv
    model.gdz = np.zeros(operators.N.shape[0])
    initial_state = init_compositional_state(
        model,
        info.pressure,
        info.T,
        np.tile([0.17, 0.74, 0.09], (global_grid.n_cells, 1)),
        info.initial,
    )
    case.global_grid = global_grid
    case.processed_planes = tuple(processed)
    case.well_connections = connections
    case.operators = operators
    case.wells = wells
    case.mixture = mixture
    case.model = model
    case.initial_state = initial_state
    case.timesteps = rampup_timesteps(15 * YEAR, 30 * DAY, 15)
    return case


def run_eagle_ford_edfm_accelerated(
    *, nstep=None, timesteps=None, max_cuts=8, verbose=False, **kwargs,
) -> EagleFordCase:
    """Preprocess and simulate the compiled-diagonal EDFM case."""
    case = setup_eagle_ford_edfm_accelerated(build_edfm=True, **kwargs)
    steps = case.timesteps if timesteps is None else np.asarray(timesteps, float)
    if nstep is not None:
        steps = steps[:int(nstep)]
    case.timesteps = steps
    case.result = run_schedule(
        case.model,
        case.initial_state,
        simple_schedule(steps, wells=case.wells),
        max_cuts=max_cuts,
        verbose=verbose,
    )
    return case


eagleFordEDFM_MEX = run_eagle_ford_edfm_accelerated

__all__ = [
    "EagleFordCase", "eagle_ford_edfm_accelerated_backend",
    "setup_eagle_ford_edfm_accelerated",
    "run_eagle_ford_edfm_accelerated", "eagleFordEDFM_MEX",
]
