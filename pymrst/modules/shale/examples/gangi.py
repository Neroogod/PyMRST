"""Native with/without Gangi comparison (MRST ``examples/gangi.m``)."""

from __future__ import annotations

import copy

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.examples._shared import run_schedule, simple_schedule
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.modules.compositional.utils.init_compositional_state import init_compositional_state
from pymrst.modules.shale.compositional_fns.nat_vars_shale_model import NatVarsShaleModel
from pymrst.modules.shale.examples.sorption import MechanismComparison
from pymrst.modules.shale.hfm_fns import (
    edfm_shale_grid,
    setup_shale_edfm_operators_tpfa,
)
from pymrst.modules.shale.other_mrst_fns import (
    add_well_shale,
    make_shale_rock,
    well_fracture_nncs_3d,
)
from pymrst.modules.shale.utils import (
    create_multistage_hfs,
    gangi_fn,
    get_shale_comp_fluid_case,
)
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import DARCY, DAY, NANO, PSIA, YEAR


def setup_gangi_case(*, full_size=False) -> MechanismComparison:
    """Build MRST's fractured Barnett Gangi-on/off comparison."""
    if full_size:
        dimensions = (61, 11, 4)
        physical = np.array([1080.0, 250.0, 100.0])
        stages = 10
        first_centroid = np.array([80.0, 125.0, 50.0])
        fracture_spacing = 100.0
        fracture_half_length = 90.0
        fracture_height = 60.0
        total_time = 15 * YEAR
        nominal_step = 30 * DAY
        ramp_steps = 15
        producer_control = "bhp"
        producer_value = 500 * PSIA
        use_water = True
    else:
        dimensions = (6, 3, 2)
        physical = np.array([120.0, 60.0, 20.0])
        stages = 2
        first_centroid = np.array([20.0, 30.0, 10.0])
        fracture_spacing = 80.0
        fracture_half_length = 25.0
        fracture_height = 12.0
        total_time = 1000.0
        nominal_step = 500.0
        ramp_steps = 2
        producer_control = "rate"
        producer_value = -1e-8
        use_water = False

    matrix_permeability = 100.0 * NANO * DARCY if full_size else 1e-3 * DARCY
    matrix_porosity = 0.063
    fracture_aperture = 0.003
    fracture_porosity = 0.5
    fracture_permeability = DARCY
    confining_pressure = 10000.0 * PSIA
    biot_coefficient = 1.0
    maximum_closure_pressure = 14000.0 * PSIA
    roughness_exponent = 0.4
    reference_permeability = 2090.2259 * NANO * DARCY

    matrix_grid = compute_geometry(cart_grid(dimensions, physical))
    matrix_rock = make_shale_rock(
        matrix_grid,
        matrix_permeability,
        matrix_porosity,
        shale_mechanisms={"Gangi": True},
    )
    planes, _ = create_multistage_hfs(
        num_stages=stages,
        frac_spacing=fracture_spacing,
        num_fracs_per_stage=1,
        frac_half_length=fracture_half_length,
        frac_height=fracture_height,
        cluster_spacing=10.0,
        frac_centroid1=first_centroid,
    )
    for plane in planes:
        plane.aperture = fracture_aperture
        plane.porosity = fracture_porosity
        plane.permeability = fracture_permeability

    global_grid, processed = edfm_shale_grid(
        matrix_grid, planes, matrix_rock,
    )
    well_path = np.array([
        first_centroid,
        [physical[0] - first_centroid[0], first_centroid[1], first_centroid[2]],
    ])
    connections = well_fracture_nncs_3d(
        [well_path],
        global_grid.fractures,
        global_grid.projections,
        matrix_cell_count=global_grid.n_matrix,
    )
    completions = np.column_stack([
        connections.fracture_cells, connections.fracture_index,
    ])
    if completions.size == 0:
        raise RuntimeError("reference well did not intersect a fracture cell")
    cell_size = physical / np.asarray(dimensions)
    wells = add_well_shale(
        [],
        global_grid,
        completions,
        fracture_aperture=np.full(stages, fracture_aperture),
        fracture_cell_side_lengths=np.tile(
            cell_size[1:], (len(completions), 1),
        ),
        radius=0.1,
        direction="x",
        well_type=producer_control,
        val=producer_value,
        compi=([0.23, 0.76, 0.01] if use_water else [0.99, 0.01]),
        sign=-1,
        name="Prod1",
        components=[0.991, 0.0088, 0.0002],
    )

    mixture, info = get_shale_comp_fluid_case("barnett3comps")
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
    model_wells = copy.deepcopy(wells) if full_size else []
    # The class default tol_cnv=1e-3 (MRST's ReservoirModel default) is coarse
    # relative to the driving forces in a shale nanoDarcy step: it reads the
    # initial guess as already converged and returns it unchanged (0 Newton
    # iterations), which defeats this example's point of comparing enabled vs.
    # disabled mechanism physics over a real step. Ask for a tight solve.
    common = dict(
        temperature=info.T,
        wells=model_wells,
        water=use_water,
        n_kr=((1, 1, 1) if use_water else (1, 1)),
        max_iterations=35,
        gravity=0,
        tol_cnv=1e-9,
    )

    def permeability_multiplier(pressure):
        return gangi_fn(
            pressure,
            confining_pressure,
            biot_coefficient,
            maximum_closure_pressure,
            roughness_exponent,
            reference_permeability,
            matrix_permeability,
        )

    enabled = NatVarsShaleModel(
        global_grid.grid,
        global_grid.rock,
        mixture,
        shale_rock=global_grid.rock,
        grain_volume=operators.grain_volume,
        gangi_function=permeability_multiplier,
        **common,
    )
    disabled_rock = copy.deepcopy(global_grid.rock)
    disabled_rock.shale_mechanisms = {}
    disabled = NaturalVariablesCompositionalModel(
        global_grid.grid,
        disabled_rock,
        mixture,
        **{**common, "wells": copy.deepcopy(model_wells)},
    )
    for model in (enabled, disabled):
        model.ops = operators
        model.pv = operators.pv
        model.gdz = np.zeros(operators.N.shape[0])

    saturation = (
        np.tile([0.23, 0.70, 0.07], (global_grid.n_cells, 1))
        if use_water else None
    )
    pressure = info.pressure
    if not full_size:
        x_coordinate = np.asarray(global_grid.grid.cells.centroids[:, 0], float)
        scale = (x_coordinate - x_coordinate.min()) / max(np.ptp(x_coordinate), 1.0)
        pressure = info.pressure - scale * 250 * PSIA
    initial_enabled = init_compositional_state(
        enabled, pressure, info.T, saturation, info.initial,
    )
    initial_disabled = init_compositional_state(
        disabled, pressure, info.T, saturation, info.initial,
    )
    timesteps = rampup_timesteps(total_time, nominal_step, ramp_steps)
    multiplier = np.asarray(enabled._gangi_multiplier(np.asarray(pressure)))
    diagnostics = {
        "matrix_cells": int(global_grid.n_matrix),
        "fracture_cells": int(global_grid.n_fracture),
        "nnc_count": int(global_grid.nnc.cells.shape[0]),
        "well_intersections": int(connections.points.shape[0]),
        "initial_gangi_multiplier": multiplier,
        "confining_pressure": confining_pressure,
        "maximum_closure_pressure": maximum_closure_pressure,
        "biot_coefficient": biot_coefficient,
        "roughness_exponent": roughness_exponent,
        "reference_permeability": reference_permeability,
    }
    return MechanismComparison(
        "gangi",
        global_grid,
        tuple(processed),
        connections,
        wells,
        enabled,
        disabled,
        initial_enabled,
        initial_disabled,
        timesteps,
        diagnostics,
    )


def run_gangi_case(
    *, full_size=False, nstep=None, max_cuts=6, verbose=False,
) -> MechanismComparison:
    """Run the native Gangi-enabled and constant-permeability schedules."""
    case = setup_gangi_case(full_size=full_size)
    steps = case.timesteps if nstep is None else case.timesteps[:int(nstep)]
    case.timesteps = steps
    active_wells = case.wells if full_size else []
    schedule = simple_schedule(steps, wells=active_wells)
    case.enabled_result = run_schedule(
        case.model_enabled,
        case.initial_enabled,
        schedule,
        max_cuts=max_cuts,
        verbose=verbose,
    )
    case.disabled_result = run_schedule(
        case.model_disabled,
        case.initial_disabled,
        schedule,
        max_cuts=max_cuts,
        verbose=verbose,
    )
    return case


__all__ = ["setup_gangi_case", "run_gangi_case"]
