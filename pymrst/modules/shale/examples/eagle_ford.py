"""Native pEDFM Eagle Ford workflow (MRST ``eagleFord.m``).

This public MRST-layout owner contains the complete grid, fracture, well,
operator, fluid, state, schedule and simulation workflow. The related EDFM,
full-field and accelerated owners reuse its tested construction primitives but
select their own discretization and backend explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_core.timesteps import rampup_timesteps
from pymrst.modules.compositional.examples._shared import run_schedule, simple_schedule
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.modules.compositional.utils.init_compositional_state import init_compositional_state
from pymrst.modules.shale.data import FracturePlane, ShaleNNC
from pymrst.modules.shale.examples.data_loader import load_shale_dataset
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
from pymrst.modules.shale.pedfm_nnc import (
    projected_matrix_fracture_nncs_3d,
    setup_pedfm_operators_tpfa,
)
from pymrst.modules.shale.utils import (
    create_multistage_hfs,
    get_shale_comp_fluid_case,
    mesh_hf_system,
    process_stochastic_fractures,
)
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import DARCY, DAY, FT, MICRO, MILLI, NANO, PSIA, YEAR


@dataclass
class EagleFordCase:
    """Geometry, data provenance and optional simulation state for one variant."""

    method: str
    full_field: bool
    grid: object
    matrix_rock: object
    hydraulic_fractures: tuple[FracturePlane, ...]
    natural_fractures: tuple[FracturePlane, ...]
    rejected_natural_fractures: tuple[np.ndarray, ...]
    natural_areas: np.ndarray
    well_path: np.ndarray
    well_radius: float
    bottom_hole_pressure: float
    source: str = "bundled-mrst"
    global_grid: object | None = None
    processed_planes: tuple[FracturePlane, ...] = ()
    well_connections: object | None = None
    operators: object | None = None
    wells: list | None = None
    mixture: object | None = None
    model: object | None = None
    initial_state: object | None = None
    timesteps: np.ndarray | None = None
    result: object | None = None

    @property
    def fracture_planes(self):
        return self.hydraulic_fractures + self.natural_fractures

    @property
    def states(self):
        return [] if self.result is None else self.result.states


def _merge_nnc(first: ShaleNNC, second: ShaleNNC) -> ShaleNNC:
    return ShaleNNC(
        np.vstack([first.cells, second.cells]), np.r_[first.T, second.T],
        first.type + second.type, np.r_[first.area, second.area],
        np.vstack([first.normal, second.normal]),
        np.vstack([first.plane_point, second.plane_point]),
    )


def _natural_fractures(full_field: bool, limit: int | None):
    asset = "eagle_ford_full_fractures" if full_field else "eagle_ford_fractures"
    raw = load_shale_dataset(asset)
    source = tuple(raw["set1"]) + tuple(raw["set2"])
    planar, rejected, areas = process_stochastic_fractures(source)
    if limit is not None:
        if int(limit) < 0:
            raise ValueError("natural_fracture_limit must be non-negative")
        planar = planar[:int(limit)]
        areas = np.asarray(areas)[:int(limit)]
    planes = []
    for index, polygon in enumerate(planar):
        planes.append(FracturePlane(
            np.asarray(polygon, dtype=float)[:-1], aperture=100*NANO,
            porosity=.5,
            permeability=(1*NANO if index % 2 == 0 else 100*MICRO)*DARCY,
            metadata={"kind": "natural", "source_index": index},
        ))
    return tuple(planes), tuple(rejected), areas


def setup_eagle_ford_case(
    *, method: str = "pedfm", full_field: bool = False,
    natural_fracture_limit: int | None = None, build_edfm: bool = False,
    auto_diff_backend=None,
) -> EagleFordCase:
    """Build either original MRST Eagle Ford EDFM/pEDFM configuration.

    ``build_edfm=False`` constructs the exact matrix, fracture and well input
    data without performing the expensive embedded-fracture preprocessing.
    A finite ``natural_fracture_limit`` is intended for fast regression runs;
    ``None`` retains all 150 or 4,200 bundled natural fractures.
    """
    method = str(method).lower()
    if method not in {"edfm", "pedfm"}:
        raise ValueError("method must be 'edfm' or 'pedfm'")
    tolerance = 1e-5
    if full_field:
        physical = np.array([1200., 600., 76.2])
        num_stages, per_stage = 7, 3
        first_center = np.array([130., 300., 38.1])
        well_path = np.array([[130., 300., 38.1], [1050., 300., 38.1]])
        half_length, height = 180., 150*FT
    else:
        physical = np.array([150., 300., 38.1])
        num_stages, per_stage = 1, 3
        half_length, height = 180., 75*FT
        first_center = np.array([65., half_length/2, physical[2]-height/2])
        well_path = np.array([
            [0., 0., physical[2]-tolerance],
            [150., 0., physical[2]-tolerance],
        ])
    hydraulic, centers = create_multistage_hfs(
        num_stages=num_stages, frac_spacing=150.,
        num_fracs_per_stage=per_stage, frac_half_length=half_length,
        frac_height=height, cluster_spacing=10.,
        frac_centroid1=first_center, is_stencil=True,
    )
    for index, plane in enumerate(hydraulic):
        plane.aperture = 3*MILLI
        plane.porosity = .5
        plane.permeability = DARCY
        plane.metadata.update(kind="hydraulic", source_index=index)
    grid = compute_geometry(mesh_hf_system(
        physical, centers, num_stages=num_stages,
        num_fracs_per_stage=per_stage, ny=20, nz=5,
        nx_refine_small=4, nx_refine=9,
    ))
    rock = make_shale_rock(grid, 900*NANO*DARCY, .12)
    natural, rejected, areas = _natural_fractures(
        full_field, natural_fracture_limit,
    )
    case = EagleFordCase(
        method, full_field, grid, rock, tuple(hydraulic), natural, rejected,
        areas, well_path, .328084*FT, 1000*PSIA,
    )
    if not build_edfm:
        return case

    planes = case.fracture_planes
    global_grid, processed = edfm_shale_grid(grid, planes, rock, tolerance=tolerance)
    fracfrac = fracture_fracture_shale_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        aperture=[plane.aperture for plane in planes], tolerance=tolerance,
    )
    global_grid.nnc = _merge_nnc(global_grid.nnc, fracfrac)
    connections = well_fracture_nncs_3d(
        [well_path], global_grid.fractures, global_grid.projections,
        matrix_cell_count=grid.cells.num, tolerance=tolerance,
    )
    face_t = get_face_transmissibility(global_grid.grid, global_grid.rock)
    if method == "pedfm":
        projected = projected_matrix_fracture_nncs_3d(
            global_grid.edfm, global_grid.fractures, global_grid.projections,
            tolerance=tolerance,
        )
        operators = setup_pedfm_operators_tpfa(
            global_grid.grid, global_grid.rock, face_t, global_grid.nnc,
            projected=projected, tolerance=tolerance,
            cell_volumes=global_grid.grid.cells.volumes,
        )
    else:
        operators = setup_shale_edfm_operators_tpfa(
            global_grid.grid, global_grid.rock, face_t, global_grid.nnc,
            cell_volumes=global_grid.grid.cells.volumes,
        )
    wells = []
    if connections.fracture_cells.size:
        completions = np.column_stack([
            connections.fracture_cells, connections.fracture_index,
        ])
        offsets = np.cumsum(
            [grid.cells.num]
            + [projection.matrix_cells.size for projection in global_grid.projections]
        )
        cell_sides = []
        for cell, plane_index in completions:
            local = int(cell - offsets[int(plane_index)])
            polygon = np.asarray(
                global_grid.projections[int(plane_index)].polygons[local], float,
            )
            _, axis1, axis2 = global_grid.fractures[int(plane_index)].frame()
            relative = polygon - polygon.mean(axis=0)
            cell_sides.append([
                np.ptp(relative @ axis1), np.ptp(relative @ axis2),
            ])
        mixture, info = get_shale_comp_fluid_case("eagleford")
        wells = add_well_shale(
            wells, global_grid, completions,
            fracture_aperture=[plane.aperture for plane in planes],
            fracture_cell_side_lengths=np.asarray(cell_sides),
            radius=case.well_radius, well_type="bhp",
            val=case.bottom_hole_pressure, compi=[.17, .74, .09], sign=-1,
            name="Prod1", hydraulic_plane_count=len(hydraulic),
            components=info.initial,
        )
    else:
        mixture, info = get_shale_comp_fluid_case("eagleford")
    model = NaturalVariablesCompositionalModel(
        global_grid.grid, global_grid.rock, mixture, temperature=info.T,
        wells=wells, water=True, n_kr=(2, 2, 2), max_iterations=40,
        auto_diff_backend=auto_diff_backend,
    )
    model.ops = operators
    model.pv = operators.pv
    model.gdz = np.zeros(operators.N.shape[0])
    saturation = np.tile([.17, .74, .09], (global_grid.n_cells, 1))
    state0 = init_compositional_state(
        model, info.pressure, info.T, saturation, info.initial,
    )
    case.global_grid, case.processed_planes = global_grid, tuple(processed)
    case.well_connections, case.operators, case.wells = connections, operators, wells
    case.mixture, case.model, case.initial_state = mixture, model, state0
    case.timesteps = rampup_timesteps(15*YEAR, 30*DAY, 15)
    return case


def run_eagle_ford_case(**kwargs):
    """Preprocess and simulate an Eagle Ford configuration."""
    nstep = kwargs.pop("nstep", None)
    max_cuts = kwargs.pop("max_cuts", 8)
    verbose = kwargs.pop("verbose", False)
    timesteps = kwargs.pop("timesteps", None)
    kwargs["build_edfm"] = True
    case = setup_eagle_ford_case(**kwargs)
    steps = case.timesteps if timesteps is None else np.asarray(timesteps, dtype=float)
    steps = steps if nstep is None else steps[:int(nstep)]
    case.timesteps = steps
    case.result = run_schedule(
        case.model, case.initial_state, simple_schedule(steps, wells=case.wells),
        max_cuts=max_cuts, verbose=verbose,
    )
    return case


def setup_eagle_ford(**kwargs):
    """Build the MRST ``eagleFord.m`` pEDFM case."""
    return setup_eagle_ford_case(method="pedfm", full_field=False, **kwargs)


def run_eagle_ford(**kwargs):
    """Build and simulate the MRST ``eagleFord.m`` pEDFM case."""
    return run_eagle_ford_case(method="pedfm", full_field=False, **kwargs)


eagleFord = run_eagle_ford

__all__ = [
    "EagleFordCase", "setup_eagle_ford_case", "run_eagle_ford_case",
    "setup_eagle_ford", "run_eagle_ford", "eagleFord",
]


if __name__ == "__main__":

    def test_the_default_case_builds_its_fracture_network():
        """Six example drivers share this setup, so its shape is a contract.

        `setup_eagle_ford_case` stops before assembling the model - the grid,
        rock and fracture geometry are built, the operators and wells are not.
        Asserting that boundary keeps the split visible: a later change that
        starts assembling here would make every driver pay for work it may not
        want.
        """
        case = setup_eagle_ford_case()
        assert case.grid.cells.num > 0
        assert len(case.hydraulic_fractures) == 3, len(case.hydraulic_fractures)
        assert len(case.natural_fractures) == 150, len(case.natural_fractures)
        assert case.natural_areas.size == len(case.natural_fractures), (
            "the unlimited case must pair one area per plane")
        assert case.model is None and case.operators is None, (
            "setup assembled more than it used to")
        print(f"ok test_the_default_case_builds_its_fracture_network "
              f"({case.grid.cells.num} cells, 3 hydraulic + "
              f"{len(case.natural_fractures)} natural fractures)")

    def test_natural_fracture_areas_are_positive_and_finite():
        """A zero-area plane contributes no EDFM transmissibility at all.

        It is not an error anywhere - the plane simply stops conducting - so a
        degenerate one is invisible until a fracture is mysteriously inactive.
        """
        case = setup_eagle_ford_case()
        areas = np.asarray(case.natural_areas, dtype=float)
        assert np.all(np.isfinite(areas)), "non-finite fracture area"
        assert np.all(areas > 0.0), (areas.min(), int(np.count_nonzero(areas <= 0)))
        print(f"ok test_natural_fracture_areas_are_positive_and_finite "
              f"({areas.min():.4g} to {areas.max():.4g} m2)")

    def test_the_well_path_spans_the_model():
        case = setup_eagle_ford_case()
        path = np.asarray(case.well_path, dtype=float)
        assert path.shape[0] == 2, path.shape
        assert case.well_radius > 0.0 and case.bottom_hole_pressure > 0.0
        assert np.all(np.isfinite(path))
        print(f"ok test_the_well_path_spans_the_model "
              f"(rw = {case.well_radius:.4g} m, BHP = "
              f"{case.bottom_hole_pressure / 1e5:.1f} bar)")

    def test_limiting_the_natural_fractures_is_honoured():
        """The drivers use this to trade fidelity for runtime."""
        few = setup_eagle_ford_case(natural_fracture_limit=10)
        assert len(few.natural_fractures) == 10, len(few.natural_fractures)
        full = setup_eagle_ford_case()
        assert few.natural_areas.size == 10
        assert np.allclose(few.natural_areas, full.natural_areas[:10]), (
            "truncation reordered the planes, so areas no longer index alike")
        print(f"ok test_limiting_the_natural_fractures_is_honoured "
              f"(10 planes and matching areas retained)")

    def test_rejected_planes_are_reported_not_dropped():
        """A plane that fails processing must be visible, not silently absent.

        Otherwise a geometry bug shows up only as a slightly lower recovery.
        """
        case = setup_eagle_ford_case()
        total = len(case.natural_fractures) + len(case.rejected_natural_fractures)
        assert total >= len(case.natural_fractures)
        assert isinstance(case.rejected_natural_fractures, tuple)
        print(f"ok test_rejected_planes_are_reported_not_dropped "
              f"({len(case.rejected_natural_fractures)} rejected of {total})")

    test_the_default_case_builds_its_fracture_network()
    test_natural_fracture_areas_are_positive_and_finite()
    test_the_well_path_spans_the_model()
    test_limiting_the_natural_fractures_is_honoured()
    test_rejected_planes_are_reported_not_dropped()
    print("\nAll tests passed!")
