import numpy as np

from pymrst.modules.shale.examples.eagle_ford import setup_eagle_ford
from pymrst.modules.shale.examples.eagle_ford_edfm import setup_eagle_ford_edfm
from pymrst.modules.shale.examples.eagle_ford_edfm import run_eagle_ford_edfm
from pymrst.modules.shale.examples.eagle_ford_edfm_full import setup_eagle_ford_edfm_full
from pymrst.modules.shale.examples.eagle_ford_full import setup_eagle_ford_full
from pymrst.modules.shale.examples.eagle_ford_accelerated import (
    setup_eagle_ford_accelerated,
)
from pymrst.modules.shale.examples.eagle_ford_edfm_accelerated import (
    setup_eagle_ford_edfm_accelerated,
)
from pymrst.utils.units import DARCY, MICRO, NANO


def test_short_eagle_ford_exact_reference_inputs_and_dataset_mapping():
    case = setup_eagle_ford(natural_fracture_limit=4)
    assert case.method == "pedfm" and not case.full_field
    assert case.grid.cells.num == 1976
    assert len(case.hydraulic_fractures) == 3
    assert len(case.natural_fractures) == 4
    assert case.natural_areas.size == 4
    np.testing.assert_allclose(case.well_path[:, 0], [0, 150])
    np.testing.assert_allclose(
        [plane.permeability for plane in case.natural_fractures],
        [NANO*DARCY, 100*MICRO*DARCY, NANO*DARCY, 100*MICRO*DARCY],
    )
    assert all(plane.metadata["kind"] == "hydraulic"
               for plane in case.hydraulic_fractures)


def test_edfm_and_pedfm_build_distinct_complete_operator_stacks():
    edfm = setup_eagle_ford_edfm(
        natural_fracture_limit=1, build_edfm=True,
    )
    pedfm = setup_eagle_ford(
        natural_fracture_limit=1, build_edfm=True,
    )
    assert edfm.global_grid.n_cells == pedfm.global_grid.n_cells
    assert edfm.global_grid.n_fracture > 0
    assert len(edfm.wells) == len(pedfm.wells) == 1
    assert edfm.well_connections.fracture_cells.size == 3
    assert pedfm.operators.N.shape[0] > edfm.operators.N.shape[0]
    assert np.any(pedfm.operators.face_multiplier < 1)
    assert edfm.initial_state.pressure.size == edfm.global_grid.n_cells


def test_full_field_entrypoints_preserve_21_hfs_and_4200_nf_asset():
    for setup, method in (
        (setup_eagle_ford_full, "pedfm"),
        (setup_eagle_ford_edfm_full, "edfm"),
    ):
        case = setup(build_edfm=False)
        assert case.method == method and case.full_field
        assert len(case.hydraulic_fractures) == 21
        assert len(case.natural_fractures) == 4200
        np.testing.assert_allclose(case.well_path[:, 0], [130, 1050])


def test_natural_fracture_limit_is_validated():
    import pytest
    with pytest.raises(ValueError, match="non-negative"):
        setup_eagle_ford(natural_fracture_limit=-1)


def test_mex_entrypoints_use_available_compiled_diagonal_backend():
    for setup, method in (
        (setup_eagle_ford_accelerated, "pedfm"),
        (setup_eagle_ford_edfm_accelerated, "edfm"),
    ):
        case = setup(natural_fracture_limit=0, build_edfm=True)
        backend = case.model.auto_diff_backend
        report = backend.kernel_report()
        assert case.method == method
        assert backend.row_major and backend.use_mex
        assert report["available"]
        assert report["kernel_family"] == "numba"
        assert report["all_requested_kernels_active"]


def test_reduced_edfm_case_advances_a_live_bhp_production_step():
    case = run_eagle_ford_edfm(
        natural_fracture_limit=0, timesteps=[1.0], max_cuts=2,
    )
    report = case.result.reports[0]
    assert report["Converged"]
    assert len(case.states) == 1
    assert case.states[0].qOs[0] < 0
    assert np.isfinite(case.states[0].pressure).all()
