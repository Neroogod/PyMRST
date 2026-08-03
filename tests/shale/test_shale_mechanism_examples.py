import numpy as np
import pytest

from pymrst.modules.shale import (
    run_diffusion_case,
    run_gangi_case,
    run_sorption_case,
    setup_diffusion_case,
    setup_gangi_case,
    setup_sorption_case,
)


def test_sorption_example_builds_live_fractured_models_and_storage_diagnostic():
    case = setup_sorption_case()
    mass = case.diagnostics["initial_sorbed_mass"]
    assert case.global_grid.n_fracture > 0
    assert case.diagnostics["well_intersections"] == 2
    assert mass.shape == (case.global_grid.n_cells, 3)
    assert np.all(mass[:case.global_grid.n_matrix] > 0)
    assert np.allclose(mass[case.global_grid.n_matrix:], 0)


def test_diffusion_and_gangi_examples_activate_their_native_kernels():
    diffusion = setup_diffusion_case()
    gradient = diffusion.diagnostics["initial_composition_gradient"]
    assert gradient.shape[1] == 3
    assert np.any(np.abs(gradient) > 0)

    gangi = setup_gangi_case()
    multiplier = gangi.diagnostics["initial_gangi_multiplier"]
    matrix = gangi.global_grid.rock.is_matrix
    assert np.any(np.abs(multiplier[matrix] - 1) > 1e-6)
    assert np.allclose(multiplier[~matrix], 1)


@pytest.mark.parametrize(
    "runner", [run_sorption_case, run_diffusion_case, run_gangi_case],
)
def test_mechanism_examples_run_enabled_and_disabled_nonlinear_schedules(runner):
    case = runner(nstep=1, max_cuts=8)
    assert len(case.enabled_result.states) == len(case.disabled_result.states) == 1
    assert case.enabled_result.iterations[0] > 0
    assert case.disabled_result.iterations[0] > 0
    assert np.all(np.isfinite(case.enabled_result.final_state.pressure))
    assert np.all(np.isfinite(case.disabled_result.final_state.pressure))
