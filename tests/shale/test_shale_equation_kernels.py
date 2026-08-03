import numpy as np
import pytest
from types import SimpleNamespace

from pymrst.modules.shale import (
    add_sorbed_accumulation,
    apply_gangi_velocity,
    competitive_langmuir_components,
    gas_diffusion_flux,
    langmuir_sorbed_mass,
    non_darcy_velocity,
    sorbed_accumulation,
)
from pymrst.utils.adi import init_variables_adi, value
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.shale.compositional_fns.nat_vars_shale_model import NatVarsShaleModel
from pymrst.modules.shale.other_mrst_fns import make_shale_rock


def test_competitive_langmuir_mass_and_accumulation():
    p = np.array([10.0, 20.0])
    y = np.array([[0.8, 0.2], [0.5, 0.5]])
    iso = np.array([[5.0, 10.0], [2.0, 4.0]])
    gv = np.array([3.0, 2.0])
    mass = langmuir_sorbed_mass(p, y, iso, gv)
    denominator = 1 + p * (y[:, 0] / 5 + y[:, 1] / 10)
    expected = np.column_stack([
        gv * (2 / 5) * y[:, 0] * p / denominator,
        gv * (4 / 10) * y[:, 1] * p / denominator,
    ])
    assert np.allclose(mass, expected)
    assert np.allclose(sorbed_accumulation(p, y, p, y, iso, gv, 2.0), 0)


def test_gas_diffusion_is_componentwise_and_gas_only():
    gradient = np.array([[2.0, -1.0], [1.0, 3.0]])
    flux = gas_diffusion_flux([2e-7, 4e-7], 2.0, np.array([10.0, 0.0]), gradient)
    assert np.allclose(flux[0], [-2e-6, 2e-6])
    assert np.allclose(flux[1], 0)


def test_non_darcy_and_gangi_velocity_limits():
    velocity = np.array([0.0, 0.1, 0.2])
    corrected = non_darcy_velocity(velocity, np.full(3, 0.5))
    assert corrected[0] == 0
    assert np.all(corrected[1:] < velocity[1:])
    assert np.allclose(apply_gangi_velocity(velocity, [1.0, 0.5, 0.25]), [0, .05, .05])
    with pytest.raises(ValueError, match="negative"):
        non_darcy_velocity(np.array([1.0]), np.array([1.0]), positive_root=False)


def test_langmuir_kernel_preserves_pressure_and_composition_jacobians():
    p0 = np.array([10.0, 20.0])
    y0 = np.array([0.8, 0.5])
    p, y_first = init_variables_adi(p0, y0)
    y = [y_first, 1.0 - y_first]
    iso = np.array([[5.0, 10.0], [2.0, 4.0]])
    mass = competitive_langmuir_components(p, y, iso, np.array([3.0, 2.0]))

    assert all(hasattr(component, "jac") for component in mass)
    eps = 1e-6
    plus = competitive_langmuir_components(
        p0 + np.array([eps, 0.0]), [y0, 1.0 - y0], iso, np.array([3.0, 2.0])
    )[0]
    base = competitive_langmuir_components(
        p0, [y0, 1.0 - y0], iso, np.array([3.0, 2.0])
    )[0]
    finite_difference = (plus[0] - base[0]) / eps
    assert mass[0].jac[0][0, 0] == pytest.approx(finite_difference, rel=2e-6)
    assert np.all(np.isfinite(value(mass[1])))


def test_sorbed_accumulation_targets_components_after_water_equation():
    equations = [np.array([10.0]), np.array([20.0]), np.array([30.0])]
    corrected = add_sorbed_accumulation(
        equations, [np.array([1.5]), np.array([-2.0])], water=True
    )
    assert np.array_equal(corrected[0], equations[0])
    assert np.allclose(corrected[1], 21.5)
    assert np.allclose(corrected[2], 28.0)


def _hook_model(mechanisms):
    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    rock = make_shale_rock(
        grid, 1e-15, 0.1,
        is_matrix=np.array([True, False, True]),
        shale_mechanisms=mechanisms,
    )
    model = NatVarsShaleModel.__new__(NatVarsShaleModel)
    model.g = grid
    model.ops = setup_operators_tpfa(grid, rock)
    model.shale_rock = rock
    model.mix = SimpleNamespace(n=2)
    model.shale_diffusion = None
    model.shale_tortuosity = None
    model.shale_gangi_function = None
    model.shale_non_darcy_coefficient = None
    return model


def test_diffusion_flux_hook_is_live_and_ad_aware():
    model = _hook_model({"diffusion": True})
    model.shale_diffusion = np.array([2e-7, 4e-7])
    model.shale_tortuosity = 2.0
    rho, y0 = init_variables_adi(
        np.array([10.0, 11.0, 12.0]), np.array([0.8, 0.6, 0.4])
    )
    flux = model._modify_natural_component_flux(
        0, np.zeros(2), [y0, 1.0 - y0], rho,
        np.array([1.0, 1.0, 1.0]), np.array([True, True]),
    )
    assert hasattr(flux, "jac")
    assert np.any(np.abs(value(flux)) > 0)
    assert flux.jac[0].nnz > 0
    assert flux.jac[1].nnz > 0


def test_closed_system_diffusion_is_componentwise_conservative():
    model = _hook_model({"diffusion": True})
    model.shale_diffusion = np.array([2e-7, 4e-7])
    model.shale_tortuosity = 2.0
    density = np.array([10.0, 11.0, 12.0])
    vapour = [np.array([0.8, 0.6, 0.4]), np.array([0.2, 0.4, 0.6])]
    for component in range(2):
        flux = model._modify_natural_component_flux(
            component, np.zeros(2), vapour, density,
            np.ones(3), np.array([True, True]),
        )
        divergence = model.ops.div(flux)
        assert np.sum(divergence) == pytest.approx(0.0, abs=1e-20)


def test_gangi_velocity_hook_scales_matrix_but_not_fracture_upwind_cell():
    model = _hook_model({"Gangi": True})
    model.shale_gangi_function = lambda pressure: pressure / 10.0
    pressure = np.array([5.0, 6.0, 7.0])
    velocity = model._modify_natural_velocity(
        "gas", np.ones(2), pressure, np.array([True, True]),
        np.ones(3), np.ones(3),
    )
    assert np.allclose(velocity, [0.5, 1.0])


def test_non_darcy_velocity_hook_uses_supplied_cell_coefficient():
    model = _hook_model({"nonDarcyEverywhere": True})
    model.shale_non_darcy_coefficient = np.full(3, 0.1)
    velocity = model._modify_natural_velocity(
        "gas", np.array([0.1, 0.1]), np.ones(3), np.array([True, True]),
        np.ones(3), np.ones(3),
    )
    assert np.all(velocity > 0.1)
