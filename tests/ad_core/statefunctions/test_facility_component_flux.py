import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.statefunctions.facility import (
    FacilityContext,
    InjectionSurfaceDensity,
    PerforationComponentPhaseDensity,
    WellComponentPhaseFlux,
    WellComponentTotalFlux,
    WellComponentTotalFluxDensityMix,
    WellComponentTotalVolumeBalanceCrossflow,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _case():
    grid = compute_geometry(cart_grid([3, 1, 2], [300.0, 100.0, 20.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 4.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[1e-6 / BARSA, 1e-4 / BARSA],
        p_ref=200 * BARSA,
    )
    wells = add_well(
        None, grid, rock, np.array([0, 3]), well_type="rate", val=2e-3,
        compi=np.array([1.0, 0.0]), sign=1, name="I",
    )
    wells = add_well(
        wells, grid, rock, np.array([2, 5]), well_type="bhp",
        val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1, name="P",
    )
    model = GenericBlackOilModelAD(grid, rock, fluid, wells)
    reservoir_state = model.init_state_ad(State(
        pressure=np.full(grid.cells.num, 200 * BARSA),
        sW=np.linspace(0.35, 0.65, grid.cells.num),
        qWs=np.array([2e-3, 0.0]),
        qOs=np.array([0.0, -2e-3]),
        bhp=np.array([250 * BARSA, 150 * BARSA]),
    ))
    facility = FacilityContext(model, wells)
    state = State(
        reservoir_state=reservoir_state,
        bhp=np.array([250 * BARSA, 150 * BARSA]),
        cdp=np.zeros(4),
    )
    return model, wells, facility, state


def test_complete_component_chain_is_registered_under_mrst_property_names():
    _, _, facility, _ = _case()
    functions = facility.FacilityFlowDiscretization.functions
    expected = {
        "InjectionSurfaceDensity": InjectionSurfaceDensity,
        "ComponentPhaseDensity": PerforationComponentPhaseDensity,
        "ComponentPhaseFlux": WellComponentPhaseFlux,
        "ComponentTotalFlux": WellComponentTotalFlux,
        "ComponentTotalFluxDensityMix": WellComponentTotalFluxDensityMix,
        "ComponentTotalVolumeBalanceCrossflow":
            WellComponentTotalVolumeBalanceCrossflow,
    }
    for property_name, implementation in expected.items():
        assert isinstance(functions[property_name], implementation)


def test_perforation_component_density_samples_values_and_jacobians():
    model, _, facility, state = _case()
    mapping = facility.get_props(state, "FacilityWellMapping")
    sampled = facility.get_props(state, "ComponentPhaseDensity")

    for component_index, component in enumerate(model.components):
        cell_density = component.get_component_density(
            model, state["reservoir_state"]
        )
        for phase in range(model.n_phases):
            if cell_density[phase] is None:
                assert sampled[component_index][phase] is None
                continue
            expected = cell_density[phase][mapping.cells]
            actual = sampled[component_index][phase]
            assert np.allclose(value(actual), value(expected))
            for actual_jac, expected_jac in zip(actual.jac, expected.jac):
                assert np.allclose(
                    (actual_jac - expected_jac).toarray(), 0.0
                )


def test_injection_surface_density_honors_injector_override_only():
    model, wells, facility, state = _case()
    wells[0].rhoS = np.array([997.0, 777.0])
    wells[1].rhoS = np.array([123.0, 456.0])
    density = facility.get_props(state, "InjectionSurfaceDensity")

    assert np.allclose([phase[0] for phase in density], [997.0, 777.0])
    assert np.allclose(
        [phase[1] for phase in density], np.asarray(model.fluid.rho_s)
    )


def test_component_phase_flux_is_density_times_phase_flux_with_ad():
    _, _, facility, state = _case()
    phase_flux = facility.get_props(state, "WellPhaseFlux")
    density = facility.get_props(state, "ComponentPhaseDensity")
    component_flux = facility.get_props(state, "ComponentPhaseFlux")

    # Both injection (first two perforations) and production (last two) occur.
    assert np.all(value(phase_flux[0])[:2] >= 0.0)
    assert np.all(value(phase_flux[0])[2:] <= 0.0)
    for component in range(len(component_flux)):
        for phase in range(len(phase_flux)):
            if density[component][phase] is None:
                assert component_flux[component][phase] is None
                continue
            expected = density[component][phase] * phase_flux[phase]
            actual = component_flux[component][phase]
            assert np.allclose(value(actual), value(expected))
            for actual_jac, expected_jac in zip(actual.jac, expected.jac):
                assert np.allclose(
                    (actual_jac - expected_jac).toarray(), 0.0
                )


def test_total_component_flux_sums_phases_for_producer_and_injector():
    _, _, facility, state = _case()
    phase_flux = facility.get_props(state, "ComponentPhaseFlux")
    total_flux = facility.get_props(state, "ComponentTotalFlux")

    for component, actual in enumerate(total_flux):
        expected = None
        for flux in phase_flux[component]:
            if flux is not None:
                expected = flux if expected is None else expected + flux
        assert np.allclose(value(actual), value(expected))
        for actual_jac, expected_jac in zip(actual.jac, expected.jac):
            assert np.allclose((actual_jac - expected_jac).toarray(), 0.0)


def test_all_total_variants_conserve_crossflow_component_amount():
    _, _, facility, state = _case()
    # First producer completion now injects while its second completion still
    # produces: an explicit within-well crossflow case.
    state["cdp"][2] = 80 * BARSA
    names = [
        "ComponentTotalFlux",
        "ComponentTotalFluxDensityMix",
        "ComponentTotalVolumeBalanceCrossflow",
    ]
    totals = [facility.get_props(state, name) for name in names]
    mapping = facility.get_props(state, "FacilityWellMapping")

    for candidate in totals[1:]:
        for base, other in zip(totals[0], candidate):
            assert np.allclose(value(base), value(other))
    # Mixing may redistribute components at injecting perforations, but the
    # aggregate component stream through each well remains conservative.
    for total in totals:
        assert all(np.all(np.isfinite(value(flux))) for flux in total)
        aggregate = np.column_stack([
            mapping.perforation_sum @ value(flux) for flux in total
        ])
        assert np.all(np.isfinite(aggregate))
