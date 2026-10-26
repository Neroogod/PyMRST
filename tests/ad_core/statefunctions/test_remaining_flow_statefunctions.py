"""Reference coverage for remaining MRST flow state-function building blocks."""

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.statefunctions.components import (
    ConcentrationComponent,
    GenericComponent,
)
from pymrst.modules.ad_core.statefunctions.composite import StateFunctionDependent
from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
    ComponentPhaseMass,
)
from pymrst.modules.ad_core.statefunctions.flowprops.component_properties import (
    ComponentProperty,
)
from pymrst.modules.ad_core.statefunctions.flowprops.saturation_limits import RvMax
from pymrst.modules.ad_core.statefunctions.flowprops.virtual import (
    SaturationProperty,
)
from pymrst.modules.ad_core.statefunctions.flux.permeability_gradient_discretization import (
    PermeabilityGradientDiscretization,
    TwoPointFluxApproximation,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.boundary_faces import (
    get_boundary_faces_merged_ad,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.utils.adi import init_variables_adi, value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _model():
    grid = compute_geometry(cart_grid([3, 1, 1], [30.0, 10.0, 10.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[1e-6 / BARSA, 1e-4 / BARSA],
        p_ref=200 * BARSA,
    )
    return GenericBlackOilModelAD(grid, rock, fluid, [])


def _state(model):
    return model.init_state_ad(State(
        pressure=np.full(3, 200 * BARSA),
        sW=np.array([0.2, 0.5, 0.8]),
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    ))


def test_component_phase_mass_matches_component_definition_with_ad():
    model = _model()
    state = _state(model)
    phase_mass = model.get_props(state, "ComponentPhaseMass")
    saturation = model.get_props(state, "Saturations")
    pore_volume = model.get_props(state, "PoreVolume")
    shrinkage = model.get_props(state, "ShrinkageFactors")
    for component_index, component in enumerate(model.components):
        density = component.component_density(
            model, state, shrinkage
        )
        for phase in range(model.n_phases):
            actual = phase_mass[component_index][phase]
            if density[phase] is None:
                assert actual is None
            else:
                expected = pore_volume * saturation[phase] * density[phase]
                assert np.allclose(value(actual), value(expected))
                assert all(
                    np.allclose((a - b).toarray(), 0.0)
                    for a, b in zip(actual.jac, expected.jac)
                )


def test_state_function_dependent_tracks_method_specific_dependencies():
    dependent = StateFunctionDependent()
    returned = dependent.function_depends_on(
        "compute_flux", ["Mobility", "Density"]
    )
    dependent.function_depends_on(
        "compute_flux", "Pressure", grouping="PVTPropertyFunctions"
    )
    assert returned is dependent
    dependencies = dependent.get_function_dependencies("compute_flux")
    assert [(item.name, item.grouping) for item in dependencies] == [
        ("Mobility", "self"),
        ("Density", "self"),
        ("Pressure", "PVTPropertyFunctions"),
    ]
    assert dependent.get_function_dependencies("unknown") == []
    dependent.clear_function_dependencies("compute_flux")
    assert dependent.get_function_dependencies("compute_flux") == []


def test_generic_and_concentration_component_contracts():
    model = _model()
    state = _state(model)
    shrinkage = model.get_props(state, "ShrinkageFactors")
    generic = GenericComponent("empty")
    assert generic.component_density(model, state, shrinkage) == [None, None]

    tracer = ConcentrationComponent("tracer", phase_index=0, state_key="cT")
    concentration = np.array([0.0, 0.5, 1.0])
    density = tracer.component_density(
        model, {"cT": concentration}, shrinkage
    )
    assert density[1] is None
    assert np.allclose(value(density[0]), concentration * value(shrinkage[0]))

    prop = ComponentProperty(model, "get_component_density")
    assert prop.component_function_name == "get_component_density"


def test_rvmax_uses_gas_phase_pressure():
    fn = RvMax()
    fn.struct_name = "FlowProps"
    state = State()
    state.containers["FlowProps"] = {
        "PhasePressures": [
            np.array([100.0, 110.0]),
            np.array([120.0, 130.0]),
            np.array([140.0, 150.0]),
        ]
    }
    model = SimpleNamespace(
        n_phases=3,
        fluid=SimpleNamespace(rv_sat=lambda pressure: 2.0 * pressure),
    )
    assert np.array_equal(
        fn.evaluate_on_domain(model, state), [280.0, 300.0]
    )


def test_saturation_property_stores_connate_water():
    model = _model()
    model.raw_fluid.swcon = 0.17
    prop = SaturationProperty(model)
    assert np.allclose(prop.get_connate_water(model), 0.17)
    prop.store_connate_water(model)
    model.raw_fluid.swcon = 0.25
    assert np.allclose(prop.get_connate_water(model), 0.17)
    assert not prop.scaling_active


def test_permeability_gradient_tpfa_numeric_and_ad():
    model = _model()
    state = State()
    nface = model.operators.T.size
    difference = np.linspace(-2.0, 3.0, nface)
    tpfa = TwoPointFluxApproximation(model)
    numeric = tpfa.get_permeability_gradient(model, state, difference)
    assert np.allclose(numeric, model.operators.T * difference)

    (difference_ad,) = init_variables_adi(difference)
    automatic = tpfa.get_permeability_gradient(model, State(), difference_ad)
    assert np.allclose(value(automatic), model.operators.T * difference)
    assert np.allclose(
        automatic.jac[0].toarray(), np.diag(model.operators.T)
    )
    with pytest.raises(NotImplementedError):
        PermeabilityGradientDiscretization().get_permeability_gradient(
            model, state, difference
        )


def test_get_boundary_faces_merged_reflects_owner_through_face():
    model = _model()
    grid = model.G
    faces, cells, ghost = get_boundary_faces_merged_ad(grid)
    face_centroid = grid.faces.centroids[faces]
    cell_centroid = grid.cells.centroids[cells]
    assert np.all(np.any(grid.faces.neighbors[faces] < 0, axis=1))
    assert np.allclose(face_centroid, 0.5 * (cell_centroid + ghost))
