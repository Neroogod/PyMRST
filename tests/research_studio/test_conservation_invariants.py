"""Conservation-basis and invariant-profile certification."""

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import OWState
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.research_studio import (
    BlackOilConservationAdapter, BlackOilPseudoComponentConservationAdapter,
    CompositionalConservationAdapter, EventKind, InvariantSuite,
    InvariantViolation, OilWaterConservationAdapter, SimulationSession,
    PoromechanicsConservationAdapter, WaterConservationAdapter, bounded_field,
    aquifer_component_rates, conservation_report, finite_field,
)


class _ConstantBlackOilFluid:
    def b_w(self, pressure):
        return np.full_like(np.asarray(pressure, dtype=float), 2.0)

    def b_o(self, pressure, rs, gas_present):
        return np.full_like(np.asarray(pressure, dtype=float), 3.0)

    def b_g(self, pressure):
        return np.full_like(np.asarray(pressure, dtype=float), 4.0)


def test_blackoil_inventory_uses_surface_components_and_dissolved_gas():
    model = SimpleNamespace(pv=np.array([10.0, 20.0]), fluid=_ConstantBlackOilFluid())
    adapter = BlackOilConservationAdapter(model)
    state = {
        "pressure": np.array([100.0, 200.0]),
        "sW": np.array([0.1, 0.2]),
        "sG": np.array([0.2, 0.1]),
        "rs": np.array([5.0, 7.0]),
        "qWs": np.array([1.0, -0.5]),
        "qOs": np.array([2.0, -0.25]),
        "qGs": np.array([8.0, -3.0]),
    }
    inventory = adapter.inventory(state)
    so = np.array([0.7, 0.7])
    np.testing.assert_allclose(inventory[:, 0], model.pv*2.0*state["sW"])
    np.testing.assert_allclose(inventory[:, 1], model.pv*3.0*so)
    np.testing.assert_allclose(
        inventory[:, 2],
        model.pv*(4.0*state["sG"] + state["rs"]*3.0*so),
    )
    # qGs is already the total surface-gas component rate.  Rs*qOs must not
    # be added again at the evidence layer.
    np.testing.assert_allclose(adapter.well_rates(state), [0.5, 1.75, 5.0])


def test_blackoil_accepts_wog_matrix_and_generic_property_graph():
    class GraphModel:
        def get_props(self, state, name):
            assert name in {"PoreVolume", "ShrinkageFactors"}
            if name == "PoreVolume":
                return np.array([4.0])
            return [np.array([2.0]), np.array([3.0]), np.array([4.0])]

    state = {
        "pressure": np.array([1.0]), "s": np.array([[0.2, 0.5, 0.3]]),
        "rs": np.array([10.0]), "qWs": np.zeros(0), "qOs": np.zeros(0),
        "qGs": np.zeros(0),
    }
    inventory = BlackOilConservationAdapter(GraphModel()).inventory(state)
    np.testing.assert_allclose(inventory, [[1.6, 6.0, 64.8]])


def test_blackoil_dissolved_gas_redistribution_preserves_total_gas_component():
    model = SimpleNamespace(pv=np.array([1.0]), fluid=_ConstantBlackOilFluid())
    adapter = BlackOilConservationAdapter(model)
    common = {
        "pressure": np.array([1.0]), "sW": np.array([0.0]),
        "qWs": np.zeros(0), "qOs": np.zeros(0), "qGs": np.zeros(0),
    }
    initial = dict(common, sG=np.array([0.25]), rs=np.array([2.0]))
    # Initial gas = 4*.25 + 2*3*.75 = 5.5.  After moving reservoir volume
    # from free gas into oil, choose Rs so total surface gas remains 5.5.
    final = dict(common, sG=np.array([0.10]), rs=np.array([5.1/2.7]))
    report = conservation_report(adapter, initial, final, 1.0)
    np.testing.assert_allclose(report.change[2], 0.0, atol=1e-14)
    np.testing.assert_allclose(report.imbalance[2], 0.0, atol=1e-14)


def test_compositional_inventory_matches_equation_mass_basis_and_well_fluxes():
    class Model:
        pv = np.array([2.0, 3.0])
        mix = SimpleNamespace(names=("C1", "C3"), n=2)
        water = False
        rock_pv_mult = None

        def _flow_props0(self, state):
            return (
                [np.array([0.8, 0.6]), np.array([0.2, 0.4])],
                [np.array([0.3, 0.1]), np.array([0.7, 0.9])],
                np.array([0.7, 0.4]), np.array([0.3, 0.6]),
                np.array([600.0, 650.0]), np.array([80.0, 90.0]),
            )

    state = {
        "pressure": np.ones(2),
        "well_components": np.array([[2.0, 3.0], [-0.5, -1.0]]),
    }
    adapter = CompositionalConservationAdapter(Model())
    expected = Model.pv[:, None]*(
        (np.array([600.0, 650.0])*np.array([0.7, 0.4]))[:, None]
        * np.array([[0.8, 0.2], [0.6, 0.4]])
        + (np.array([80.0, 90.0])*np.array([0.3, 0.6]))[:, None]
        * np.array([[0.3, 0.7], [0.1, 0.9]])
    )
    np.testing.assert_allclose(adapter.inventory(state), expected)
    np.testing.assert_allclose(adapter.well_rates(state), [1.5, 2.0])
    assert adapter.component_names == ("C1", "C3")
    assert adapter.units == "kg"


def test_compositional_phase_split_change_does_not_change_overall_components():
    class Model:
        pv = np.array([5.0])
        mix = SimpleNamespace(names=("A", "B"), n=2)
        water = False
        rock_pv_mult = None

        def _flow_props0(self, state):
            split = float(state["split"])
            # Equal phase density and composition isolate the important
            # invariant: changing only the phase label/split changes no mass.
            composition = [np.array([0.25]), np.array([0.75])]
            return composition, composition, np.array([split]), np.array([1.0-split]), np.array([10.0]), np.array([10.0])

    empty_rates = np.empty((0, 2))
    initial = {"pressure": np.ones(1), "split": 0.1, "well_components": empty_rates}
    final = {"pressure": np.ones(1), "split": 0.9, "well_components": empty_rates}
    report = conservation_report(
        CompositionalConservationAdapter(Model()), initial, final, 2.0,
    )
    np.testing.assert_allclose(report.change, np.zeros(2), atol=1e-14)
    np.testing.assert_allclose(report.imbalance, np.zeros(2), atol=1e-14)


def test_poromechanics_conserves_fluid_and_reports_coupling_separately():
    fluid = init_simple_adi_fluid(mu=[1.0, 2.0], rho=[1000.0, 800.0])

    class MechWaterModel:
        # `raw_fluid.b_w` is deliberately present *and* deliberately wrong, so
        # the test distinguishes a b taken from the state-function graph from
        # one taken off the fluid. A real poromechanics model has the same
        # shape with the roles reversed: its PhaseFluid has no `b_w` at all,
        # and the adapter used to silently substitute b = 1 there.
        raw_fluid = SimpleNamespace(b_w=lambda p: np.full(np.size(p), 99.0))

        def get_props(self, state, name):
            assert name in {"PoreVolume", "ShrinkageFactors"}, name
            if name == "ShrinkageFactors":
                return (np.array([2.0]),)
            return np.array([1.0 + float(state["vdiv"][0])])

    model = MechWaterModel()
    adapter = PoromechanicsConservationAdapter(model)
    initial = {
        "pressure": np.ones(1), "qWs": np.zeros(1),
        "u": np.array([0.0, 0.0]), "vdiv": np.array([0.0]),
    }
    final = {
        "pressure": np.ones(1), "qWs": np.array([0.05]),
        "u": np.array([0.1, 0.0]), "vdiv": np.array([0.1]),
    }
    report = conservation_report(adapter, initial, final, 2.0)
    # Pore volume 1.0 -> 1.1, b = 2.0 from the graph: inventory 2.0 -> 2.2.
    # Taken off `raw_fluid.b_w` instead it would be 99.0 -> 108.9, and taken
    # from neither it would be 1.0 -> 1.1.
    np.testing.assert_allclose(report.initial, [2.0])
    np.testing.assert_allclose(report.final, [2.2])
    np.testing.assert_allclose(report.change, [0.2])
    np.testing.assert_allclose(report.wells, [0.1])
    assert report.coupling_diagnostics == {
        "initial_displacement_l2": 0.0,
        "initial_volumetric_strain_l2": 0.0,
        "initial_mean_volumetric_strain": 0.0,
        "initial_effective_pore_volume": 1.0,
        "final_displacement_l2": 0.1,
        "final_volumetric_strain_l2": 0.1,
        "final_mean_volumetric_strain": 0.1,
        "final_effective_pore_volume": 1.1,
        "effective_pore_volume_change": pytest.approx(0.1),
    }


def test_poromechanics_requires_an_unambiguous_fluid_basis():
    with pytest.raises(ValueError, match="cannot infer"):
        PoromechanicsConservationAdapter(SimpleNamespace())
    with pytest.raises(ValueError, match="unsupported"):
        PoromechanicsConservationAdapter(SimpleNamespace(), "compositional")


def test_aquifer_rates_use_committed_fetkovich_and_carter_tracy_state():
    adapter = SimpleNamespace(component_names=("water", "oil", "gas"))
    initial = {
        "aquifer_V": np.array([100.0, 50.0]),
        "aquifer_ct_W": np.array([2.0]),
    }
    final = {
        "aquifer_V": np.array([96.0, 49.0]),
        "aquifer_ct_W": np.array([5.0]),
    }
    # Five surface m3 from Fetkovich plus three from Carter-Tracy in 2 s.
    np.testing.assert_allclose(
        aquifer_component_rates(adapter, initial, final, 2.0),
        [4.0, 0.0, 0.0],
    )


def test_aquifer_rates_validate_history_and_nonwater_basis():
    hydrocarbon = SimpleNamespace(component_names=("C1", "C2"))
    assert np.array_equal(
        aquifer_component_rates(hydrocarbon, {}, {}, 1.0), np.zeros(2),
    )
    water = SimpleNamespace(component_names=("water",))
    with pytest.raises(ValueError, match="incomplete"):
        aquifer_component_rates(water, {"aquifer_V": [1.0]}, {}, 1.0)
    with pytest.raises(ValueError, match="positive"):
        aquifer_component_rates(water, {}, {}, 0.0)


def test_water_closed_and_source_driven_balances_and_regions():
    fluid = init_simple_adi_fluid(mu=[1.0, 2.0], rho=[1000.0, 800.0])
    model = SimpleNamespace(pv=np.array([2.0, 3.0]), raw_fluid=fluid)
    adapter = WaterConservationAdapter(model)
    initial = {"pressure": np.ones(2), "qWs": np.zeros(1)}
    closed = conservation_report(adapter, initial, initial, 10.0, regions=[1, 2])
    assert closed.maximum_normalized_imbalance == 0.0
    np.testing.assert_array_equal(closed.regional_imbalance, [[0.0], [0.0]])

    initial_driven = {
        "pressure": np.ones(2), "qWs": np.zeros(1),
        "pore_volume": np.array([2.0, 3.0]),
    }
    final = {
        "pressure": np.ones(2), "qWs": np.array([0.1]),
        "pore_volume": np.array([2.5, 3.5]),
    }
    # Increase surface inventory by one m3 and inject 0.1 m3/s for ten seconds.
    driven = conservation_report(adapter, initial_driven, final, 10.0)
    np.testing.assert_allclose(driven.change, [1.0])
    np.testing.assert_allclose(driven.wells, [1.0])
    np.testing.assert_allclose(driven.imbalance, [0.0])


def test_oil_water_inventory_and_signed_well_terms_are_explicit():
    fluid = init_simple_adi_fluid(mu=[1.0, 2.0], rho=[1000.0, 800.0])
    model = SimpleNamespace(pv=np.array([10.0, 20.0]), fluid=fluid)
    adapter = OilWaterConservationAdapter(model)
    initial = OWState(
        pressure=np.ones(2), sW=np.array([0.2, 0.4]),
        qWs=np.zeros(2), qOs=np.zeros(2), bhp=np.zeros(2),
    )
    final = initial.copy()
    final.sW = np.array([0.3, 0.35])  # water +0, oil -0 globally
    final.qWs = np.array([0.2, -0.2])
    final.qOs = np.array([0.1, -0.1])
    report = conservation_report(adapter, initial, final, 5.0)
    np.testing.assert_allclose(report.change, [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(report.wells, [0.0, 0.0])
    assert adapter.units == "surface_m3"
    assert adapter.sign_convention == "positive_into_reservoir"


def test_boundary_and_source_terms_use_same_positive_into_reservoir_basis():
    fluid = init_simple_adi_fluid(mu=[1.0, 2.0], rho=[1000.0, 800.0])
    model = SimpleNamespace(pv=np.array([1.0]), raw_fluid=fluid)
    adapter = WaterConservationAdapter(model)
    initial = {
        "pressure": np.ones(1), "qWs": np.zeros(0),
        "pore_volume": np.array([1.0]),
    }
    final = {
        "pressure": np.ones(1), "qWs": np.zeros(0),
        "pore_volume": np.array([1.6]),
    }
    report = conservation_report(
        adapter, initial, final, 2.0,
        boundary_rates=[0.2], source_rates=[0.1],
    )
    np.testing.assert_allclose(report.change, [0.6])
    np.testing.assert_allclose(report.imbalance, [0.0], atol=1e-15)


def test_invariants_observe_without_mutation_and_strict_raises():
    state = {"pressure": np.array([1.0e7, np.nan]), "sW": np.array([0.2, 1.1])}
    original_saturation = state["sW"].copy()
    suite = InvariantSuite([
        finite_field("pressure"), bounded_field("sW", 0.0, 1.0, 1e-8),
    ])
    results = suite.evaluate(state, "observe")
    assert len(results) == 2 and not all(result.passed for result in results)
    np.testing.assert_array_equal(state["sW"], original_saturation)
    with pytest.raises(InvariantViolation) as error:
        suite.evaluate(state, "strict")
    assert len(error.value.failures) == 2
    # Reference records the same evidence without changing numerical policy.
    assert suite.evaluate(state, "reference") == results


def test_session_records_conservation_and_invariant_evidence_at_boundaries():
    from dataclasses import dataclass

    @dataclass
    class Step:
        converged: bool = True
        iterations: int = 1
        dt: float = 2.0
        residuals: list = None
        cuts: int = 0
        line_searches: int = 0
        ministeps: list = None

        def __post_init__(self):
            self.residuals = [] if self.residuals is None else self.residuals
            self.ministeps = [] if self.ministeps is None else self.ministeps

    class Report:
        steps = [Step()]
        converged = True

    class Solver:
        def simulate_schedule(self, model, state, schedule):
            return [{
                "pressure": np.ones(1), "qWs": np.array([0.25]),
                "pore_volume": np.array([1.5]), "sW": np.array([1.0]),
            }], Report()

    fluid = init_simple_adi_fluid(mu=[1.0, 2.0], rho=[1000.0, 800.0])
    adapter = WaterConservationAdapter(
        SimpleNamespace(pv=np.array([1.0]), raw_fluid=fluid),
    )
    initial = {
        "pressure": np.ones(1), "qWs": np.zeros(1),
        "pore_volume": np.array([1.0]), "sW": np.array([1.0]),
    }
    result = SimulationSession(
        model="model", initial_state=initial, schedule=[2.0],
        nonlinear_solver=Solver(), conservation_adapter=adapter,
        invariant_suite=InvariantSuite([
            finite_field("pressure"), bounded_field("sW", 0.0, 1.0),
        ]),
    ).run()
    step = result.evidence.conservation["steps"][0]
    np.testing.assert_allclose(step["change"], [0.5])
    np.testing.assert_allclose(step["wells"], [0.5])
    assert step["maximum_normalized_imbalance"] == 0.0
    assert all(item["passed"] for item in result.evidence.invariants)
    assert not any(event.kind is EventKind.INVARIANT_VIOLATED for event in result.events)


def _pseudo_component_model(pore_volume=np.array([2.0, 3.0])):
    """A converted black-oil model, shaped the way the real one is.

    The wrapper `convertBlackOilModelToCompositionalModel` returns spells its
    mixture `mixture` rather than `mix` and keeps the black-oil model it came
    from; the pore volume and surface densities live on that inner model.
    """
    return SimpleNamespace(
        mixture=SimpleNamespace(names=("PseudoOil", "PseudoGas"), n=2),
        blackoil_model=SimpleNamespace(
            pv=pore_volume,
            fluid=SimpleNamespace(rho_s=np.array([1000.0, 800.0, 1.0])),
        ),
    )


def _pseudo_component_state(*, s_oil, s_gas, xo, yo, qOs=0.0, qGs=0.0):
    return {
        "pressure": np.ones(np.size(s_oil)),
        "sO": np.asarray(s_oil, dtype=float),
        "sG": np.asarray(s_gas, dtype=float),
        "qOs": np.atleast_1d(qOs), "qGs": np.atleast_1d(qGs),
        "blackoil_mass_fractions": SimpleNamespace(
            xo=np.asarray(xo, dtype=float), xg=1.0-np.asarray(xo, dtype=float),
            yo=np.asarray(yo, dtype=float), yg=1.0-np.asarray(yo, dtype=float),
            rhoO=np.array([600.0, 650.0]), rhoG=np.array([80.0, 90.0]),
        ),
    }


def test_pseudo_component_adapter_uses_the_documented_mass_basis():
    adapter = BlackOilPseudoComponentConservationAdapter(_pseudo_component_model())
    state = _pseudo_component_state(
        s_oil=[0.7, 0.4], s_gas=[0.3, 0.6], xo=[0.8, 0.6], yo=[0.3, 0.1],
    )
    expected = np.array([2.0, 3.0])[:, None]*(
        (np.array([600.0, 650.0])*np.array([0.7, 0.4]))[:, None]
        * np.array([[0.8, 0.2], [0.6, 0.4]])
        + (np.array([80.0, 90.0])*np.array([0.3, 0.6]))[:, None]
        * np.array([[0.3, 0.7], [0.1, 0.9]])
    )
    np.testing.assert_allclose(adapter.inventory(state), expected)
    assert adapter.component_names == ("PseudoOil", "PseudoGas")
    assert adapter.units == "kg"


def test_pseudo_component_well_rates_convert_surface_streams_by_component():
    """The mapping that makes the SPE1 balance close.

    Surface oil *is* `PseudoOil` and surface gas *is* `PseudoGas`, so the two
    surface densities must not be interchanged. `rho_s` is ordered
    (water, oil, gas), so picking the wrong index is a live mistake: the water
    density here is close enough to the oil density to look plausible and is
    still wrong by 25%.
    """
    adapter = BlackOilPseudoComponentConservationAdapter(_pseudo_component_model())
    state = _pseudo_component_state(
        s_oil=[1.0, 1.0], s_gas=[0.0, 0.0], xo=[1.0, 1.0], yo=[0.0, 0.0],
        qOs=[2.0, -0.5], qGs=[7.0, -3.0],
    )
    np.testing.assert_allclose(adapter.well_rates(state), [1.5*800.0, 4.0*1.0])


def test_pseudo_component_adapter_rejects_a_model_it_cannot_read():
    """The failure that motivated this adapter, kept loud.

    `CompositionalConservationAdapter` reads `mix`; this wrapper spells it
    `mixture`. An adapter that quietly accepted either name for any mixture
    would conserve two arbitrary components under black-oil pseudo-component
    labels, so the mixture identity is checked rather than assumed.
    """
    model = _pseudo_component_model()
    model.mixture = SimpleNamespace(names=("C1", "C3"), n=2)
    with pytest.raises(TypeError, match="PseudoOil"):
        BlackOilPseudoComponentConservationAdapter(model)

    missing = _pseudo_component_model()
    del missing.blackoil_model
    with pytest.raises(TypeError, match="blackoil_model"):
        BlackOilPseudoComponentConservationAdapter(missing)


def test_pseudo_component_transfer_between_phases_conserves_both_components():
    """Vaporizing oil moves mass between phases without creating any.

    Densities and compositions are equal across the phases so that only the
    phase split changes; the components' totals must not move.
    """
    adapter = BlackOilPseudoComponentConservationAdapter(_pseudo_component_model())
    common = {"xo": [0.5, 0.5], "yo": [0.5, 0.5]}
    initial = _pseudo_component_state(s_oil=[0.9, 0.9], s_gas=[0.1, 0.1], **common)
    final = _pseudo_component_state(s_oil=[0.2, 0.2], s_gas=[0.8, 0.8], **common)
    for state in (initial, final):
        state["blackoil_mass_fractions"].rhoG = state["blackoil_mass_fractions"].rhoO
    report = conservation_report(adapter, initial, final, 2.0)
    np.testing.assert_allclose(report.change, np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(report.imbalance, np.zeros(2), atol=1e-12)


def test_oil_water_adapter_resolves_the_fluid_its_sibling_does():
    """A poromechanics oil-water model carries two fluids, and only one works.

    `fluid` is a `PhaseFluid` whose b-factors are indexed per phase and which
    has no `b_w`; `raw_fluid` is the `ADFluid` that does. Reading `model.fluid`
    directly raised AttributeError on precisely the coupled models
    `PoromechanicsConservationAdapter` routes here.
    """
    class PhaseFluidWithoutBW:
        def b(self, phase, pressure):
            raise AssertionError("the per-phase fluid must not be used here")

    model = SimpleNamespace(
        pv=np.array([2.0, 4.0]),
        fluid=PhaseFluidWithoutBW(),
        raw_fluid=SimpleNamespace(
            b_w=lambda p: np.full(np.size(p), 1.25),
            b_o=lambda p: np.full(np.size(p), 0.8),
        ),
    )
    state = {
        "pressure": np.full(2, 2.7e7), "sW": np.array([0.75, 0.5]),
        "qWs": np.array([3.0]), "qOs": np.array([-1.0]),
    }
    adapter = OilWaterConservationAdapter(model)
    np.testing.assert_allclose(
        adapter.inventory(state),
        np.column_stack((
            np.array([2.0, 4.0])*1.25*np.array([0.75, 0.5]),
            np.array([2.0, 4.0])*0.8*np.array([0.25, 0.5]),
        )),
    )
    np.testing.assert_allclose(adapter.well_rates(state), [3.0, -1.0])


def test_oil_water_adapter_refuses_a_fluid_with_no_water_b_factor():
    """Missing b_w must raise, not silently become one.

    A unit b-factor would produce a plausible-looking imbalance derived from a
    property the model never supplied - the failure mode this whole gate is
    meant to detect rather than emit.
    """
    model = SimpleNamespace(
        pv=np.array([1.0]),
        raw_fluid=SimpleNamespace(b_o=lambda p: np.ones(np.size(p))),
    )
    state = {"pressure": np.ones(1), "sW": np.array([0.5])}
    with pytest.raises(AttributeError, match="b_w"):
        OilWaterConservationAdapter(model).inventory(state)


def test_water_b_falls_back_to_the_fluid_when_a_model_has_no_graph():
    """The other half of the shrinkage-factor lookup.

    Models without a state-function graph - the plain black-oil and MRST-shaped
    ones - carry `b_w` on a raw fluid instead, and that path had no coverage:
    disabling it left every test passing. Its sibling defect is what this whole
    lookup exists to prevent, so both branches are pinned.
    """
    model = SimpleNamespace(
        pv=np.array([2.0, 4.0]),
        raw_fluid=SimpleNamespace(b_w=lambda p: np.full(np.size(p), 1.25)),
    )
    state = {"pressure": np.full(2, 3.0e7), "qWs": np.array([7.0])}
    adapter = WaterConservationAdapter(model)
    np.testing.assert_allclose(adapter.inventory(state),
                               np.array([[2.0*1.25], [4.0*1.25]]))
    np.testing.assert_allclose(adapter.well_rates(state), [7.0])


def test_water_refuses_a_model_that_defines_b_nowhere():
    """An incompressible fluid and a b that could not be found look identical.

    This returned ones until that silence put 5.25e-07 on the record as
    poromechanical coupling. Every sibling adapter refuses a model it cannot
    read, no case in the characterization set reaches this branch, and a model
    that really is incompressible can say so by exposing a b of one.
    """
    model = SimpleNamespace(pv=np.array([3.0]), raw_fluid=SimpleNamespace())
    state = {"pressure": np.ones(1), "qWs": np.zeros(1)}
    with pytest.raises(AttributeError, match="refusing to assume"):
        WaterConservationAdapter(model).inventory(state)

    # ...and an explicit unit b is accepted, so the refusal is about silence
    # rather than about incompressible fluids.
    explicit = SimpleNamespace(
        pv=np.array([3.0]),
        raw_fluid=SimpleNamespace(b_w=lambda p: np.ones(np.size(p))),
    )
    np.testing.assert_allclose(
        WaterConservationAdapter(explicit).inventory(state), np.array([[3.0]]))
