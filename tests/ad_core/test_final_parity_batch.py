"""The last of MRST's `ad_core` methods: aquifers, relperm, tables, seams.

Twelve small methods that finish the module. They have little in common except
that each is a place where MRST's structure carries information PyMRST's
reorganisation had dropped - a lookup by name instead of index, a seam a
subclass is meant to override, a cached interpolant that must be rebuilt.

The recurring failure mode in this batch is *plausible staleness*: a table
swapped under its interpolant, a flux stored per connection instead of per
aquifer, an equation found by position in a model that reorders its equations.
None of them raise. Each test names the wrong answer it prevents.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import (
    DiagonalJacobian,
)
from pymrst.modules.ad_core.models.aquifers.aquifer import (
    AquiferConnections, AquiferModelAD,
)
from pymrst.modules.ad_core.models.facilities.vfp.vfp_table import VFPTable
from pymrst.params.wells_and_bc.multisegment import segment_upstream_value


# -- aquifers ---------------------------------------------------------------

def _aquifer(n_aq=2):
    conn = AquiferConnections(
        aquid=np.array([0, 0, 1]), cells=np.array([0, 1, 2]),
        j=np.full(3, 1e-10), c=np.full(3, 1e-9), alpha=np.ones(3),
        depth_conn=np.zeros(3), depth_aq=np.zeros(3),
        c_aq=np.full(n_aq, 1e-9))
    return AquiferModelAD(conn, np.array([2.0e7, 2.5e7]),
                        np.array([1.0e6, 2.0e6]))


def _aq_model(n_cells=3):
    """The smallest model the aquifer flux needs."""
    return SimpleNamespace(
        fluid=SimpleNamespace(rho_s=np.array([1000.0, 800.0])),
        gravity_magnitude=0.0,
        get_props=lambda st, name: (
            [np.full(n_cells, 1.9e7), np.full(n_cells, 1.9e7)]
            if name == "PhasePressures"
            else [np.ones(n_cells), np.ones(n_cells)]),
        _basis_scale=lambda: np.array([1.0, 1.0]))


def test_the_aquifer_state_carries_both_representations():
    """The flat arrays drive the solve; `aquiferSol` additionally holds the
    flux, which cannot be recovered from pressure and volume afterwards."""
    aq = _aquifer()
    state = aq.init_state_aquifer({})

    np.testing.assert_allclose(state["aquifer_p"], [2.0e7, 2.5e7])
    assert len(state["aquiferSol"]) == 2
    assert state["aquiferSol"][0]["pressure"] == 2.0e7
    assert state["aquiferSol"][0]["flux"] is None


def test_the_influx_goes_to_the_named_water_equation():
    """`add_contribution` indexes slot zero; this finds "water" by name.

    A compositional or thermal model orders its equations differently, and
    indexing then subtracts the water influx from whatever sits first - a
    per-phase conservation error that balances globally.
    """
    aq = _aquifer()
    model = _aq_model()
    state = aq.init_state_aquifer({})

    eqs = [np.zeros(3), np.zeros(3)]
    out = aq.add_aquifers_contribution(model, eqs, ["energy", "water"],
                                       state, 1.0)
    assert np.all(out[0] == 0.0), "the influx went to the wrong equation"
    assert np.any(out[1] != 0.0), "the water equation received nothing"


def test_a_model_without_water_is_left_alone():
    """No water equation is a configuration, not a fault."""
    aq = _aquifer()
    eqs = [np.zeros(3)]
    out = aq.add_aquifers_contribution(_aq_model(), eqs, ["oil"],
                                       aq.init_state_aquifer({}), 1.0)
    np.testing.assert_array_equal(out[0], 0.0)


def test_fluxes_are_recorded_per_aquifer():
    """Aquifer 0 feeds two cells and aquifer 1 feeds one. A report wants each
    aquifer's own contribution, so the split must survive."""
    aq = _aquifer()
    state = aq.init_state_aquifer({})
    out = aq.update_aquifer_fluxes(_aq_model(), state, 1.0)

    assert out["aquiferSol"][0]["flux"].size == 2
    assert out["aquiferSol"][1]["flux"].size == 1


# -- relative permeability --------------------------------------------------

class _RelPerm:
    """Only the pieces `rel_perm_unified` and the chop touch."""
    from pymrst.modules.ad_core.statefunctions.flowprops.base_relative_permeability import (  # noqa: E501
        BaseRelativePermeability as _Base,
    )
    rel_perm_unified = _Base.rel_perm_unified
    apply_immobile_chop = _Base.apply_immobile_chop

    def __init__(self, immobile_chop=False, critical=0.1):
        self.immobile_chop = immobile_chop
        self._critical = critical

    def evaluate_phase_relative_permeability(self, model, phase, s):
        return {"W": 1.0, "O": 2.0, "G": 3.0}[phase]*np.asarray(s)

    def get_critical_phase_saturations(self, model, state):
        n = np.asarray(state["s"]).shape[0]
        return np.full((n, np.asarray(state["s"]).shape[1]), self._critical)


def test_the_unified_path_asks_every_phase_for_its_own_curve():
    model = SimpleNamespace(
        get_phase_names=lambda: ["W", "O"],
        get_prop=lambda st, name: st[name])
    state = {"sW": np.array([0.3, 0.4]), "sO": np.array([0.7, 0.6])}

    kr = _RelPerm().rel_perm_unified(model, state)
    np.testing.assert_allclose(kr[0], [0.3, 0.4])
    np.testing.assert_allclose(kr[1], [1.4, 1.2])


def test_the_chop_is_off_unless_asked_for():
    """It perturbs saturations, so it must be requested."""
    prop = _RelPerm(immobile_chop=False)
    state = {"s": np.array([[0.1, 0.9]])}
    out, chopped = prop.apply_immobile_chop(None, state, {"s": np.array([[0.05, 0.95]])})
    assert chopped is False
    np.testing.assert_array_equal(out["s"], [[0.1, 0.9]])


def test_a_cell_crossing_the_critical_saturation_is_nudged_past_it():
    """At exactly the critical saturation kr and dkr are both zero, so a
    Newton step cannot move the cell in either direction. It stays stuck at
    the boundary whatever the driving force."""
    prop = _RelPerm(immobile_chop=True, critical=0.1)
    state = {"s": np.array([[0.1, 0.9]])}          # sitting on the boundary
    state0 = {"s": np.array([[0.05, 0.95]])}       # arriving from below

    out, chopped = prop.apply_immobile_chop(None, state, state0)
    assert np.any(chopped)
    assert out["s"][0, 0] > 0.1, "the cell was left on the critical value"


def test_a_cell_becoming_immobile_is_nudged_below_the_boundary():
    """The other direction, and it needs its own case: a cell arriving from
    the mobile side and landing on the critical value is just as stuck as one
    arriving from below, and a chop that only handles one direction leaves
    half of them there."""
    prop = _RelPerm(immobile_chop=True, critical=0.1)
    state = {"s": np.array([[0.1, 0.9]])}          # on the boundary
    state0 = {"s": np.array([[0.2, 0.8]])}         # arriving from above

    out, chopped = prop.apply_immobile_chop(None, state, state0)
    assert np.any(chopped)
    assert out["s"][0, 0] < 0.1, "the cell was left on the critical value"


def test_a_cell_far_from_the_critical_saturation_is_untouched():
    prop = _RelPerm(immobile_chop=True, critical=0.1)
    state = {"s": np.array([[0.6, 0.4]])}
    out, chopped = prop.apply_immobile_chop(
        None, state, {"s": np.array([[0.5, 0.5]])})
    np.testing.assert_allclose(out["s"], [[0.6, 0.4]])
    assert not np.any(chopped)


# -- the explicit-flux builder ----------------------------------------------

def test_explicit_groups_report_what_was_asked_for():
    """Filtering to what the model defines would make a misspelled property
    name look like a supported configuration."""
    from pymrst.modules.ad_core.statefunctions.flux.flow_state.explicit_flow_state_builder import (  # noqa: E501
        ExplicitFlowStateBuilder,
    )
    builder = ExplicitFlowStateBuilder(explicit_flux=["FaceMobility"],
                                       explicit_pvt=["ShrinkageFactors"])
    model = SimpleNamespace(PVTPropertyFunctions="pvt",
                            FlowPropertyFunctions="flow",
                            FlowDiscretization="flux")

    groups, names = builder.get_explicit_groups(model)
    assert groups == ["pvt", "flux"]
    assert names == [["ShrinkageFactors"], ["FaceMobility"]]


def test_an_undeclared_group_is_omitted():
    from pymrst.modules.ad_core.statefunctions.flux.flow_state.explicit_flow_state_builder import (  # noqa: E501
        ExplicitFlowStateBuilder,
    )
    builder = ExplicitFlowStateBuilder(explicit_flux=[], explicit_flow=[],
                                       explicit_pvt=[])
    groups, names = builder.get_explicit_groups(SimpleNamespace())
    assert groups == [] and names == []


def test_the_first_timestep_is_the_one_with_no_flux():
    from pymrst.modules.ad_core.statefunctions.flux.flow_state.explicit_flow_state_builder import (  # noqa: E501
        ExplicitFlowStateBuilder,
    )
    builder = ExplicitFlowStateBuilder()
    assert builder.is_first_time_step({}) is True
    assert builder.is_first_time_step({"flux": np.zeros(4)}) is True
    assert builder.is_first_time_step({"flux": np.array([0.0, 1.0])}) is False


# -- the flow-state seam ----------------------------------------------------

def test_the_flow_state_comes_from_the_builder():
    """Calling the properties directly instead hard-wires the implicit
    choice, and the explicit/adaptive builders stop having any effect."""
    from pymrst.modules.ad_core.statefunctions.flow_discretization import (
        FlowDiscretization,
    )
    fd = FlowDiscretization.__new__(FlowDiscretization)
    fd.FlowStateBuilder = SimpleNamespace(
        build=lambda model, state, state0, dt: "built")
    assert fd.build_flow_state(None, {"a": 1}, {"a": 0}, 1.0) == "built"


# -- components -------------------------------------------------------------

def test_the_base_component_refuses_an_injection_fraction():
    """Returning zero would silently inject pure carrier fluid while
    reporting a component injector."""
    from pymrst.modules.ad_core.statefunctions.components.components import (
        ConcentrationComponent,
    )
    component = ConcentrationComponent.__new__(ConcentrationComponent)
    with pytest.raises(NotImplementedError, match="injection mass"):
        component.get_injection_mass_fraction(None, None)


# -- multisegment topology --------------------------------------------------

def test_segment_upstream_takes_the_neighbour_the_flow_comes_from():
    topo = np.array([[0, 2], [1, 3], [2, 4]])
    val = np.array([10.0, 20.0, 30.0, 40.0])

    from_inlet = segment_upstream_value(np.array([True]*3), val, topo)
    from_outlet = segment_upstream_value(np.array([False]*3), val, topo)
    np.testing.assert_allclose(from_outlet, [20.0, 30.0, 40.0])
    # The top segment has no inlet, so it falls back to its own outlet
    # rather than reading segment zero.
    assert from_inlet[0] == 20.0
    np.testing.assert_allclose(from_inlet[1:], [10.0, 20.0])


def test_a_zero_marker_cannot_wrap_to_the_last_segment():
    """`index - 1` on a surviving 0 marker is -1, which indexes the last entry.

    The marker has to be on a segment *other* than the first: the top
    segment's 0 is substituted away by the rule above, so putting the test
    case there checks the substitution instead of the floor.
    """
    topo = np.array([[1, 2], [0, 3]])
    val = np.array([5.0, 6.0, 7.0])
    out = segment_upstream_value(np.array([False, True]), val, topo)
    assert out[1] == 5.0, "the 0 marker wrapped to the last segment"


# -- crossflow --------------------------------------------------------------

def test_an_inexpressible_mixing_rule_is_refused():
    """Mass-fraction mixing needs a quantity this basis does not carry;
    substituting the volume rule would give a different composition under the
    same name."""
    from pymrst.modules.ad_core.statefunctions.facility.well_component_total_volume_balance_crossflow import (  # noqa: E501
        WellComponentTotalVolumeBalanceCrossflow as X,
    )
    prop = X.__new__(X)
    prop.mass_type = "surfacevolume"
    with pytest.raises(NotImplementedError, match="surface volume"):
        prop.compute_cross_flux(None, None, None,
                                mass_type="exactmassfractions")


# -- diagonal storage -------------------------------------------------------

def test_setting_the_diagonal_writes_only_stored_entries():
    """Creating an entry where the pattern has none would change the sparsity
    between iterations, and a symbolic factorisation is built on the pattern."""
    jac = DiagonalJacobian(np.array([1.0, 2.0, 3.0]),
                           np.array([0, 1, 1]), np.array([0, 1, 2]), (3, 3))
    before = jac.nnz
    jac.set_diagonal(9.0)

    assert jac.nnz == before, "the sparsity pattern changed"
    np.testing.assert_allclose(jac.to_sparse().diagonal(), [9.0, 9.0, 0.0])
    # The (1, 2) entry is stored but is not on the diagonal, and must not be
    # written - checking only nnz and the diagonal cannot tell the difference.
    assert jac.to_sparse().toarray()[1, 2] == pytest.approx(3.0)


def test_setting_the_diagonal_can_be_restricted_to_rows():
    jac = DiagonalJacobian(np.array([1.0, 2.0]), np.array([0, 1]),
                           np.array([0, 1]), (2, 2))
    jac.set_diagonal(7.0, subs=np.array([1]))
    np.testing.assert_allclose(jac.to_sparse().diagonal(), [1.0, 7.0])


# -- VFP tables -------------------------------------------------------------

def _vfp():
    flow = np.array([0.0, 1.0])
    thp = np.array([1.0e6, 2.0e6])
    return VFPTable(flow, thp, [], [], np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_updating_a_table_rebuilds_its_interpolant():
    """The interpolant caches its axes. A table swapped underneath it keeps
    answering from the previous grid - same shape, plausible numbers, stale.
    """
    table = _vfp()
    before = table.evaluate_bhp(0.0, 1.0e6)

    table.update_table(np.array([0.0, 1.0]), np.array([1.0e6, 2.0e6]),
                       [], [], np.array([[10.0, 20.0], [30.0, 40.0]]))
    after = table.evaluate_bhp(0.0, 1.0e6)

    assert before == pytest.approx(1.0)
    assert after == pytest.approx(10.0), "the interpolant was not rebuilt"


def test_loading_a_deck_record_replaces_the_table():
    table = _vfp()
    table.set_table_from_deck({"FLO": np.array([0.0, 1.0]),
                               "THP": np.array([1.0e6, 2.0e6]),
                               "BHP": np.array([[5.0, 6.0], [7.0, 8.0]]),
                               "FLOID": "LIQ"})
    assert table.flow_type == "liq"
    assert table.evaluate_bhp(0.0, 1.0e6) == pytest.approx(5.0)


# -- hysteresis -------------------------------------------------------------

class _Hyst:
    def __call__(self, sg, sg_max):
        return np.asarray(sg)*0.5

    @staticmethod
    def trapped_saturation(sg_max):
        return np.asarray(sg_max)*0.2


def _hysteretic(hysteresis=None):
    from pymrst.modules.ad_core.statefunctions.flowprops.hysteresis.hysteretic_relative_permeability import (  # noqa: E501
        HystereticRelativePermeability as H,
    )
    prop = H.__new__(H)
    prop.hysteresis = hysteresis
    prop.sat_max_key = "sGMax"
    prop.evaluate_phase_relative_permeability = \
        lambda model, phase, s: np.asarray(s)*10.0
    prop.evaluate_gas_rel_perm = \
        lambda model, sg, state: prop.hysteresis(sg, state["sGMax"])
    return prop


def test_only_the_gas_phase_takes_the_imbibition_branch():
    """Gas trapped behind an advancing front stays trapped; water and oil
    take the ordinary drainage curve, and asking for them is the common case
    in a three-phase model."""
    prop = _hysteretic(_Hyst())
    state = {"sGMax": np.array([0.5])}
    sg = np.array([0.4])

    water = prop.evaluate_phase_relative_permeability_with_hysteresis(
        None, "W", sg, state)
    gas = prop.evaluate_phase_relative_permeability_with_hysteresis(
        None, "G", sg, state)
    np.testing.assert_allclose(water, [4.0])
    np.testing.assert_allclose(gas, [0.2])


def test_the_scanning_curve_reports_its_reversal_point():
    """Two cells at the same gas saturation have different kr if they arrived
    from different maxima - which is why `sGMax` is carried on the state."""
    kr, trapped = _hysteretic(_Hyst()).scan_rel_perm(np.array([0.4]),
                                                     np.array([0.5]))
    np.testing.assert_allclose(kr, [0.2])
    np.testing.assert_allclose(trapped, [0.1])


def test_a_scanning_curve_without_a_hysteresis_model_is_refused():
    """Returning the drainage value would report a hysteretic result from a
    non-hysteretic model."""
    with pytest.raises(ValueError, match="hysteresis model"):
        _hysteretic(None).scan_rel_perm(np.array([0.4]), np.array([0.5]))
