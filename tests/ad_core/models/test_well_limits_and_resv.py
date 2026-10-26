"""MRST `GenericFacilityModel`'s control switching and RESV conversion.

**Control switching.** A well runs on one control and carries limits on the
others. A producer on a rate target whose pressure falls below its minimum
stops being a rate well and becomes a BHP well. Skipping this does not fail -
the well simply produces at a rate the reservoir cannot support, and the run
completes with a plausible profile no facility could have delivered.

Three details carry the behaviour and each has its own test:

* the **defaults** for unset limits are `+inf` for an injector and `-inf` for a
  producer. Backwards, every unset limit reads as violated on the first check;
* the well's **current mode is never a switch candidate** - a well sitting
  exactly on its BHP limit satisfies `bhp <= lims.bhp` as an equality and would
  otherwise switch to the control it is already on, every step;
* after a switch the **stored rates are reset**, or the next Newton iteration
  starts from the previous control's solution.

**RESV.** A `resv` well is controlled on volume at *reservoir* conditions while
the well equations are in surface volume. The conversion needs shrinkage
factors at the well's own averaged pressure and dissolved ratios, per PVT
region - which is why the averages are pore-volume weighted and taken region by
region.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.models.facilities.generic_facility_model import (
    GenericFacilityModel,
)


def _facility(wells, phases=("W", "O"), **model_kw):
    """A facility with `WellModels` attached and nothing else built."""
    fm = GenericFacilityModel.__new__(GenericFacilityModel)
    fm.WellModels = list(wells)
    fm.ReservoirModel = SimpleNamespace(
        get_phase_names=lambda: list(phases),
        get_phase_index=lambda n: list(phases).index(n),
        **model_kw)
    return fm


def _well(sign=-1.0, lims=None, status=True, switching=True, compi=(1.0, 0.0),
          well_type="rate", val=-1.0):
    W = SimpleNamespace(lims=dict(lims or {}), status=status, sign=sign,
                        compi=np.array(compi, dtype=float), type=well_type,
                        val=val, cells=np.array([0]))
    return SimpleNamespace(W=W, allow_control_switching=switching)


def _sol(bhp=2.0e7, qWs=-1.0, qOs=-2.0, type="rate", val=-3.0, **extra):
    return dict(bhp=bhp, qWs=qWs, qOs=qOs, type=type, val=val, **extra)


# -- filling in unset limits ------------------------------------------------

def test_unset_limits_take_the_supplied_default():
    filled = GenericFacilityModel.set_missing_limits(
        {"bhp": 1.0e7}, ["bhp", "orat", "vrat"], -np.inf)
    assert filled["bhp"] == 1.0e7
    assert filled["orat"] == -np.inf and filled["vrat"] == -np.inf


def test_filling_limits_does_not_mutate_the_wells_own_dict():
    """The well's limits are reused every step; filling them in place would
    make the second step see defaults as if they had been configured."""
    original = {"bhp": 1.0e7}
    GenericFacilityModel.set_missing_limits(original, ["bhp", "orat"], -np.inf)
    assert original == {"bhp": 1.0e7}


# -- the guards -------------------------------------------------------------

def test_a_well_pinned_to_its_control_is_not_switched():
    fm = _facility([_well(lims={"bhp": 3.0e7}, switching=False)])
    sol = _sol(bhp=1.0e7)
    assert fm.apply_well_limits_well_sol(fm.WellModels[0], sol)["type"] == "rate"


def test_a_shut_well_is_not_switched():
    fm = _facility([_well(lims={"bhp": 3.0e7}, status=False)])
    sol = _sol(bhp=1.0e7)
    assert fm.apply_well_limits_well_sol(fm.WellModels[0], sol)["type"] == "rate"


def test_a_well_without_limits_is_left_alone():
    fm = _facility([_well(lims={})])
    sol = _sol(bhp=1.0e7)
    assert fm.apply_well_limits_well_sol(fm.WellModels[0], sol)["type"] == "rate"


# -- switching --------------------------------------------------------------

def test_a_producer_below_its_pressure_limit_switches_to_bhp():
    """The canonical case: the reservoir cannot support the rate target."""
    fm = _facility([_well(sign=-1.0, lims={"bhp": 2.5e7})])
    out = fm.apply_well_limits_well_sol(fm.WellModels[0], _sol(bhp=2.0e7))
    assert out["type"] == "bhp"
    assert out["val"] == 2.5e7
    assert out["bhp"] == 2.5e7


def test_a_producer_above_its_pressure_limit_keeps_its_control():
    fm = _facility([_well(sign=-1.0, lims={"bhp": 1.0e7})])
    out = fm.apply_well_limits_well_sol(fm.WellModels[0], _sol(bhp=2.0e7))
    assert out["type"] == "rate"


def test_an_injector_within_its_limits_keeps_its_control():
    """Guards the *default* for an injector's unset limits.

    Only `bhp` is configured and it is not violated, so nothing should switch.
    With `-inf` defaults the unset `rate` limit reads as violated immediately
    and the well switches to a control nobody configured.

    The well is on **bhp** control here, which is the point: on `rate` control
    the wrongly-violated `rate` limit is the well's own mode and is excluded,
    so the defect hides behind the exclusion. Two guards can mask each other,
    and this one had to be arranged so only one is in play.
    """
    fm = _facility([_well(sign=1.0, lims={"bhp": 3.0e7}, well_type="bhp")])
    out = fm.apply_well_limits_well_sol(
        fm.WellModels[0], _sol(bhp=2.0e7, qWs=1.0, qOs=0.0, type="bhp"))
    assert out["val"] == -3.0, "an unset limit was treated as violated"


def test_an_injector_above_its_pressure_limit_switches():
    """The inequality points the other way for an injector: it is limited from
    above, a producer from below. One shared comparison would be wrong for one
    of them in every case."""
    fm = _facility([_well(sign=1.0, lims={"bhp": 2.5e7}, well_type="rate")])
    out = fm.apply_well_limits_well_sol(fm.WellModels[0],
                                        _sol(bhp=3.0e7, qWs=1.0, qOs=0.0))
    assert out["type"] == "bhp" and out["val"] == 2.5e7


def test_a_well_is_never_switched_to_the_control_it_is_on():
    """A BHP well *below* its own BHP limit reads as violating it.

    The comparison must be strict inequality on a mode that is excluded, not
    equality on one that never fires: an earlier version of this test used
    `bhp == lims.bhp`, which is not a violation either way, so it passed
    whether or not the exclusion existed.
    """
    fm = _facility([_well(sign=-1.0, lims={"bhp": 2.0e7}, well_type="bhp")])
    out = fm.apply_well_limits_well_sol(fm.WellModels[0],
                                        _sol(bhp=1.9e7, type="bhp"))
    assert out["val"] == -3.0, "the well switched to the control it was on"


def test_the_first_violated_limit_wins():
    """MRST takes the first in its fixed order, so the order is behaviour."""
    fm = _facility([_well(sign=-1.0,
                          lims={"bhp": 2.5e7, "orat": -1.0})])
    out = fm.apply_well_limits_well_sol(fm.WellModels[0],
                                        _sol(bhp=2.0e7, qOs=-5.0))
    assert out["type"] == "bhp", "bhp precedes orat in MRST's order"


def test_switching_to_a_rate_splits_the_target_by_composition():
    """The next Newton iteration starts from these rates; leaving the old ones
    makes it solve the previous control's problem."""
    fm = _facility([_well(sign=1.0, lims={"rate": 4.0},
                          compi=(0.25, 0.75), well_type="bhp")])
    out = fm.apply_well_limits_well_sol(
        fm.WellModels[0], _sol(bhp=1.0e6, qWs=5.0, qOs=5.0, type="bhp"))
    assert out["type"] == "rate"
    assert out["qWs"] == pytest.approx(1.0)
    assert out["qOs"] == pytest.approx(3.0)


def test_switching_to_a_phase_rate_sets_that_phase():
    fm = _facility([_well(sign=-1.0, lims={"orat": -1.0}, well_type="bhp")])
    out = fm.apply_well_limits_well_sol(
        fm.WellModels[0], _sol(bhp=2.0e7, qOs=-5.0, type="bhp"))
    assert out["type"] == "orat" and out["qOs"] == pytest.approx(-1.0)


def test_a_summed_limit_gets_no_rate_guess():
    """`lrat` constrains a sum. Splitting it across the phases would invent a
    split the scheme never computed; MRST leaves the rates alone."""
    fm = _facility([_well(sign=-1.0, lims={"lrat": -1.0}, well_type="bhp")])
    out = fm.apply_well_limits_well_sol(
        fm.WellModels[0], _sol(bhp=2.0e7, qWs=-4.0, qOs=-4.0, type="bhp"))
    assert out["type"] == "lrat"
    assert out["qWs"] == -4.0 and out["qOs"] == -4.0


def test_apply_well_limits_covers_every_active_well():
    fm = _facility([_well(sign=-1.0, lims={"bhp": 2.5e7}),
                    _well(sign=-1.0, lims={"bhp": 2.5e7})])
    fm.get_indices_of_active_wells = lambda ws=None: np.array([0, 1])
    state = {"wellSol": [_sol(bhp=2.0e7), _sol(bhp=2.0e7)]}

    out = fm.apply_well_limits(state)
    assert [s["type"] for s in out["wellSol"]] == ["bhp", "bhp"]


def test_only_active_wells_are_touched():
    fm = _facility([_well(sign=-1.0, lims={"bhp": 2.5e7}),
                    _well(sign=-1.0, lims={"bhp": 2.5e7})])
    fm.get_indices_of_active_wells = lambda ws=None: np.array([0])
    state = {"wellSol": [_sol(bhp=2.0e7), _sol(bhp=2.0e7)]}

    out = fm.apply_well_limits(state)
    assert [s["type"] for s in out["wellSol"]] == ["bhp", "rate"]


# -- RESV -------------------------------------------------------------------

def _resv_facility(well_type="resv", val=-3.0, n_cells=4, regions=None):
    well = _well(sign=-1.0, well_type=well_type, val=val, compi=(0.5, 0.5))
    fm = _facility([well])
    fm.get_well_status_mask = lambda ws=None: np.array([True])

    b_w = np.full(n_cells, 1.05)
    b_o = np.full(n_cells, 1.20)

    class _Shrinkage:
        @staticmethod
        def evaluate_on_domain(model, state):
            n = np.size(state["pressure"])
            return [np.full(n, 1.05), np.full(n, 1.20)]

    groupings = SimpleNamespace(
        Density=SimpleNamespace(regions=regions),
        ShrinkageFactors=_Shrinkage())
    fm.ReservoirModel = SimpleNamespace(
        get_phase_names=lambda: ["W", "O"],
        get_phase_index=lambda n: ["W", "O"].index(n),
        disgas=False, vapoil=False,
        G=SimpleNamespace(cells=SimpleNamespace(num=n_cells)),
        PVTPropertyFunctions=groupings,
        get_prop=lambda st, name: (np.full(n_cells, 100.0) if name == "PoreVolume"
                                   else np.full(n_cells, 0.2)))
    return fm


def test_a_model_with_no_resv_wells_is_untouched():
    fm = _resv_facility(well_type="rate")
    state = {"wellSol": [_sol()]}
    _, out = fm.update_resv_controls(state, {"pressure": np.full(4, 2.0e7)})
    assert out["wellSol"][0]["type"] == "rate"


def test_a_history_well_becomes_an_ordinary_resv_well():
    """`resv_history` has its target computed here from historical surface
    rates; leaving the type alone would recompute it every step."""
    fm = _resv_facility(well_type="resv_history", val=-4.0)
    state = {"wellSol": [_sol(type="resv_history")]}
    _, out = fm.update_resv_controls(state, {"pressure": np.full(4, 2.0e7)})

    assert out["wellSol"][0]["type"] == "resv"
    assert fm.WellModels[0].W.type == "resv"


def test_the_converted_rate_is_the_reservoir_volume_of_the_surface_rates():
    """target * compi / b, summed over phases: surface volume divided by the
    shrinkage factor is reservoir volume."""
    fm = _resv_facility(well_type="resv_history", val=-4.0)
    state = {"wellSol": [_sol(type="resv_history")]}
    _, out = fm.update_resv_controls(state, {"pressure": np.full(4, 2.0e7)})

    expected = (-4.0*0.5)/1.05 + (-4.0*0.5)/1.20
    assert out["wellSol"][0]["val"] == pytest.approx(expected)
    # The well struct carries the target the next step reads; leaving it at
    # the raw surface value there converts nothing where it matters.
    assert fm.WellModels[0].W.val == pytest.approx(expected)


def test_the_control_density_is_the_per_phase_conversion_factor():
    """The well equations multiply their surface rates by this to get the
    reservoir volume the control is written in."""
    fm = _resv_facility()
    state = {"wellSol": [_sol(type="resv")]}
    fm.update_resv_controls(state, {"pressure": np.full(4, 2.0e7)})

    np.testing.assert_allclose(fm.WellModels[0].ControlDensity,
                               [1/1.05, 1/1.20])


def test_the_averaging_is_pore_volume_weighted_per_region():
    """Two regions with different pressures must not be averaged together -
    the shrinkage factor would come from a pressure neither region is at."""
    regions = np.array([1, 1, 2, 2])
    fm = _resv_facility(regions=regions)
    seen = {}

    original = fm.ReservoirModel.PVTPropertyFunctions.ShrinkageFactors

    class _Spy:
        @staticmethod
        def evaluate_on_domain(model, state):
            seen["pressure"] = np.asarray(state["pressure"]).copy()
            return original.evaluate_on_domain(model, state)

    fm.ReservoirModel.PVTPropertyFunctions.ShrinkageFactors = _Spy()
    state = {"wellSol": [_sol(type="resv")]}
    fm.update_resv_controls(
        state, {"pressure": np.array([1.0e7, 1.0e7, 9.0e7, 9.0e7])})

    # The well is in cell 0, which is region 1 - so it must see region 1's
    # average, not the mean over the whole grid.
    assert seen["pressure"][0] == pytest.approx(1.0e7)


# -- MRST's zeroTarget rule -------------------------------------------------
#
# A zero target on a surface-rate or reservoir control closes the well with the
# TOTAL rate, not with the single phase the mode names. The single-phase form
# is not merely less general - for a well that does not produce that phase it
# is the *same equation* as that phase's mass balance, which makes the Jacobian
# exactly singular. Norne's D-1H, switched onto `wrat = 0` while producing no
# water, is the case that found this.


def _rate_vars(n=1):
    from pymrst.utils.adi import init_variables_adi
    return init_variables_adi(*[np.zeros(n) for _ in range(4)])


def _ctrl_row(well_type, val, nph=3):
    """The control equation's derivatives w.r.t. (qWs, qOs, qGs, bhp)."""
    from pymrst.modules.ad_core.models.facilities.facility_model import (
        FacilityModel)
    qW, qO, qG, bhp = _rate_vars()
    w = SimpleNamespace(type=well_type, val=val, sign=-1.0,
                        allow_sign_change=False,
                        compi=np.array([0.0, 0.01, 0.99]))
    eq = FacilityModel._control_equation(w, [qW, qO, qG], bhp, nph)
    return np.array([float(j.toarray().ravel()[0]) for j in eq.jac])


def test_a_zero_rate_target_closes_on_the_total_rate():
    """MRST: `ctrl_eq(zeroRates) = q_t`, summed over every phase."""
    assert np.allclose(_ctrl_row("wrat", 0.0), [1.0, 1.0, 1.0, 0.0])
    assert np.allclose(_ctrl_row("orat", 0.0), [1.0, 1.0, 1.0, 0.0])
    assert np.allclose(_ctrl_row("grat", 0.0), [1.0, 1.0, 1.0, 0.0])
    assert np.allclose(_ctrl_row("resv", 0.0), [1.0, 1.0, 1.0, 0.0])


def test_a_nonzero_rate_target_still_uses_its_own_phase():
    """The substitution must fire on a zero target only - otherwise every
    single-phase control silently becomes a total-rate control."""
    assert np.allclose(_ctrl_row("wrat", -1.0), [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(_ctrl_row("orat", -1.0), [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(_ctrl_row("grat", -1.0), [0.0, 0.0, 1.0, 0.0])


def test_a_zero_bhp_target_is_not_substituted():
    """`zeroTarget` covers surface and reservoir controls, not BHP: a well
    held at zero pressure is a pressure well, and `qt` would not hold it."""
    row = _ctrl_row("bhp", 0.0)
    assert np.allclose(row, [0.0, 0.0, 0.0, 1.0]), row


def test_the_zero_target_row_is_independent_of_a_single_mass_balance():
    """The property that actually matters, stated directly.

    A well on `wrat = 0` that produces no water has the mass-balance row
    `q_Ws = 0`. If the control row were also `q_Ws`, the two are linearly
    dependent, the bottom-hole pressure is unconstrained, and the factorization
    is exactly singular at every timestep. Rank, not values, is the assertion.
    """
    control = _ctrl_row("wrat", 0.0)[:3]
    water_mass_balance = np.array([1.0, 0.0, 0.0])   # q_Ws - (no perforations)
    rank = np.linalg.matrix_rank(np.vstack([control, water_mass_balance]))
    assert rank == 2, (
        "control row duplicates the water mass balance: %s" % control)


# -- MRST's shut-well rule (GenericFacilityModel.prepareTimestep) ------------
#
#   ctrl = wm.W.type;
#   if ~(strcmpi(ctrl,'bhp') || strcmpi(ctrl,'thp'))
#       wellSol(wellNo).status = wm.W.val ~= 0;
#   end
#
# A well on any *rate* control with a zero target is shut, not open and held
# at zero. MRST never asks Newton to find the pressure at which an open well's
# perforations cancel; it closes the well.


def _prepare_facility(*wells):
    fm = GenericFacilityModel.__new__(GenericFacilityModel)
    fm.wells = list(wells)
    fm.WellModels = []
    fm.get_indices_of_active_wells = lambda ws: np.array([], dtype=int)
    return fm


def _plain_well(name, well_type, val):
    return SimpleNamespace(name=name, type=well_type, val=val, status=True,
                           lims={}, sign=-1.0)


def test_a_zero_rate_target_shuts_the_well():
    fm = _prepare_facility(_plain_well("B-2H", "orat", 0.0))
    state = {"wellSol": [{"status": True}]}
    _, state = fm.prepare_timestep(state, state, 1.0)
    assert fm.wells[0].status is False
    assert state["wellSol"][0]["status"] is False
    print("ok")


def test_a_nonzero_rate_target_leaves_the_well_open():
    fm = _prepare_facility(_plain_well("C-4H", "rate", 4.54))
    _, _ = fm.prepare_timestep({"wellSol": [{"status": True}]},
                               {"wellSol": [{}]}, 1.0)
    assert fm.wells[0].status is True


def test_a_pressure_control_is_exempt():
    """Zero *pressure* is a target, not a closure - MRST excludes bhp and thp
    explicitly. Without the exemption a well told to draw down to datum would
    be shut instead."""
    fm = _prepare_facility(_plain_well("P", "bhp", 0.0),
                           _plain_well("T", "thp", 0.0))
    _, _ = fm.prepare_timestep({"wellSol": [{}, {}]}, {"wellSol": [{}, {}]}, 1.0)
    assert all(w.status is True for w in fm.wells)


def test_the_rule_only_ever_closes_a_well():
    """It conjoins with the existing status; it does not replace it.

    MRST's `status = W.val ~= 0` overwrites a status the deck has already set,
    so it can only close. Assigning it directly re-opens every well the deck
    shut that still carries a target - on Norne that is 14 of 36 wells, and it
    makes the first step singular. Reopening is the schedule's job: the next
    control period supplies a fresh record.
    """
    shut_but_targeted = _plain_well("D-2H", "resv_history", -0.21)
    shut_but_targeted.status = False
    fm = _prepare_facility(shut_but_targeted)
    fm.prepare_timestep({"wellSol": [{}]}, {"wellSol": [{}]}, 1.0)
    assert shut_but_targeted.status is False, "a deck-shut well must stay shut"


def test_an_open_well_with_a_target_is_untouched():
    well = _plain_well("D-1H", "resv_history", -7.41)
    fm = _prepare_facility(well)
    fm.prepare_timestep({"wellSol": [{}]}, {"wellSol": [{}]}, 1.0)
    assert well.status is True
