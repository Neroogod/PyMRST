"""`UniformFacilityModel` and MRST's classic well stack.

This class was a wrapper - inheriting everything, overriding nothing - while
claiming a "simplified facility model". It behaved exactly like
`FacilityModel`, so asking for it changed nothing, and the ledger reported it
`ported` because a file of the right name existed.

It then became an honest *gap*: the one method it exists for raised, naming
`computeWellContributionsNew` as what was missing. That function and
`setupWellControlEquations` were excluded from the ledger as "deprecated,
superseded by FacilityModel/SimpleWell" - a reason that was wrong in a
specific way, since this class is not deprecated and calls them directly.

Both are ported now, so this file tests the assembly rather than the refusal.
The properties worth pinning are the ones a single-pass well model gets wrong
quietly:

* **crossflow** - a well can take fluid in at one perforation and put it out
  at another, so a producing connection delivers the *wellbore mixture*, not
  its own cell's composition;
* **the restriction** - "uniform" means no components and no extra equations,
  and MRST asserts it. Dropping components silently is the failure mode;
* **undeliverable controls** - an oil-rate target on a mixture with no oil has
  a structurally singular row unless it is substituted.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_core.models.facilities import (
    FacilityModel, UniformFacilityModel,
)
from pymrst.modules.ad_core.models.wells.compute_well_contributions_new import (
    compute_well_contributions_new,
)
from pymrst.modules.ad_core.models.wells.setup_well_control_equations import (
    setup_well_control_equations,
)
from pymrst.utils.adi import init_variables_adi, value

MODEL = SimpleNamespace(disgas=False, vapoil=False,
                        get_phase_names=lambda: ["W", "O"])


def _stack(n_perf=2, sign=-1.0, cdp=None, crossflow=True, status=True):
    """One well with `n_perf` perforations, and the pieces the stack needs."""
    W = [SimpleNamespace(WI=np.full(n_perf, 1e-11),
                         compi=np.array([1.0, 0.0]), status=status,
                         cstatus=np.ones(n_perf, dtype=bool), sign=sign)]
    sol = [SimpleNamespace(cdp=np.zeros(n_perf) if cdp is None else cdp,
                           cstatus=np.ones(n_perf, dtype=bool), status=True,
                           type="bhp", val=1.9e7, bhp=1.9e7)]
    wellmodel = SimpleNamespace(
        W=W, referencePressure=np.full(n_perf, 2.0e7),
        bfactors=[np.ones(n_perf), np.ones(n_perf)], components=[],
        mobilities=[np.full(n_perf, 1e-3), np.full(n_perf, 1e-4)],
        perf2well=np.zeros(n_perf, dtype=int),
        Rw=sp.csr_matrix(np.ones((n_perf, 1))),
        allowWellSignChange=False, allowCrossflow=crossflow)
    return wellmodel, sol


# -- the class ---------------------------------------------------------------

def test_it_still_constructs_and_is_a_facility_model():
    model = UniformFacilityModel([])
    assert isinstance(model, FacilityModel)
    assert model.setWellTargets is True


def test_the_override_is_real_not_inherited():
    """The check that would have caught the original wrapper: the method must
    belong to this class, not be borrowed from its base."""
    assert "get_well_contributions" in vars(UniformFacilityModel)
    assert UniformFacilityModel.get_well_contributions is not \
        FacilityModel.get_well_contributions


def test_components_are_refused_rather_than_dropped():
    """"Uniform" means no components, and MRST asserts it. Assembling anyway
    would return a model that silently ignores component transport."""
    facility = UniformFacilityModel([])
    facility.ReservoirModel = SimpleNamespace(
        get_component_names=lambda: ["CO2"])
    with pytest.raises(NotImplementedError, match="does not support components"):
        facility.get_well_contributions()


def test_extra_well_equations_are_refused():
    facility = UniformFacilityModel([])
    facility.ReservoirModel = SimpleNamespace(get_component_names=lambda: [])
    facility.addedEquationNames = ["polymer"]
    with pytest.raises(NotImplementedError, match="extra well equations"):
        facility.get_well_contributions()


# -- the classic contribution stack -----------------------------------------

def test_a_producer_draws_from_every_connection():
    """Wellbore below reservoir pressure everywhere: all connections produce,
    so all surface fluxes are negative."""
    wellmodel, sol = _stack()
    _, cq_s, _, _, cstatus, _ = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.9e7]),
        [np.array([-1e-3]), np.array([-1e-4])])

    assert np.all(np.asarray(cq_s[0]) < 0)
    assert cstatus.all()


def test_the_wellbore_mixture_is_a_composition():
    """`mix_s` is what an injecting connection delivers, so it must be a
    composition - non-negative and summing to one."""
    wellmodel, sol = _stack()
    _, _, mix_s, *_ = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.9e7]),
        [np.array([-1e-3]), np.array([-1e-4])])

    np.testing.assert_allclose(mix_s.sum(axis=1), 1.0)
    assert np.all(mix_s >= -1e-12)


def test_a_crossflowing_connection_can_be_closed():
    """With crossflow disallowed, a connection flowing against its well's
    direction is shut instead of reversing.

    `drawdown = p - p_bh - cdp`, so a connection injects when its pressure
    drop exceeds `p - p_bh` (here 1.0e6). The second perforation is given
    2.0e6 and therefore flows the other way from its own well.
    """
    cdp = np.array([0.0, 2.0e6])
    open_wm, open_sol = _stack(cdp=cdp, crossflow=True)
    shut_wm, shut_sol = _stack(cdp=cdp, crossflow=False)
    args = (np.array([1.9e7]), [np.array([-1e-3]), np.array([-1e-4])])

    _, _, _, _, cstatus_open, _ = compute_well_contributions_new(
        open_wm, MODEL, open_sol, *args)
    _, _, _, _, cstatus_shut, _ = compute_well_contributions_new(
        shut_wm, MODEL, shut_sol, *args)

    assert cstatus_open.all(), "crossflow allowed: nothing should close"
    assert not cstatus_shut.all(), "crossflow disallowed: one must close"


def test_a_producing_connection_uses_its_cells_mobilities():
    """The discriminator between the two flux branches.

    A producing connection delivers what its *cell* holds, weighted by each
    phase's mobility. An injecting one delivers the well's declared
    composition, `compi`. With `compi = [1, 0]` the injecting branch gives the
    oil connection exactly zero, so a non-zero oil flux can only have come
    from the producing branch.
    """
    wellmodel, sol = _stack()
    _, _, _, _, _, cq_r = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.9e7]),
        [np.array([-1e-3]), np.array([-1e-4])])

    assert np.all(np.asarray(cq_r[1]) != 0), (
        "oil flux is zero, so the injecting branch was taken for a producer")
    # And the ratio is the mobility ratio, not the composition.
    ratio = np.asarray(cq_r[1])/np.asarray(cq_r[0])
    np.testing.assert_allclose(ratio, 1e-4/1e-3, rtol=1e-12)


def test_a_closed_connection_carries_no_flux():
    """Closing must stop the flow, not merely relabel its direction."""
    cdp = np.array([0.0, 2.0e6])
    wellmodel, sol = _stack(cdp=cdp, crossflow=False)
    _, cq_s, _, _, cstatus, _ = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.9e7]),
        [np.array([-1e-3]), np.array([-1e-4])])

    closed = ~cstatus
    assert closed.any()
    np.testing.assert_allclose(np.asarray(cq_s[0])[closed], 0.0, atol=1e-30)
    np.testing.assert_allclose(np.asarray(cq_s[1])[closed], 0.0, atol=1e-30)


def test_an_injecting_connection_is_converted_to_surface_volume():
    """The `volRat` division, which `b = 1` everywhere cannot detect.

    An injector delivers its own composition, so `mix_s = compi = [1, 0]` and
    `volRat = 1/b_w`. With `b_w = 1.25` the surface rate is `1.25x` the
    reservoir rate, and skipping the division loses exactly that factor.
    """
    n_perf = 1
    wellmodel, sol = _stack(n_perf=n_perf, sign=1.0)
    wellmodel.bfactors = [np.full(n_perf, 1.25), np.full(n_perf, 1.0)]
    # p_bh above the reservoir pressure, so the connection injects.
    _, cq_s, mix_s, _, _, cq_r = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([2.1e7]),
        [np.array([1e-3]), np.array([0.0])])

    np.testing.assert_allclose(mix_s[0], [1.0, 0.0], atol=1e-12)
    vol_ratio = 1.0/1.25
    np.testing.assert_allclose(np.asarray(cq_s[0]),
                               np.asarray(cq_r[0])/vol_ratio, rtol=1e-10)


def test_the_well_equation_balances_rate_against_connections():
    """`eqs = q_s - sum of connection fluxes`. The sign is what makes it a
    balance rather than a doubling."""
    wellmodel, sol = _stack()
    q_s = [np.array([-1e-3]), np.array([-1e-4])]
    eqs, cq_s, *_ = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.9e7]), q_s)

    for ph in range(2):
        expected = q_s[ph] - np.asarray(cq_s[ph]).sum()
        np.testing.assert_allclose(np.asarray(eqs[ph]), expected, rtol=1e-10)


def test_a_shut_well_keeps_a_solvable_row():
    """`q - q = 0` holds trivially but keeps a unit derivative, where an empty
    row would be structurally singular."""
    wellmodel, sol = _stack(status=False)
    q, = init_variables_adi(np.array([-1e-3]))
    eqs, *_ = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.9e7]), [q, q*1.0])

    np.testing.assert_allclose(value(eqs[0]), 0.0, atol=1e-30)
    assert eqs[0].jac[0].nnz > 0, "the row lost its derivative"


def test_the_stack_carries_ad_through():
    wellmodel, sol = _stack()
    bhp, = init_variables_adi(np.array([1.9e7]))
    eqs, cq_s, *_ = compute_well_contributions_new(
        wellmodel, MODEL, sol, bhp,
        [np.array([-1e-3]), np.array([-1e-4])])

    assert hasattr(eqs[0], "jac"), "AD did not survive the assembly"
    assert eqs[0].jac[0].nnz > 0


def test_a_higher_wellbore_pressure_reduces_production():
    """The sign that makes the whole thing physical: raising the BHP of a
    producer reduces its drawdown and so its rate."""
    wellmodel, sol = _stack()
    rates = [np.array([-1e-3]), np.array([-1e-4])]

    low = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.8e7]), rates)[1][0]
    high = compute_well_contributions_new(
        wellmodel, MODEL, sol, np.array([1.95e7]), rates)[1][0]
    assert np.sum(np.asarray(high)) > np.sum(np.asarray(low))


# -- the control equations ---------------------------------------------------

def test_a_bhp_well_closes_on_its_pressure():
    eq = setup_well_control_equations(
        [SimpleNamespace(type="bhp", val=2.0e7, bhp=2.0e7, status=True)],
        np.array([1.9e7]), [np.array([-1.0]), np.array([-2.0])],
        np.array([True]), np.array([[0.5, 0.5]]), MODEL)
    np.testing.assert_allclose(eq, 1.9e7 - 2.0e7)


def test_a_rate_well_closes_on_its_total():
    eq = setup_well_control_equations(
        [SimpleNamespace(type="rate", val=-3.0, bhp=2.0e7, status=True)],
        np.array([2.0e7]), [np.array([-1.0]), np.array([-2.0])],
        np.array([True]), np.array([[0.5, 0.5]]), MODEL)
    np.testing.assert_allclose(eq, 0.0)


def test_an_undeliverable_phase_becomes_a_zero_rate_equation():
    """An oil-rate target on a mixture containing no oil has a zero derivative
    in every unknown - nothing the solver changes moves the residual."""
    eq = setup_well_control_equations(
        [SimpleNamespace(type="orat", val=-3.0, bhp=2.0e7, status=True)],
        np.array([2.0e7]), [np.array([-1.0]), np.array([-2.0])],
        np.array([True]), np.array([[1.0, 0.0]]), MODEL)
    np.testing.assert_allclose(eq, -3.0)


def test_a_deliverable_phase_keeps_its_own_equation():
    eq = setup_well_control_equations(
        [SimpleNamespace(type="orat", val=-2.0, bhp=2.0e7, status=True)],
        np.array([2.0e7]), [np.array([-1.0]), np.array([-2.0])],
        np.array([True]), np.array([[0.5, 0.5]]), MODEL)
    np.testing.assert_allclose(eq, 0.0)


def test_a_zero_rate_target_becomes_a_total_rate_equation():
    """How a well is shut in practice.

    The control is `orat`, not `rate`: on a total-rate well the substituted
    equation and the ordinary one are the same expression, so the case cannot
    tell whether the substitution happened. On a phase-rate well they differ -
    `q_oil` against `q_total`.
    """
    eq = setup_well_control_equations(
        [SimpleNamespace(type="orat", val=0.0, bhp=2.0e7, status=True)],
        np.array([2.0e7]), [np.array([-1.0]), np.array([-2.0])],
        np.array([True]), np.array([[0.5, 0.5]]), MODEL)
    np.testing.assert_allclose(eq, -3.0), "expected the total rate, not q_oil"


def test_a_disabled_well_is_pinned_to_its_pressure_and_scaled():
    """Unscaled, a pressure residual in pascals sits beside rate residuals in
    m^3/s and the linear solver sees its worst row on the least important
    equation."""
    eq = setup_well_control_equations(
        [SimpleNamespace(type="bhp", val=2.0e7, bhp=1.8e7, status=False)],
        np.array([1.9e7]), [np.array([-1.0]), np.array([-2.0])],
        np.array([False]), np.array([[0.5, 0.5]]), MODEL)

    expected = 3.0*(1.9e7 - 1.8e7)/1.8e7
    np.testing.assert_allclose(eq, expected, rtol=1e-12)
    assert abs(float(np.ravel(eq)[0])) < 1.0, "the residual was not scaled"


def test_the_control_equation_carries_ad():
    bhp, = init_variables_adi(np.array([1.9e7]))
    eq = setup_well_control_equations(
        [SimpleNamespace(type="bhp", val=2.0e7, bhp=2.0e7, status=True)],
        bhp, [np.array([-1.0]), np.array([-2.0])],
        np.array([True]), np.array([[0.5, 0.5]]), MODEL)
    assert hasattr(eq, "jac") and eq.jac[0].nnz > 0
