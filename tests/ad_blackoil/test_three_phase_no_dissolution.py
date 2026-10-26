"""A three-phase model with no dissolution - dead oil and dry gas.

`GenericBlackOilModelAD(G, rock, fluid, 'disgas', false, 'vapoil', false)` on a
three-phase fluid is a legitimate MRST case and the one
`blackoilTutorialMexAcceleration` builds. PyMRST could not run it, because
four separate places decided the phase count from **the dissolution flags**
rather than from the fluid:

* `init_simple_adi_fluid` refused three phases outright;
* `GenericBlackOilModelAD` routed the fluid to the two-phase adapter, so a
  three-phase case reported two phases;
* `Saturations` returned two saturations, leaving the gas component with no
  phase to live in;
* `update_state` unpacked the Newton increment at the two-phase offsets.

The last was the one that could not be seen by inspection. `init_state_ad`
seeds `sG` as a primary variable directly after `sW`, so the well block starts
at `3*nc`; reading it from `2*nc` gave every bottom-hole pressure the gas
saturation's increment - which is to say, froze them - while the rates absorbed
the rest. The residuals looked *almost* right: CNV and MB fell three orders of
magnitude and only the well and control residuals stalled, then grew linearly.

These tests pin all four, on a case small enough to run in a second.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_core.initialization.get_initialization_regions_black_oil import (
    get_initialization_regions_black_oil_ad,
)
from pymrst.modules.ad_core.initialization.init_state_black_oil_ad import (
    init_state_black_oil_ad_ad,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.params.wells_and_bc.vertical_well import vertical_well
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, POISE

NORM_GRAVITY = 9.80665


@pytest.fixture(scope="module")
def case():
    g = compute_geometry(cart_grid([6, 6, 6], [200.0, 200.0, 100.0]))
    rock = make_rock(g, 0.1*DARCY, 0.3)
    fluid = init_simple_adi_fluid(
        n=[2, 3, 1], rho=[1000.0, 700.0, 10.0],
        mu=np.array([1.0, 5.0, 0.1])*CENTI*POISE,
        c=np.array([0.0, 1e-6, 1e-4])/BARSA)

    W = vertical_well([], g, rock, 0, 0, None,
                      compi=np.array([1.0, 0.0, 0.0]), well_type="rate",
                      val=1e-3, name="I1")
    W = vertical_well(W, g, rock, 5, 5, [0],
                      compi=np.array([1.0, 0.0, 0.0]), well_type="bhp",
                      val=100*BARSA, name="P1")

    model = GenericBlackOilModelAD(g, rock, fluid, W, disgas=False,
                                 vapoil=False, gravity=NORM_GRAVITY)
    region = get_initialization_regions_black_oil_ad(
        model, [70.0, 30.0], datum_pressure=200*BARSA)
    state = init_state_black_oil_ad_ad(model, region)

    n = len(W)
    state["qWs"] = np.array([1e-3, 0.0])
    state["qOs"] = np.zeros(n)
    state["qGs"] = np.zeros(n)
    state["bhp"] = np.array([float(np.mean(state["pressure"])) + 10*BARSA,
                             100*BARSA])
    return model, state


def test_the_fluid_carries_three_phases(case):
    model, _ = case
    assert model.raw_fluid.n_phases_declared == 3
    assert model.raw_fluid.b_g is not None
    assert model.n_phases == 3, model.n_phases


def test_the_model_is_not_a_two_phase_one(case):
    """Deciding on `disgas or vapoil` gave a two-phase model here, and the
    gas simply was not there."""
    model, _ = case
    assert model.disgas is False and model.vapoil is False
    assert model.fluid.n_phases == 3
    assert model.fluid.phases == ["water", "oil", "gas"]


def test_there_are_three_saturations(case):
    """`Saturations` decided three-phase from the dissolution flags, so this
    returned two - and the gas component had no phase to live in."""
    model, state = case
    ad = model.init_state_ad(state)
    saturations = model.get_props(ad, "Saturations")
    assert len(saturations) == 3, len(saturations)


def test_the_equations_assemble(case):
    model, state = case
    eqs = model.get_equations(state, state.copy(), float(DAY))
    assert len(eqs) == 3 + 3 + 1, len(eqs)   # 3 conservation, 3 well, 1 control


def test_the_increment_reaches_the_bottom_hole_pressures(case):
    """The defect that could not be seen by inspection.

    `init_state_ad` seeds `sG` right after `sW`, so the well block starts at
    `3*nc`. Unpacking from `2*nc` hands the bottom-hole pressures the gas
    saturation's increment - which for a converged-enough saturation is zero,
    so they never move at all.
    """
    model, state = case
    solver = NonLinearSolver()
    solver.verbose = False
    eqs = model.get_equations(state, state.copy(), float(DAY))
    increment = solver._solve_linear(eqs, model, state, float(DAY))
    updated = model.update_state(state.copy(), increment)

    moved = np.abs(np.asarray(updated["bhp"]) - np.asarray(state["bhp"]))
    assert moved[0] > 1.0, (
        "the rate-controlled injector's bottom-hole pressure did not move; "
        "the well block is being read at the wrong offset")


def test_the_gas_saturation_is_updated(case):
    model, state = case
    solver = NonLinearSolver()
    solver.verbose = False
    eqs = model.get_equations(state, state.copy(), float(DAY))
    increment = solver._solve_linear(eqs, model, state, float(DAY))
    updated = model.update_state(state.copy(), increment)
    assert not np.allclose(updated["sG"], state["sG"])
    assert np.allclose(np.asarray(updated["sW"]) + np.asarray(updated["sO"])
                       + np.asarray(updated["sG"]), 1.0)


def test_it_converges(case):
    """The whole point. Before the offset fix this stalled at every timestep,
    down to a tenth of a day - so it was never a stiffness problem."""
    model, state = case
    solver = NonLinearSolver(max_iterations=30)
    solver.verbose = False
    _, report = solver.solve_ministep(model, state, state.copy(), 30.0*DAY)
    assert report.converged, report.residuals[-1]
    assert report.iterations < 25, report.iterations


def test_the_rate_controlled_injector_holds_its_target(case):
    """A `rate` control constrains the **total** surface rate, not the water
    rate. The injector is perforated through all six layers including the gas
    cap, so some perforations flow gas back into the wellbore and the mixture
    is not the declared `compi` - which is crossflow working, not a defect.
    Asserting `qWs == target` would forbid it.
    """
    model, state = case
    solver = NonLinearSolver(max_iterations=30)
    solver.verbose = False
    new, report = solver.solve_ministep(model, state, state.copy(), 30.0*DAY)
    assert report.converged
    total = sum(float(np.asarray(new[key])[0])
                for key in ("qWs", "qOs", "qGs"))
    assert np.isclose(total, 1e-3, rtol=1e-6), total
    assert float(np.asarray(new["qWs"])[0]) > 0.0, "the injector must inject"


def test_a_two_phase_model_is_unaffected():
    """The path everything else takes."""
    g = compute_geometry(cart_grid([5, 5, 2], [100.0, 100.0, 20.0]))
    rock = make_rock(g, 0.1*DARCY, 0.3)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0])*CENTI*POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6/BARSA, 1e-6/BARSA], p_ref=200*BARSA)
    W = vertical_well([], g, rock, 0, 0, None, compi=np.array([1.0, 0.0]),
                      well_type="rate", val=1e-4, name="I")
    W = vertical_well(W, g, rock, 4, 4, None, compi=np.array([0.0, 1.0]),
                      well_type="bhp", val=150*BARSA, name="P")
    model = GenericBlackOilModelAD(g, rock, fluid, W)
    assert model.n_phases == 2
    assert model.fluid.phases == ["water", "oil"]

    from pymrst.modules.ad_core import State
    state = State(pressure=np.full(g.cells.num, 200*BARSA),
                  sW=np.full(g.cells.num, 0.2),
                  qWs=np.array([1e-4, 0.0]), qOs=np.array([0.0, 0.0]),
                  bhp=np.array([210*BARSA, 150*BARSA]))
    solver = NonLinearSolver()
    solver.verbose = False
    _, report = solver.solve_ministep(model, state, state.copy(), 10.0*DAY)
    assert report.converged
