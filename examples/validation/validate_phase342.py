"""Phase 342 (AC6): component flow properties, and FaceComponentMobility.

Machinery - proved against the already-validated component physics.

The component flux the graph validates today (`ComponentPhaseFlux`, Phases
5b/24/47/54-57) upwinds the component density and multiplies by the phase flux.
AC6 adds the MRST component-property route: `ComponentPhaseDensity`,
`ComponentMobility` (= phase mobility x component density), and the AC1-deferred
`FaceComponentMobility` (the upwinded component mobility). The proof closes the
loop:

    scale_c * FaceComponentMobility_{c,ph} * (K grad Phi_ph)
        == the validated ComponentPhaseFlux_{c,ph}

to round-off. This holds because single-point upwind satisfies
``upw(a) * upw(b) = upw(a*b)``, so upwinding the *product* (component mobility)
and upwinding the density then multiplying by the phase flux are the same
number. If they agree, the component-property route inherits the validation of
the component flux, and a new species is a component object rather than a new
residual.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.flowprops.component_properties import (
    ComponentMobility,
    ComponentPhaseDensity,
)
from pymrst.modules.ad_core.statefunctions.flux import (
    FaceComponentMobility,
    PermeabilityPotentialGradient,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def build():
    g = compute_geometry(cart_grid([8, 6, 3], [400.0, 300.0, 30.0]))
    kvals = (50 + (11 * np.arange(g.cells.num)) % 200) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=3e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
    model = GenericBlackOilModel(g, rock, fluid, w, gravity=9.80665)
    model.FlowPropertyFunctions.set_state_function(
        "ComponentPhaseDensity", ComponentPhaseDensity(model))
    model.FlowPropertyFunctions.set_state_function(
        "ComponentMobility", ComponentMobility(model))
    model.FlowDiscretization.set_state_function(
        "FaceComponentMobility", FaceComponentMobility(model))
    model.FlowDiscretization.set_state_function(
        "PermeabilityPotentialGradient", PermeabilityPotentialGradient(model))
    st = model.init_state_ad(State(
        pressure=np.linspace(230, 180, g.cells.num) * BARSA,
        sW=np.full(g.cells.num, 0.5), qWs=np.array([3e-3]),
        qOs=np.array([-3e-3]), bhp=np.array([150 * BARSA])))
    return model, st


def check_component_mobility(model, st):
    cmob = model.get_props(st, "ComponentMobility")
    cpd = model.get_props(st, "ComponentPhaseDensity")
    mob = model.get_props(st, "Mobility")
    worst = 0.0
    for c in range(len(cmob)):
        for ph in range(model.n_phases):
            if cmob[c][ph] is None:
                continue
            worst = max(worst, np.max(np.abs(
                value(cmob[c][ph]) - value(mob[ph] * cpd[c][ph]))))
    print(f"  ComponentMobility vs Mobility*ComponentPhaseDensity: {worst:.2e}")
    assert worst == 0.0


def check_flux_reproduction(model, st):
    existing = model.get_props(st, "ComponentPhaseFlux")
    fcm = model.get_props(st, "FaceComponentMobility")
    kpg = model.get_props(st, "PermeabilityPotentialGradient")
    scale = model._basis_scale()
    worst = fscale = 0.0
    for c in range(len(fcm)):
        for ph in range(model.n_phases):
            if fcm[c][ph] is None:
                assert existing[c][ph] is None
                continue
            recomposed = -scale[c] * fcm[c][ph] * kpg[ph]
            worst = max(worst, np.max(np.abs(
                value(recomposed) - value(existing[c][ph]))))
            fscale = max(fscale, np.max(np.abs(value(existing[c][ph]))))
    print(f"  component flux via FaceComponentMobility vs validated: "
          f"{worst:.2e} (flux scale {fscale:.2e})")
    assert worst < 1e-12 * fscale


def main():
    print("Phase 342 (AC6): component flow properties + FaceComponentMobility")
    model, st = build()
    check_component_mobility(model, st)
    check_flux_reproduction(model, st)
    print("Phase 342 passed - the component-property route reproduces the "
          "validated component flux exactly.")


if __name__ == "__main__":
    main()
