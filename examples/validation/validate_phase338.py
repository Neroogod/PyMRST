"""Phase 338 (AC2): the upwind discretization hierarchy.

Machinery, not physics - proved the way the graph phases are, by reproducing
the already-validated upwinding bit for bit.

`FaceMobility` (validated through Phases 5/24) upwinds each phase mobility to
the interior faces with `model.operators.upw`. This phase wraps that operator
behind MRST's `UpwindDiscretization` interface. The check: upwinding the phase
mobilities through `UpwindFunctionWrapperDiscretization` (via the
`UpwindProperty` mixin) must give exactly `FaceMobility`, value and Jacobian.
If it does, any state function can take an upwind *object* instead of the raw
operator, and a higher-order scheme drops in without touching the flux group.

No `.mat`: two Python paths that must agree to round-off.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind import (
    UpwindFunctionWrapperDiscretization,
    UpwindProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def build_ow():
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
    st = State(pressure=np.linspace(240, 170, g.cells.num) * BARSA,
               sW=np.full(g.cells.num, 0.45),
               qWs=np.array([3e-3]), qOs=np.array([-3e-3]),
               bhp=np.array([150 * BARSA]))
    return model, model.init_state_ad(st)


def check_wrapper_reproduces_face_mobility():
    model, st = build_ow()
    facemob = model.get_props(st, "FaceMobility")
    flag = model.get_props(st, "PhaseUpwindFlag")
    mob = model.get_props(st, "Mobility")

    prop = UpwindProperty(UpwindFunctionWrapperDiscretization(model))
    worst_val = worst_jac = 0.0
    for ph in range(model.n_phases):
        through_wrapper = prop.face_upstream(model, st, flag[ph], mob[ph])
        worst_val = max(worst_val, np.max(np.abs(
            value(through_wrapper) - value(facemob[ph]))))
        worst_jac = max(worst_jac, np.max(np.abs(
            (through_wrapper - facemob[ph]).jac[0].toarray())))
    print(f"  FaceMobility vs UpwindProperty wrapper: "
          f"val {worst_val:.2e}, jac {worst_jac:.2e}")
    assert worst_val == 0.0 and worst_jac == 0.0


def main():
    print("Phase 338 (AC2): upwind discretization hierarchy")
    check_wrapper_reproduces_face_mobility()
    print("Phase 338 passed - the upwind wrapper reproduces the validated "
          "face upwinding exactly.")


if __name__ == "__main__":
    main()
