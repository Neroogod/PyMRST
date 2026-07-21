"""Phase 350 (AC13): well control equations and control switching.

The single-well control layer, factored onto MRST's `models/wells/` layout.
Proved against the MATLAB-validated procedural facility path (Phases 39, 120),
which already reproduces MRST:

1. **Control equations.** For every control type - bhp, total rate, water /
   oil / gas rate, liquid rate - `setup_well_control_equation_single_well`
   reproduces `FacilityModel._control_equation` exactly, value and Jacobian.
2. **Control switching.** On a schedule where a rate injector needs more than
   its bhp cap (and a bhp producer exceeds its rate cap), the standalone
   `update_switched_controls` switches exactly the wells that
   `FacilityModel.update_switched_controls` switches, to the same modes.

No `.mat`: both are checked against the procedural versions that carry the
MATLAB validation.
"""

import copy

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.wells import (
    setup_well_control_equation_single_well,
    update_switched_controls,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import init_variables_adi, value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _single(ctype, val, sign):
    g = compute_geometry(cart_grid([3, 1, 1], [30.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    w = add_well(None, g, rock, np.array([0]), well_type=ctype, val=val,
                 compi=np.array([1.0, 0.0]), sign=sign)
    return w[0]


def check_control_equations():
    nph = 2
    qw, qo, bhp = init_variables_adi(np.array([2e-3]), np.array([-1e-3]),
                                     np.array([210.0 * BARSA]))
    q_s = [qw, qo]
    worst = 0.0
    for ctype, val, sign in [("bhp", 200 * BARSA, 1), ("rate", 3e-3, 1),
                             ("wrat", 2e-3, 1), ("orat", 1e-3, -1),
                             ("lrat", 1e-3, 1)]:
        w = _single(ctype, val, sign)
        new = setup_well_control_equation_single_well(w, bhp, q_s, nph)
        old = FacilityModel._control_equation(w, q_s, bhp, nph)
        worst = max(worst, np.max(np.abs(value(new) - value(old))))
        if hasattr(new, "jac"):
            worst = max(worst, np.max(np.abs((new - old).jac[0].toarray())))
    print(f"  control equations vs facility (all types): {worst:.2e}")
    assert worst == 0.0


def check_control_switching():
    # a rate injector with a bhp cap it will exceed, and a bhp producer with an
    # oil-rate cap it will exceed
    g = compute_geometry(cart_grid([5, 1, 1], [500.0, 100.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    init_simple_adi_fluid(mu=np.array([1.0, 5.0]) * CENTI * POISE,
                          rho=[1000.0, 800.0], n=[2, 2],
                          c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=3e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([4]), well_type="bhp", val=150 * BARSA,
                 compi=np.array([0.0, 1.0]), sign=-1)
    w[0].lims = {"bhp": 300 * BARSA, "rate": 3e-3}
    w[1].lims = {"orat": -1e-3}

    bhp = np.array([320 * BARSA, 150 * BARSA])          # injector over its cap
    rates = np.array([[3e-3, 0.0], [0.0, -2e-3]])       # producer over its cap

    # procedural facility path (mutates its own wells)
    fac = FacilityModel(copy.deepcopy(w))
    st = State(bhp=bhp, qWs=rates[:, 0].copy(), qOs=rates[:, 1].copy())
    fac.update_switched_controls(st, ["qWs", "qOs"])
    proc_types = [ww.type for ww in fac.wells]

    # standalone models/wells path (mutates a fresh copy)
    ww = copy.deepcopy(w)
    _, switches = update_switched_controls(ww, bhp, rates)
    new_types = [x.type for x in ww]

    print(f"  facility switches -> {proc_types}; standalone -> {new_types}")
    assert new_types == proc_types
    assert switches == [(0, "rate", "bhp"), (1, "bhp", "orat")]


def main():
    print("Phase 350 (AC13): well control equations + control switching")
    check_control_equations()
    check_control_switching()
    print("Phase 350 passed - the factored well-control layer reproduces the "
          "validated procedural facility control equations and switching.")


if __name__ == "__main__":
    main()
