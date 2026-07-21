"""Phase 339 (AC3): the flow-state builders (implicit / explicit / AIM).

Machinery - proved against the already-validated fully-implicit flux, the way
the graph phases are.

Three checks, each pinning one builder to a value that would move if the
hybridization were wrong:

1. **Implicit == fully implicit.** The implicit builder must leave the flux the
   graph computes unchanged - bit for bit against the validated `PhaseFlux`.
2. **Explicit == IMPES split.** The explicit builder must freeze the face
   mobility at the old state while leaving pressure implicit, so its `PhaseFlux`
   equals ``-T * FaceMobility(state0) * PhasePotentialDifference(state)`` and
   the frozen term carries no derivative.
3. **AIM blends per cell.** An adaptive mask must reproduce the fully-implicit
   result where the mask is implicit, and the frozen old value where it is
   explicit, cell by cell - and the AIM state with an all-implicit mask must
   equal the implicit builder exactly.

No `.mat`: two Python paths that must agree to round-off.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.flux.flow_state import (
    AdaptiveImplicitFlowStateBuilder,
    ExplicitFlowStateBuilder,
    FlowStateBuilder,
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
    st0 = State(pressure=np.linspace(230, 180, g.cells.num) * BARSA,
                sW=np.full(g.cells.num, 0.4),
                qWs=np.array([3e-3]), qOs=np.array([-3e-3]),
                bhp=np.array([150 * BARSA]))
    st_new = State(pressure=np.linspace(228, 178, g.cells.num) * BARSA,
                   sW=np.full(g.cells.num, 0.6),
                   qWs=np.array([3e-3]), qOs=np.array([-3e-3]),
                   bhp=np.array([150 * BARSA]))
    return model, model.init_state_ad(st_new), st0, g.cells.num


def check_implicit_is_identity(model, st, st0):
    base = model.get_props(st, "PhaseFlux")
    fs = FlowStateBuilder().build(model, st, st0, dt=30.0)
    flux = model.get_props(fs, "PhaseFlux")
    worst = max(np.max(np.abs(value(flux[p]) - value(base[p])))
                for p in range(model.n_phases))
    print(f"  implicit builder vs fully-implicit PhaseFlux: {worst:.2e}")
    assert worst == 0.0


def check_explicit_is_impes(model, st, st0):
    fs = ExplicitFlowStateBuilder(
        explicit_flux=["FaceMobility"]).build(model, st, st0, dt=30.0)
    flux = model.get_props(fs, "PhaseFlux")
    t = model.operators.T
    mob0 = model.get_props(st0, "FaceMobility")
    pot = model.get_props(st, "PhasePotentialDifference")
    worst_v = worst_j = 0.0
    for p in range(model.n_phases):
        manual = -t * value(mob0[p]) * pot[p]
        worst_v = max(worst_v, np.max(np.abs(value(flux[p]) - value(manual))))
        worst_j = max(worst_j, np.max(np.abs((flux[p] - manual).jac[0].toarray())))
    print(f"  explicit builder vs IMPES split: val {worst_v:.2e}, jac {worst_j:.2e}")
    assert worst_v == 0.0 and worst_j == 0.0


def check_aim_blend(model, st, st0, nc):
    new = model.get_props(st, "Mobility")
    old = model.get_props(st0, "Mobility")
    mask = np.zeros(nc, bool)
    mask[::3] = True
    st["implicit"] = mask
    fs = AdaptiveImplicitFlowStateBuilder().build(model, st, st0, dt=30.0)
    mob = model.get_props(fs, "Mobility")
    worst = 0.0
    for p in range(model.n_phases):
        expect = np.where(mask, value(new[p]), value(old[p]))
        worst = max(worst, np.max(np.abs(value(mob[p]) - expect)))
    print(f"  AIM per-cell blend vs manual: {worst:.2e}")
    assert worst == 0.0

    st["implicit"] = np.ones(nc, bool)
    fs_all = AdaptiveImplicitFlowStateBuilder().build(model, st, st0, dt=30.0)
    imp = FlowStateBuilder().build(model, st, st0, dt=30.0)
    for p in range(model.n_phases):
        assert np.allclose(value(model.get_props(fs_all, "Mobility")[p]),
                           value(model.get_props(imp, "Mobility")[p]))


def main():
    print("Phase 339 (AC3): flow-state builders")
    model, st, st0, nc = build()
    check_implicit_is_identity(model, st, st0)
    check_explicit_is_impes(model, st, st0)
    check_aim_blend(model, st, st0, nc)
    print("Phase 339 passed - the builders hybridize the flux state exactly as "
          "MRST does (implicit / IMPES / AIM).")


if __name__ == "__main__":
    main()
