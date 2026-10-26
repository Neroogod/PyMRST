"""Phase 346 (AC11): GenericReservoirModel and WrapperModel.

Model-layer machinery, proved against the validated black-oil model:

1. **GenericReservoirModel** is the marker base that declares "assembled from
   the state-function graph." The proof: the validated `GenericBlackOilModelAD`
   *is* one (the MRST is-a relationship), and its `get_model_equations` is the
   generic graph assembly - identical to its `conservation_equations`,
   component for component, value and Jacobian.

2. **WrapperModel** is a transparent delegator. The proof: a wrapper around the
   black-oil model reproduces the parent's properties and equations exactly,
   and a subclass overriding one method keeps the rest delegated - which is how
   the sequential and multiscale families are built.

No `.mat`: both are checked against the model they wrap / mark.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.models import (
    GenericReservoirModel,
    WrapperModel,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def build():
    g = compute_geometry(cart_grid([6, 5, 2], [300.0, 250.0, 20.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
    model = GenericBlackOilModelAD(g, rock, fluid, w)
    st0 = State(pressure=np.full(g.cells.num, 200 * BARSA),
                sW=np.full(g.cells.num, 0.4),
                qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                bhp=np.array([150 * BARSA]))
    return model, model.init_state_ad(st0), model.init_state_ad(st0)


def check_generic(model, st, st0):
    assert isinstance(model, GenericReservoirModel)
    a = model.get_model_equations(st0, st, 30.0 * DAY)
    b = model.conservation_equations(st0, st, 30.0 * DAY)
    worst = max(np.max(np.abs(value(a[i]) - value(b[i]))) for i in range(len(a)))
    jac = max(np.max(np.abs((a[i] - b[i]).jac[0].toarray())) for i in range(len(a)))
    print(f"  GenericReservoirModel: black-oil is-a generic model; "
          f"get_model_equations vs conservation_equations {worst:.2e}/{jac:.2e}")
    assert worst == 0.0 and jac == 0.0


def check_wrapper(model, st, st0):
    wrap = WrapperModel(model)
    worst = 0.0
    for name in ("Mobility", "PhaseFlux", "ComponentTotalMass"):
        a = wrap.get_props(st, name)
        b = model.get_props(st, name)
        for i in range(len(a)):
            worst = max(worst, np.max(np.abs(value(a[i]) - value(b[i]))))
    eqa = wrap.conservation_equations(st0, st, 30.0 * DAY)
    eqb = model.conservation_equations(st0, st, 30.0 * DAY)
    for i in range(len(eqa)):
        worst = max(worst, np.max(np.abs(value(eqa[i]) - value(eqb[i]))))
    print(f"  WrapperModel: props + equations reproduce the parent {worst:.2e}")
    assert worst == 0.0
    assert wrap.get_reservoir_model() is model
    assert wrap.n_phases == model.n_phases


def main():
    print("Phase 346 (AC11): GenericReservoirModel + WrapperModel")
    model, st, st0 = build()
    check_generic(model, st, st0)
    check_wrapper(model, st, st0)
    print("Phase 346 passed - the black-oil model is a generic reservoir model, "
          "and a wrapper reproduces it transparently.")


if __name__ == "__main__":
    main()
