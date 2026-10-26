"""Phase 347 (AC12 + AC9): CompositeModel and coupling terms.

Model-layer machinery. Its headline use - flow coupled to geomechanics through
the graph - needs geomechanics expressed as a state-function grouping, which
PyMRST's hand-coupled `ad_mechanics` is not; that coupled *solve* is a
documented follow-on. What is validated here is the composition itself and the
one property a coupling must have: conservation.

Three checks, against the validated black-oil submodels:

1. **Composition is faithful.** A one-submodel composite yields exactly the
   submodel's own equations; a two-submodel composite yields both submodels'
   equations concatenated, each matching the standalone model.
2. **Coupling is conservative.** A `SourceCouplingTerm` transfers a component
   source from submodel A to submodel B: A's residual loses exactly ``q``, B's
   gains exactly ``q``, and the summed residual ``eq_A + eq_B`` is unchanged.
   That conservation is the defining property of a physical coupling.

No `.mat`: composition is checked against the submodels, conservation against
itself.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.models import CompositeModel
from pymrst.modules.ad_core.statefunctions.composite import SourceCouplingTerm
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def _reservoir(seed_sw=0.4):
    g = compute_geometry(cart_grid([5, 4, 1], [250.0, 200.0, 10.0]))
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
                sW=np.full(g.cells.num, seed_sw),
                qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                bhp=np.array([150 * BARSA]))
    return model, model.init_state_ad(st0), model.init_state_ad(st0)


def check_composition():
    subA, stA, st0A = _reservoir(0.4)
    subB, stB, st0B = _reservoir(0.6)
    comp = CompositeModel({"A": subA, "B": subB})
    eqs, _ = comp.get_model_equations(
        {"A": st0A, "B": st0B}, {"A": stA, "B": stB}, 30.0 * DAY)
    refA = subA.conservation_equations(st0A, stA, 30.0 * DAY)
    refB = subB.conservation_equations(st0B, stB, 30.0 * DAY)
    wa = max(np.max(np.abs(value(eqs["A"][i]) - value(refA[i])))
             for i in range(len(refA)))
    wb = max(np.max(np.abs(value(eqs["B"][i]) - value(refB[i])))
             for i in range(len(refB)))
    print(f"  composition faithful: submodel A {wa:.2e}, submodel B {wb:.2e}")
    assert wa == 0.0 and wb == 0.0
    assert len(comp.flatten_equations(eqs)) == len(refA) + len(refB)


def check_coupling_conserves():
    subA, stA, st0A = _reservoir(0.4)
    subB, stB, st0B = _reservoir(0.6)
    comp = CompositeModel({"A": subA, "B": subB})
    comp.set_coupling_term(SourceCouplingTerm(comp, "A", "B"), "AtoB")
    state, state0 = {"A": stA, "B": stB}, {"A": st0A, "B": st0B}
    nc = subA.G.cells.num
    q = [np.full(nc, 1.5e-3), np.full(nc, 2.5e-3)]

    eqs0, _ = comp.get_model_equations(state0, state, 30.0 * DAY)
    eqs1, _ = comp.get_model_equations(state0, state, 30.0 * DAY,
                                       sources={"AtoB": q})
    worst_cons = worst_transfer = 0.0
    for i in range(2):
        before = value(eqs0["A"][i]) + value(eqs0["B"][i])
        after = value(eqs1["A"][i]) + value(eqs1["B"][i])
        worst_cons = max(worst_cons, np.max(np.abs(after - before)))
        worst_transfer = max(
            worst_transfer,
            np.max(np.abs((value(eqs1["A"][i]) - value(eqs0["A"][i])) + q[i])),
            np.max(np.abs((value(eqs1["B"][i]) - value(eqs0["B"][i])) - q[i])))
    print(f"  coupling conserves total: {worst_cons:.2e}; "
          f"exact A->B transfer: {worst_transfer:.2e}")
    assert worst_cons < 1e-18 and worst_transfer < 1e-18


def main():
    print("Phase 347 (AC12 + AC9): CompositeModel + coupling terms")
    check_composition()
    check_coupling_conserves()
    print("Phase 347 passed - submodel composition is faithful and the "
          "SourceCouplingTerm is exactly conservative.")


if __name__ == "__main__":
    main()
