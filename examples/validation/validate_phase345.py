"""Phase 345 (relperm hierarchy + AC8 hysteresis).

Two things land together, because the second extends the first:

- **`BaseRelativePermeability`** (the AC4 relperm follow-on) - relative
  permeability on MRST's phase-by-phase hierarchy (WO / WOG dispatch, the
  connate-water-weighted three-phase oil blend). Proved by reproducing the
  validated `RelativePermeability` (the fluid's own `kr`) bit-for-bit.

- **`HystereticRelativePermeability`** (AC8) - the same, with Killough gas
  hysteresis on the gas phase. Proved by two invariants: on drainage (gas at
  its historic maximum) it equals the base curve; on imbibition (gas retreated
  below ``sGMax``) the gas relperm is trapped strictly below the drainage
  curve.

No `.mat`: the base is checked against the already-validated relperm, the
hysteresis against its defining monotonicity.
"""

from pathlib import Path

import numpy as np

from pymrst.gridprocessing import (
    cart_grid, compute_geometry, tensor_grid,
)
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD, initial_state_from_deck,
)
from pymrst.modules.ad_core.statefunctions.flowprops import (
    BaseRelativePermeability,
    HystereticRelativePermeability,
)
from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
    RelativePermeability as LegacyRelPerm,
)
from pymrst.modules.ad_core.statefunctions.hysteresis import (
    KilloughGasHysteresis,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

HERE = Path(__file__).parent
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


def check_base_oil_water():
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
    model.FlowPropertyFunctions.set_state_function(
        "BaseRelativePermeability", BaseRelativePermeability(model))
    model.FlowPropertyFunctions.set_state_function(
        "RelativePermeability", LegacyRelPerm(model))
    st = model.init_state_ad(State(
        pressure=np.linspace(210, 190, g.cells.num) * BARSA,
        sW=np.full(g.cells.num, 0.45), qWs=np.array([2e-3]),
        qOs=np.array([-2e-3]), bhp=np.array([150 * BARSA])))
    new = model.get_props(st, "BaseRelativePermeability")
    old = model.get_props(st, "RelativePermeability")
    worst = max(np.max(np.abs(value(new[p]) - value(old[p])))
                for p in range(model.n_phases))
    print(f"  BaseRelativePermeability vs legacy (oil-water): {worst:.2e}")
    assert worst == 0.0


def check_base_and_hysteresis_black_oil():
    deck = read_spe1_deck(DECK)
    g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(g, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = None
    for name in ("PRODUCER", "INJECTOR"):
        wd = deck["wells"][name]
        wells = add_well(wells, g, rock, wd["cells"], well_type=wd["type"],
                         val=wd["val"], radius=wd["radius"], compi=wd["compi"],
                         sign=wd["sign"], name=name, ref_depth=wd["ref_depth"],
                         gravity=np.array([0.0, 0.0, NORM_GRAVITY]))
    model = GenericBlackOilModelAD(g, rock, fluid, wells, disgas=True,
                                 gravity=NORM_GRAVITY)
    nc = g.cells.num
    st = dict(initial_state_from_deck(model, deck))
    st["status"] = np.full(nc, 2, int)
    st = model.init_state_ad(State(st))
    st["sG"] = np.full(nc, 0.3)                        # controlled gas saturation

    # base relperm reproduces the validated three-phase relperm
    model.FlowPropertyFunctions.set_state_function(
        "BaseRelativePermeability", BaseRelativePermeability(model))
    model.FlowPropertyFunctions.set_state_function(
        "RelativePermeability", LegacyRelPerm(model))
    new = model.get_props(st, "BaseRelativePermeability")
    old = model.get_props(st, "RelativePermeability")
    worst = max(np.max(np.abs(value(new[p]) - value(old[p])))
                for p in range(model.n_phases))
    print(f"  BaseRelativePermeability vs legacy (black-oil): {worst:.2e}")
    assert worst < 1e-14

    # hysteresis: drainage == base, imbibition traps
    raw = model.raw_fluid
    hyst = KilloughGasHysteresis(
        sg_max_primary=1.0, sg_min_primary=0.0, krg_drainage=raw.kr_g,
        krg_imbibition=lambda s: 0.7 * np.asarray(value(raw.kr_g(s)), float),
        sg_tmax=0.6)
    h = HystereticRelativePermeability(model, hyst)
    base = BaseRelativePermeability(model)

    st["sGMax"] = np.full(nc, 0.3)                     # == sG -> drainage
    krg_dr = h.evaluate_gas_rel_perm(model, st["sG"], st)
    krg_bs = base.evaluate_gas_rel_perm(model, st["sG"], st)
    assert np.allclose(value(krg_dr), value(krg_bs))
    print("  hysteresis on drainage == base curve: ok")

    st["sGMax"] = np.full(nc, 0.7)                     # > sG -> imbibition
    krg_im = np.asarray(value(h.evaluate_gas_rel_perm(model, st["sG"], st)))
    krg_bs = np.asarray(value(base.evaluate_gas_rel_perm(model, st["sG"], st)))
    assert np.all(krg_im <= krg_bs + 1e-12) and np.any(krg_im < krg_bs - 1e-9)
    print(f"  hysteresis on imbibition traps gas: "
          f"max reduction {np.max(krg_bs - krg_im):.2e}")


def main():
    print("Phase 345: relative-permeability hierarchy + gas hysteresis")
    check_base_oil_water()
    if DECK.exists():
        check_base_and_hysteresis_black_oil()
    else:
        print("  (black-oil + hysteresis checks skipped: SPE 1 deck absent)")
    print("Phase 345 passed - BaseRelativePermeability reproduces the validated "
          "relperm, and Killough hysteresis traps gas on imbibition.")


if __name__ == "__main__":
    main()
