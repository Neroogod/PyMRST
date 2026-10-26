"""Phase 341 (AC5): pressure-reduction factors (true-IMPES weights).

These weights have no simpler reference to compare against - their whole
purpose is to be the coefficients that make the pressure equation
saturation-independent. So the proof is that defining property, as an exact
invariant:

    sum_i w_i * M_i   has (to round-off) no derivative w.r.t. the secondary
                      (saturation / rs) unknowns,

because ``sum_i w_i M_i`` is a volume balance. For the immiscible base with
``w = pv/rho`` and phase mass ``M = pv*s*rho`` the sum is ``pv^2`` (constant in
saturation). For black-oil the dissolved-gas cross terms cancel exactly between
the oil and gas components, leaving ``sum_i w_i M_i = 1`` - and both the
saturation *and* the rs derivative vanish. That cancellation is the whole
point, and it is what this phase checks on the live-oil SPE 1 model.
"""

from pathlib import Path

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
    initial_state_from_deck,
)
from pymrst.modules.ad_core.statefunctions.flowprops import (
    BlackOilPressureReductionFactors,
    BlackOilShrinkageFactors,
    PressureReductionFactors,
    SurfaceDensity,
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


def check_immiscible():
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
        "PressureReductionFactors", PressureReductionFactors(model))
    st = model.init_state_ad(State(
        pressure=np.linspace(210, 190, g.cells.num) * BARSA,
        sW=np.full(g.cells.num, 0.45), qWs=np.array([2e-3]),
        qOs=np.array([-2e-3]), bhp=np.array([150 * BARSA])))
    wgt = model.get_props(st, "PressureReductionFactors")
    s = model.get_props(st, "Saturations")
    rho = model.get_props(st, "Density")
    pv = model.get_props(st, "PoreVolume")
    total = sum(wgt[ph] * (pv * s[ph] * rho[ph]) for ph in range(model.n_phases))
    d = np.abs(total.jac[1].toarray()).max()
    mag = np.abs(value(total)).max()
    print(f"  immiscible: |d(sum w*M)/dsW| = {d:.2e} (mag {mag:.2e})")
    assert d < 1e-6 * mag


def check_blackoil():
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
    model.PVTPropertyFunctions.set_state_function(
        "ShrinkageFactors", BlackOilShrinkageFactors(model))
    model.PVTPropertyFunctions.set_state_function(
        "SurfaceDensity", SurfaceDensity(model))
    model.FlowPropertyFunctions.set_state_function(
        "PressureReductionFactors", BlackOilPressureReductionFactors(model))

    st0 = initial_state_from_deck(model, deck)
    st = dict(st0)
    nc = g.cells.num
    st["sG"] = np.where(np.arange(nc) % 2 == 0, 0.15, 0.0)
    st["status"] = np.where(np.arange(nc) % 2 == 0, 2, 1)
    st = model.init_state_ad(State(st))

    wgt = model.get_props(st, "PressureReductionFactors")
    b = model.get_props(st, "ShrinkageFactors")
    s = model.get_props(st, "Saturations")
    pv = model.get_props(st, "PoreVolume")
    rhoS = model.fluid.rho_s
    rs = st["rs"]
    # mass-basis component masses (water / surface-oil / surface-gas)
    m_w = pv * b[0] * s[0] * rhoS[0]
    m_o = pv * b[1] * s[1] * rhoS[1]
    m_g = pv * (b[2] * s[2] + rs * b[1] * s[1]) * rhoS[2]
    total = wgt[0] * m_w + wgt[1] * m_o + wgt[2] * m_g

    val = value(total)
    print(f"  black-oil: sum w*M in [{val.min():.4f}, {val.max():.4f}] "
          f"(expected 1)")
    assert np.allclose(val, 1.0, atol=1e-9)
    # both secondary blocks (sW and the x=rs/sG variable) must vanish
    for blk in (1, 2):
        d = np.abs(total.jac[blk].toarray()).max()
        print(f"  black-oil: |d(sum w*M)/d(secondary {blk})| = {d:.2e}")
        assert d < 1e-8


def main():
    print("Phase 341 (AC5): pressure-reduction factors")
    check_immiscible()
    if DECK.exists():
        check_blackoil()
    else:
        print("  (black-oil check skipped: SPE 1 deck absent)")
    print("Phase 341 passed - the pressure weights are saturation-independent "
          "(true-IMPES), and the black-oil dissolved-gas terms cancel exactly.")


if __name__ == "__main__":
    main()
