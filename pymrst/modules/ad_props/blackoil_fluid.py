"""Black-oil fluid from ECLIPSE deck tables (MRST initDeckADIFluid subset).

Builds AD-transparent property functions for a live-oil / wet-gas /
compressible-water system: bO(p, rs, sat_flag) and muO via the 'parallel'
PVT interpolation, rsSat(p) from the bubble-point envelope, gas properties
(dry from PVDG, or wet with rv from PVTG), analytic water, SWOF/SGOF relperms
with flat extension, and the MRST default saturation-weighted three-phase oil
relperm.

**The two dissolutions are symmetric, and the flags are not.** `rs` is gas
dissolved in oil; `rv` is oil vaporised into gas. Each `b` takes the *other*
phase's presence as its saturation flag: `bO(p, rs, gas_present)` - oil is
saturated with gas exactly when free gas exists - and `bG(p, rv, oil_present)`.
Swapping them is an easy mistake and a silent one, because both flags are True
in the saturated cells that dominate a typical field.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from pymrst.modules.ad_props.tables import (
    extend_tab,
    interp_table,
    make_pvt_table,
    pvt_interp,
)
from pymrst.utils.adi import value


@dataclass
class BlackOilFluid:
    """Property functions for the water/oil/gas black-oil system."""

    rhoS: np.ndarray            # (3,) surface densities [water, oil, gas]
    b_w: Callable               # bW(p)
    mu_w: Callable              # muW(p)
    b_o: Callable               # bO(p, rs, gas_present)
    mu_o: Callable              # muO(p, rs, gas_present)
    rs_sat: Callable            # rsSat(p)
    b_g: Callable               # bG(p, rv, oil_present) - rv ignored if dry
    mu_g: Callable              # muG(p, rv, oil_present)
    kr_w: Callable
    kr_g: Callable
    kr_ow: Callable             # oil relperm in oil-water system, of sO
    kr_og: Callable             # oil relperm in oil-gas system, of sO
    pv_mult: Callable           # rock compressibility multiplier of p
    swcon: float                # connate water saturation
    pc_ow: Optional[Callable] = None   # p_o - p_w
    pc_og: Optional[Callable] = None   # p_g - p_o
    rv_sat: Optional[Callable] = None   # rvSat(p); None for a dry gas
    vapoil: bool = False

    def kr_o(self, sw, so, sg):
        """MRST default three-phase oil relperm (relPermWOG)."""
        swcon = np.minimum(self.swcon, value(sw) - 1e-5)
        d = sg + sw - swcon
        ww = (sw - swcon) / d
        return (1.0 - ww) * self.kr_og(so) + ww * self.kr_ow(so)


def init_deck_blackoil_fluid(deck: Dict, rock=None) -> BlackOilFluid:
    """Assemble a BlackOilFluid from a parsed (SI) SPE1-style deck."""
    # Water: analytic constant compressibility (assignPVTW).
    w = deck["pvtw"]
    assert w["viscosibility"] == 0.0, "water viscosibility not supported"

    def b_w(p):
        x = w["c"] * (p - w["p_ref"])
        return (1.0 + x + 0.5 * x**2) / w["b_ref"]

    mu_w = lambda p: w["mu_ref"] + 0.0 * p

    # Rock compressibility multiplier (assignROCK).
    r = deck["rock_cp"]

    def pv_mult(p):
        x = r["c"] * (p - r["p_ref"])
        return 1.0 + x + 0.5 * x**2

    # Gas: tabulated with a duplicated row at p = 0 (assignPVDG).
    pvdg = deck["pvdg"]
    p_g = np.concatenate([[0.0], pvdg[:, 0]])
    bg_tab = 1.0 / pvdg[:, 1]
    bg_tab = np.concatenate([[bg_tab[0]], bg_tab])
    mug_tab = np.concatenate([[pvdg[0, 2]], pvdg[:, 2]])
    # A dry gas carries no oil, so `rv` and its flag are accepted and ignored -
    # which lets every caller use one signature regardless of the fluid.
    b_g = lambda p, rv=None, flag=None: interp_table(p_g, bg_tab, p)
    mu_g = lambda p, rv=None, flag=None: interp_table(p_g, mug_tab, p)

    # Live oil: PVTO branches (assignPVTO, method 'parallel').
    keys = np.array([b[0] for b in deck["pvto"]])
    bo_tab = make_pvt_table(
        keys, [(rows[:, 0], 1.0 / rows[:, 1]) for _, rows in deck["pvto"]]
    )
    muo_tab = make_pvt_table(
        keys, [(rows[:, 0], rows[:, 2]) for _, rows in deck["pvto"]]
    )
    p_bub = np.array([rows[0, 0] for _, rows in deck["pvto"]])
    rs_t = np.concatenate([[0.0], keys])
    p_t = np.concatenate([[0.0], p_bub])

    b_o = lambda p, rs, flag: pvt_interp(bo_tab, p, rs, flag)
    mu_o = lambda p, rs, flag: pvt_interp(muo_tab, p, rs, flag)
    rs_sat = lambda p: interp_table(p_t, rs_t, p)

    # Relative permeabilities (assignSWOF / assignSGOF + extendTab).
    swof = extend_tab(deck["swof"])
    sgof = extend_tab(deck["sgof"])
    swcon = float(deck["swof"][0, 0])

    sw_tab, krw_tab = swof[:, 0], swof[:, 1]
    so_w_tab = 1.0 - swof[::-1, 0]
    krow_tab = swof[::-1, 2]
    pcow_tab = swof[:, 3] if swof.shape[1] > 3 else None

    sg_tab, krg_tab = sgof[:, 0], sgof[:, 1]
    so_g_tab = 1.0 - (sgof[::-1, 0] + swcon)
    krog_tab = sgof[::-1, 2]
    pcog_tab = sgof[:, 3] if sgof.shape[1] > 3 else None

    fluid = BlackOilFluid(
        rhoS=deck["rhoS"],
        b_w=b_w, mu_w=mu_w,
        b_o=b_o, mu_o=mu_o, rs_sat=rs_sat,
        b_g=b_g, mu_g=mu_g,
        kr_w=lambda s: interp_table(sw_tab, krw_tab, s),
        kr_g=lambda s: interp_table(sg_tab, krg_tab, s),
        kr_ow=lambda s: interp_table(so_w_tab, krow_tab, s),
        kr_og=lambda s: interp_table(so_g_tab, krog_tab, s),
        pv_mult=pv_mult,
        swcon=swcon,
        pc_ow=(None if pcow_tab is None
               else lambda s: interp_table(sw_tab, pcow_tab, s)),
        pc_og=(None if pcog_tab is None
               else lambda s: interp_table(sg_tab, pcog_tab, s)),
    )
    jfunc = deck.get("jfunc") or deck.get("JFUNC")
    if jfunc is not None and rock is not None:
        from pymrst.modules.ad_blackoil.utils.j_function_scaling import (
            apply_j_function_scaling,
        )

        fluid = apply_j_function_scaling(
            fluid, rock, jfunc, deck.get("pcunit", deck.get("PCUNIT", 1.0))
        )
    return fluid


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.modules.deckformat import read_spe1_deck
    from pymrst.utils.units import PSIA

    DECK = Path(__file__).parents[4] / (
        "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
    )
    deck = read_spe1_deck(DECK)
    f = init_deck_blackoil_fluid(deck)

    STB = 0.158987294928
    MSCF = 1000.0 * 0.3048**3

    def test_water_and_rock():
        """bW at ref pressure is 1/B_ref; pv_mult slope is c_r."""
        assert np.isclose(f.b_w(np.array([4014.7 * PSIA]))[0], 1.0 / 1.029)
        p = np.array([14.7 * PSIA])
        assert np.isclose(f.pv_mult(p)[0], 1.0)
        print("ok test_water_and_rock")

    def test_gas_table():
        """bG hits table nodes: B(2014.7 psia) = 1.614 rb/Mscf."""
        bg = f.b_g(np.array([2014.7 * PSIA]))[0]
        assert np.isclose(1.0 / bg, 1.614 * STB / MSCF, rtol=1e-12)
        mug = f.mu_g(np.array([3014.7 * PSIA]))[0]
        assert np.isclose(mug, 0.0228e-3)
        print("ok test_gas_table")

    def test_oil_saturated_and_undersaturated():
        """bO on the saturated envelope and along the rs=1.27 branch."""
        rs127 = 1.27 * MSCF / STB
        p_bub = 4014.7 * PSIA
        flag_t, flag_f = np.array([True]), np.array([False])
        bo = f.b_o(np.array([p_bub]), np.array([rs127]), flag_t)
        assert np.isclose(1.0 / bo[0], 1.695, rtol=1e-12)
        bo_u = f.b_o(np.array([5014.7 * PSIA]), np.array([rs127]), flag_f)
        assert np.isclose(1.0 / bo_u[0], 1.671, rtol=1e-6)
        mu_u = f.mu_o(np.array([9014.7 * PSIA]), np.array([rs127]), flag_f)
        assert np.isclose(mu_u[0], 0.74e-3, rtol=1e-6)
        print("ok test_oil_saturated_and_undersaturated")

    def test_rs_sat_envelope():
        """rsSat interpolates the bubble-point curve."""
        assert np.isclose(f.rs_sat(np.array([4014.7 * PSIA]))[0],
                          1.27 * MSCF / STB, rtol=1e-12)
        mid = f.rs_sat(np.array([(4014.7 + 5014.7) / 2 * PSIA]))[0]
        assert np.isclose(mid, (1.27 + 1.618) / 2 * MSCF / STB, rtol=1e-12)
        print("ok test_rs_sat_envelope")

    def test_relperms():
        """Table nodes reproduce; three-phase krO blends krow/krog."""
        assert np.isclose(f.kr_w(np.array([0.52]))[0], 4.545454545e-6)
        assert np.isclose(f.kr_g(np.array([0.3]))[0], 0.19)
        assert np.isclose(f.kr_ow(np.array([1 - 0.24]))[0], 0.7)
        assert np.isclose(f.kr_og(np.array([1 - 0.12 - 0.2]))[0], 0.35)
        # two-phase limit sg=0: krO == krow
        sw, sg = np.array([0.3]), np.array([0.0])
        so = 1 - sw - sg
        assert np.isclose(f.kr_o(sw, so, sg)[0], f.kr_ow(so)[0])
        print("ok test_relperms")

    test_water_and_rock()
    test_gas_table()
    test_oil_saturated_and_undersaturated()
    test_rs_sat_envelope()
    test_relperms()
    print("\nAll tests passed!")
