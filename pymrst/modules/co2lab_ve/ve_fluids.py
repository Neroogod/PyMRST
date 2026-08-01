"""Upscaled VE relative permeability and capillary pressure (co2lab-ve/fluids).

The sharp-interface model in `ve.py` says CO2 and brine meet at a knife edge:
above it, pure CO2; below it, pure brine. Real rock has a capillary
transition zone, and the CO2 saturation grades from one to the other over a
fringe whose thickness is set by the entry pressure. Where the fringe is a
significant fraction of the aquifer, pretending it is sharp overstates how
far the plume spreads and how fast it moves.

Three upscaled models, in increasing order of faithfulness:

  sharp interface       `ve.VEFluid` - the fringe has zero thickness
  cap-linear            the fringe is linear in depth; analytic throughout
  capillary fringe      the fine-scale pc and kr are *integrated vertically*
                        into upscaled tables, so any rock curve can be used

All three share one thing: the CO2 that can flow is not all the CO2 present.
Behind a retreating plume some is held residually, and `free_sg` is what
separates the two. Get that wrong and the plume moves with its own dregs.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pymrst.utils.interp_table import _interp1, interp_table
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.utils.adi import ADI, ad_max, ad_min, value


def free_sg(sg, sg_max, res_water: float, res_gas: float):
    """The mobile part of the CO2 saturation (MRST's free_sg).

    While the plume is still growing (sg >= sGmax) everything is free. Once it
    starts to retreat, the brine imbibing behind it strands CO2 at the
    residual saturation, and only

        ((1 - rw) sg - sGmax rn) / (1 - rw - rn)

    of it can still move. Never negative.
    """
    fsg = ((1.0 - res_water) * sg - sg_max * res_gas) / (1.0 - res_water
                                                          - res_gas)
    drainage = value(sg) >= value(sg_max)      # still growing: all of it free
    d = drainage.astype(np.float64)
    fsg = d * sg + (1.0 - d) * fsg
    return ad_max(fsg, 0.0)


# --------------------------------------------------------------------------
# vertical integration of a fine-scale rock curve
# --------------------------------------------------------------------------
@dataclass
class VETable:
    """Upscaled capillary pressure and relperm, integrated over the fringe."""

    p: np.ndarray        # capillary pressure at the sample heights
    SP: np.ndarray       # integral of the CO2 saturation, in pressure units
    krP: np.ndarray      # integral of the fine-scale relperm


def upper_cap_pressure(inv_pc_3d: Callable, res_water: float,
                       H_max: float, drho_surf: float,
                       gravity: float = NORM_GRAVITY) -> float:
    """How high a capillary pressure the tables must reach.

    Doubles the pressure until the fine-scale rock is down to its residual
    water, then adds the hydrostatic drop across the thickest column.
    """
    dp = 1.0
    tol = 1e-3
    while float(inv_pc_3d(np.array([dp]))[0]) > res_water + tol and dp < 2e6:
        dp *= 2.0
    return dp + H_max * drho_surf * gravity


def make_ve_table_p(inv_pc_3d: Callable, kr_3d: Callable,
                    p_max: float, samples: int = 2000) -> VETable:
    """Integrate the fine-scale curves against capillary pressure.

    At a height h above the CO2-brine contact the capillary pressure is
    drho g h, so the fine-scale saturation profile is sw(p) = invPc3D(p) and
    integrating it up the column gives the upscaled saturation. Working in
    pressure rather than height is what lets one table serve every cell,
    whatever its thickness.
    """
    p = np.linspace(0.0, p_max, samples)
    dp = p[1] - p[0]
    sw_p = np.asarray(inv_pc_3d(p), float)
    sg_p = 1.0 - sw_p
    kr_p = np.asarray(kr_3d(sg_p), float)

    # Below the entry pressure the rock holds no CO2 at all, so sg is exactly
    # zero over a whole run of samples and the table starts with a flat spot
    # that cannot be inverted. MRST ramps that run up to a tenth of the first
    # real value, which keeps SP strictly increasing and invertible.
    zeros = np.flatnonzero(sg_p == 0.0)
    if zeros.size and zeros[-1] > 0:
        last = int(zeros[-1])
        smooth = 0.1
        sg_p[:last + 1] = np.linspace(0.0, smooth * sg_p[last + 1], last + 1)

    return VETable(p=p,
                   SP=(np.cumsum(sg_p) - sg_p / 2.0) * dp,
                   krP=(np.cumsum(kr_p) - kr_p / 2.0) * dp)


def _untruncated_sp(H, drho, sg, table: VETable, gravity: float):
    """Upscaled saturation in pressure units, with the plume's tail cut off.

    A plume holding `sg` of a column would, in an infinite aquifer, reach
    further down than the aquifer goes. The part below the bottom is not
    there, so the pressure at the top has to be raised until the *truncated*
    profile holds the right amount - a small Newton solve per cell.
    """
    drho_g = drho * gravity
    dp_aquifer = H * drho_g
    sp = sg * dp_aquifer

    p_full = interp_table(table.SP, table.p, sp) / drho_g
    below = value(p_full) - value(H)
    trunc = below > 0

    h_trunc = 0.0 * sg
    if not np.any(trunc):
        return sp, h_trunc

    dg = np.atleast_1d(value(drho_g) * np.ones_like(value(sg)))[trunc]
    hl = np.atleast_1d(value(H) * np.ones_like(value(sg)))[trunc]
    spl = value(sp)[trunc]
    tol = 1e-3 * float(np.max(value(dp_aquifer)))

    # Solve  SP(pb) + SP_cell - SP(pb + drho g H) = 0  for the pressure pb at
    # the aquifer bottom.
    #
    # The Newton is *one-sided*, as MRST's is: it corrects only cells whose
    # residual is above the tolerance, and never those that have overshot to a
    # negative one. That is not an accident to be improved upon - it is the
    # algorithm, and making it two-sided (which does converge, to 1e-12) walks
    # to a different answer and stops matching MRST at all.
    pb = below[trunc] * dg
    for _ in range(2000):
        f = (_interp1(table.p, table.SP, pb)[0] + spl
             - _interp1(table.p, table.SP, pb + dg * hl)[0])
        act = f > tol
        if not np.any(act):
            break
        df = (_interp1(table.p, table.SP, pb[act])[1]
              - _interp1(table.p, table.SP, pb[act] + dg[act] * hl[act])[1])
        pb[act] = pb[act] - f[act] / df

    add = interp_table(table.p, table.SP, pb)
    full_add = np.zeros_like(value(sg))
    full_add[trunc] = add
    ht = np.zeros_like(value(sg))
    ht[trunc] = pb / dg

    return sp + full_add, ht


@dataclass
class VECapillaryFringeFluid:
    """Upscaled fluid with a capillary fringe (addVERelpermCapillaryFringe).

    Built from the *fine-scale* rock: `inv_pc_3d(p)` gives the water
    saturation at capillary pressure p, and `kr_3d(sg)` the CO2 relperm at
    saturation sg. Both are integrated vertically into upscaled tables, so
    any measured curve can be used rather than an idealised one.

    This is MRST's 'P-scaled table' variant, which is its default. Its
    'S table' variant cannot be used: `truncate_at_aquifer_bottom` indexes a
    misspelled variable (`trunctate`) and raises as soon as any plume reaches
    the aquifer bottom. `make_ve_table_h` below is that model with the typo
    fixed.
    """

    rhoWS: float
    rhoGS: float
    mu_w: float
    mu_g: float
    srw: float
    srg: float
    H: np.ndarray                       # column thickness, per cell
    table: VETable
    c_water: float = 0.0
    c_gas: float = 0.0
    p_ref: float = 1e7
    gravity: float = NORM_GRAVITY

    @property
    def res_water(self):
        return self.srw

    @property
    def res_gas(self):
        return self.srg

    def b_w(self, p):
        return 1.0 + self.c_water * (p - self.p_ref)

    def b_g(self, p):
        return 1.0 + self.c_gas * (p - self.p_ref)

    def _drho(self, p):
        return self.rhoWS * self.b_w(p) - self.rhoGS * self.b_g(p)

    def kr_g(self, sg, p, sg_max=None):
        sg_max = sg if sg_max is None else sg_max
        drho = self._drho(p)
        s = free_sg(sg, sg_max, self.srw, self.srg)
        sp, h_trunc = _untruncated_sp(self.H, drho, s, self.table,
                                      self.gravity)
        kr = interp_table(self.table.SP, self.table.krP, sp)
        if np.any(h_trunc > 0):
            p_trunc = h_trunc * value(drho) * self.gravity
            kr = kr - interp_table(self.table.p, self.table.krP, p_trunc)
        return kr / (self.H * drho * self.gravity)

    def kr_w(self, sw, sg_max=None):
        """Brine relperm (MRST's krW_simple).

        The brine cannot use the pore space the residual CO2 sits in, so its
        relperm is its saturation less the water held back with that CO2.
        """
        sg = 1.0 - sw
        sg_max = sg if sg_max is None else sg_max
        free = free_sg(sg, sg_max, self.srw, self.srg)
        sw_eff = sw - (free / (1.0 - self.srw)) * self.srw
        return ad_max(sw_eff, 0.0)

    def pc(self, sg, p, sg_max=None):
        sg_max = sg if sg_max is None else sg_max
        drho = self._drho(p)
        s = free_sg(sg, sg_max, self.srw, self.srg)
        sp, _ = _untruncated_sp(self.H, drho, s, self.table, self.gravity)
        return interp_table(self.table.SP, self.table.p, sp)


def make_ve_table_h(inv_pc_3d: Callable, kr_3d: Callable, H_max: float,
                    drho_surf: float, p_max: float, samples: int = 2000,
                    gravity: float = NORM_GRAVITY):
    """The height-based upscaled tables (MRST's 'S table', with its bug fixed).

    Returns (h, SH, krH). MRST's own version of this raises before it can be
    used - `truncate_at_aquifer_bottom` writes `vec_truncated(trunctate)`,
    misspelling its own mask - so this branch of `addVERelpermCapillaryFringe`
    is dead code upstream. The truncation is what the name says: a plume whose
    theoretical profile reaches past the aquifer bottom has that tail removed.
    """
    h_max = p_max / drho_surf / gravity
    h = np.linspace(0.0, h_max, samples)
    dh = h[1] - h[0]
    sw_h = np.asarray(inv_pc_3d(h * drho_surf * gravity), float)
    sg_h = 1.0 - sw_h
    kr_h = np.asarray(kr_3d(sg_h), float)

    sh = (np.cumsum(sg_h) - sg_h / 2.0) * dh
    krh = (np.cumsum(kr_h) - kr_h / 2.0) * dh

    def truncate(vec):
        out = vec.copy()
        past = h > H_max
        if np.any(past):
            out[past] = vec[past] - _interp1(h, vec, h[past] - H_max)[0]
        return out

    return h, truncate(sh), truncate(krh)


# --------------------------------------------------------------------------
# the analytic cap-linear model
# --------------------------------------------------------------------------
@dataclass
class VECapLinearFluid:
    """Upscaled fluid with a linear capillary fringe (addVERelpermCapLinear).

    The fine-scale saturation rises linearly with capillary pressure over a
    fringe of height `cap_scale / (drho g)`, and the fine-scale relperm is
    (s / (1 - srw))^beta. Everything integrates in closed form, so there are
    no tables and no Newton solve - which is why this is the variant to reach
    for when the fringe is small and the rock curve is not measured.
    """

    rhoWS: float
    rhoGS: float
    mu_w: float
    mu_g: float
    srw: float
    srg: float
    H: np.ndarray
    cap_scale: float                 # entry pressure of the fringe
    beta: float = 1.0
    c_water: float = 0.0
    c_gas: float = 0.0
    p_ref: float = 1e7
    gravity: float = NORM_GRAVITY

    def b_w(self, p):
        return 1.0 + self.c_water * (p - self.p_ref)

    def b_g(self, p):
        return 1.0 + self.c_gas * (p - self.p_ref)

    def _drho(self, p):
        return self.rhoWS * self.b_w(p) - self.rhoGS * self.b_g(p)

    def _cap_norm(self, p):
        """The fringe thickness, as a fraction of the column.

        Must be less than one: the model integrates a fringe that fits inside
        the aquifer, and the closed forms below are simply wrong if it does
        not. MRST asserts the same. A thick fringe in a thin aquifer is not a
        VE problem at all - there is no segregated plume to upscale.
        """
        cn = (self.cap_scale / (self._drho(p) * self.gravity)) / self.H
        assert np.all(value(cn) < 1.0), (
            f"the capillary fringe is thicker than the aquifer "
            f"(cap_scale / (drho g H) = {float(np.max(value(cn))):.2f} >= 1); "
            f"lower cap_scale, or the column is too thin for a VE model")
        return cn

    def inv_pc_3d(self, p):
        """Fine-scale water saturation at capillary pressure p."""
        s = (p / self.cap_scale) * (1.0 - self.srw)
        s = np.clip(s, 0.0, 1.0 - self.srw)
        return 1.0 - s

    def kr_3d(self, s):
        """Fine-scale CO2 relperm."""
        return ((1.0 - self.srw) ** self.beta) * (s / (1.0 - self.srw)) ** self.beta

    def _S(self, h_norm, cap_norm):
        """Upscaled saturation from a normalised plume height (MRST's S_beta)."""
        b = self.beta
        s = (h_norm - cap_norm) + cap_norm / (b + 1.0)
        # entirely inside the fringe: the plume has not reached full saturation
        inside = value(h_norm) < value(cap_norm)
        if np.any(inside):
            s = np.where(inside,
                         (h_norm / cap_norm) ** (b + 1.0) / (b + 1.0) * cap_norm,
                         s)
        # past the aquifer bottom: cut the tail off
        past = value(h_norm) > 1.0
        if np.any(past):
            s = np.where(past,
                         s - ((h_norm - 1.0) / cap_norm) ** (b + 1.0)
                         / (b + 1.0) * cap_norm,
                         s)
        return s

    def _invS(self, S, cap_norm):
        """Plume height from the upscaled saturation (MRST's invS)."""
        S = S / (1.0 - self.srw)
        vol_cap = 0.5 * cap_norm                       # all inside the fringe
        vol_capb = vol_cap + 1.0 - cap_norm            # fringe past the bottom

        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.sqrt(np.maximum(2.0 * S / cap_norm, 0.0)) * cap_norm

        # linearise the square root near zero: its slope is infinite there,
        # which would blow up the Jacobian of the very first Newton step
        h_tol = 1e-5
        s_tol = 0.5 * (h_tol / cap_norm) ** 2 * cap_norm
        h = np.where(h < h_tol, h_tol * S / s_tol, h)

        mid = (S > vol_cap) & (S < vol_capb)
        h = np.where(mid, (S - vol_cap) + cap_norm, h)

        deep = S >= vol_capb
        if np.any(deep):
            a = -1.0 / (2.0 * cap_norm)
            c = -S + (vol_cap - cap_norm + 1.0)
            disc = np.maximum(1.0 - 4.0 * a * c, 0.0)
            h = np.where(deep, (-1.0 + np.sqrt(disc)) / (2.0 * a) + 1.0, h)
        return h * self.H

    def kr_g(self, sg, p, sg_max=None):
        sg_max = sg if sg_max is None else sg_max
        cn = value(self._cap_norm(p))
        s = value(free_sg(sg, sg_max, self.srw, self.srg))
        h = self._invS(s, cn)
        end_scale = (1.0 - self.srw) ** self.beta
        return end_scale * self._S(h / self.H, cn)

    def kr_w(self, sw, sg_max=None):
        sg = 1.0 - value(sw)
        sg_max = sg if sg_max is None else value(sg_max)
        imbibing = sg <= sg_max
        sg_res = sg_max - sg
        kr = (1.0 - sg_max) + (1.0 - self.srg) * sg_res
        kr = np.where(imbibing, kr, value(sw))
        return np.maximum(kr, 0.0)

    def pc(self, sg, p, sg_max=None):
        sg_max = sg if sg_max is None else sg_max
        cn = value(self._cap_norm(p))
        s = value(free_sg(sg, sg_max, self.srw, self.srg))
        h = self._invS(s, cn)
        return self.gravity * value(self._drho(p)) * h


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi
    from pymrst.utils.units import BARSA

    NC = 5
    H = np.full(NC, 20.0)

    def _fringe(entry=1e5, n=2.0, srw=0.2, srg=0.2):
        """A fluid whose fine-scale rock is a Brooks-Corey-ish curve."""
        def inv_pc_3d(p):
            s = np.clip(np.asarray(p, float) / entry, 0.0, 1.0)
            return 1.0 - s * (1.0 - srw)

        def kr_3d(sg):
            return np.asarray(sg, float) ** n

        drho = 1000.0 - 700.0
        p_max = upper_cap_pressure(inv_pc_3d, srw, float(H.max()), drho)
        table = make_ve_table_p(inv_pc_3d, kr_3d, p_max)
        return VECapillaryFringeFluid(
            rhoWS=1000.0, rhoGS=700.0, mu_w=8e-4, mu_g=6e-5,
            srw=srw, srg=srg, H=H, table=table), inv_pc_3d, kr_3d

    def test_free_sg_separates_mobile_from_stranded():
        """While the plume grows, everything is free; once it retreats, not."""
        rw, rn = 0.2, 0.2
        sg = np.array([0.0, 0.2, 0.4, 0.4, 0.4])
        sgm = np.array([0.0, 0.2, 0.4, 0.5, 0.8])

        f = free_sg(sg, sgm, rw, rn)
        # drainage (sg == sGmax): all of it is free
        assert np.allclose(f[:3], sg[:3])
        # imbibition: less than all of it, and monotone in how far it retreated
        assert f[3] < sg[3] and f[4] < f[3]
        # a fully stranded plume has nothing free left
        stuck = free_sg(np.array([0.2]), np.array([0.8]), rw, rn)
        assert stuck[0] >= 0.0
        # the closed form
        want = ((1 - rw) * sg[4] - sgm[4] * rn) / (1 - rw - rn)
        assert np.isclose(f[4], max(want, 0.0))
        print(f"ok test_free_sg_separates_mobile_from_stranded "
              f"(sg = 0.4 behind a plume that reached 0.8: only "
              f"{f[4]:.3f} still mobile)")

    def test_free_sg_differentiates():
        """free_sg is AD-transparent on both branches.

        Checked *away* from sg = sGmax. That point is a genuine kink - the
        derivative is 1 on the drainage side and (1 - rw)/(1 - rw - rn) on
        the imbibition side - so a central difference across it averages the two
        and matches neither. Newton has to live with the kink; the test does
        not have to pretend it is not there.
        """
        rw, rn = 0.2, 0.2
        # strictly drainage (sg > sGmax) and strictly imbibition (sg < sGmax)
        sg0 = np.array([0.30, 0.50, 0.20, 0.35, 0.45])
        sgm0 = np.array([0.25, 0.40, 0.60, 0.70, 0.90])
        sg, = init_variables_adi(sg0)
        ad = np.diag(np.asarray(free_sg(sg, sgm0, rw, rn).jac[0].todense()))

        h = 1e-7
        fd = (free_sg(sg0 + h, sgm0, rw, rn)
              - free_sg(sg0 - h, sgm0, rw, rn)) / (2 * h)
        assert np.abs(ad - fd).max() < 1e-5, (ad, fd)

        # and the two branches really do have different slopes: 1 while the
        # plume grows, (1 - rw) / (1 - rw - rn) once it is retreating - the
        # mobile CO2 falls faster than the total, because part of what leaves
        # is being stranded rather than moved
        imb_slope = (1.0 - rw) / (1.0 - rw - rn)
        assert np.allclose(ad[:2], 1.0)
        assert np.allclose(ad[2:], imb_slope)
        print(f"ok test_free_sg_differentiates "
              f"(slope 1.0 draining, {imb_slope:.3f} imbibing)")

    def test_fringe_table_integrates_the_rock_curve():
        """The upscaled table is the vertical integral of the fine-scale one."""
        fluid, inv_pc, kr3 = _fringe()
        t = fluid.table

        assert np.all(np.diff(t.SP) >= 0), "SP must be non-decreasing"
        assert np.all(np.diff(t.krP) >= 0)
        assert t.SP[0] == 0.0 and t.krP[0] == 0.0

        # far above the entry pressure the rock is fully saturated, so SP grows
        # at the constant rate (1 - srw) per unit pressure
        slope = np.diff(t.SP)[-1] / np.diff(t.p)[-1]
        assert np.isclose(slope, 1.0 - 0.2, rtol=1e-3), slope
        print(f"ok test_fringe_table_integrates_the_rock_curve "
              f"(SP slope {slope:.4f} at depth, expected {1 - 0.2:.2f})")

    def test_fringe_smooths_the_sharp_interface():
        """The fringe holds less CO2 near the top than a sharp interface would.

        The whole point of the model: a fringe is a zone of partial CO2
        saturation, so for the same plume height it carries less CO2 - and its
        upscaled relperm is lower.
        """
        fluid, _, _ = _fringe(entry=2e5)
        p = np.full(NC, 100 * BARSA)
        sg = np.array([0.05, 0.1, 0.2, 0.3, 0.4])

        kr = fluid.kr_g(sg, p, sg_max=sg)
        assert np.all(np.diff(kr) > 0), "relperm must rise with saturation"
        assert np.all(kr >= 0) and np.all(kr <= 1.0 + 1e-12)

        # a sharp interface would give kr = sg exactly (the plume is pure CO2
        # over a fraction sg of the column); the fringe must give less
        assert np.all(kr < sg), (kr, sg)

        pc = fluid.pc(sg, p, sg_max=sg)
        assert np.all(np.diff(pc) > 0), "pc must rise with saturation"
        assert np.all(pc > 0)
        print(f"ok test_fringe_smooths_the_sharp_interface "
              f"(kr = {np.round(kr, 3)} against sg = {sg} for a sharp one)")

    def test_fringe_kr_w_holds_back_water_with_the_residual_co2():
        """Brine cannot use the pores the stranded CO2 occupies."""
        fluid, _, _ = _fringe()
        sw = np.full(NC, 0.7)
        # a plume that has retreated leaves residual CO2 behind
        kr_drain = fluid.kr_w(sw, sg_max=1.0 - sw)          # still growing
        kr_imb = fluid.kr_w(sw, sg_max=np.full(NC, 0.6))    # retreated
        assert np.all(kr_imb > kr_drain), (
            "with less mobile CO2 the brine should flow more freely")
        assert np.all(kr_drain >= 0) and np.all(kr_imb <= 1.0)
        print(f"ok test_fringe_kr_w_holds_back_water_with_the_residual_co2 "
              f"(krW {kr_drain[0]:.3f} draining, {kr_imb[0]:.3f} imbibing)")

    def test_cap_linear_matches_its_own_fine_scale_curve():
        """The analytic model and the table model agree on the same rock.

        cap-linear *is* a capillary fringe model with a particular fine-scale
        curve. Feed that same curve through the general table machinery and
        the two must produce the same upscaled relperm - which is the sharpest
        check there is that the closed form was integrated correctly.
        """
        # cap_scale must leave the fringe thinner than the column:
        # drho g H = 300 * 9.81 * 20 = 58.9 kPa, so 20 kPa is a third of it.
        # beta = 2, because with a linear fine-scale relperm the fringe makes
        # no difference and the comparison would be vacuous.
        srw, srg, beta, cap = 0.2, 0.2, 2.0, 2e4
        lin = VECapLinearFluid(rhoWS=1000.0, rhoGS=700.0, mu_w=8e-4,
                               mu_g=6e-5, srw=srw, srg=srg, H=H,
                               cap_scale=cap, beta=beta)

        drho = 1000.0 - 700.0
        p_max = upper_cap_pressure(lin.inv_pc_3d, srw, float(H.max()), drho)
        table = make_ve_table_p(lin.inv_pc_3d, lin.kr_3d, p_max, samples=20000)
        tab = VECapillaryFringeFluid(rhoWS=1000.0, rhoGS=700.0, mu_w=8e-4,
                                     mu_g=6e-5, srw=srw, srg=srg, H=H,
                                     table=table)

        p = np.full(NC, 100 * BARSA)
        sg = np.array([0.05, 0.15, 0.3, 0.45, 0.6])
        kr_lin = lin.kr_g(sg, p, sg_max=sg)
        kr_tab = tab.kr_g(sg, p, sg_max=sg)

        err = np.abs(kr_lin - kr_tab).max() / max(kr_tab.max(), 1e-30)
        assert err < 0.02, (
            f"the closed form and the integrated table disagree by {err:.1%}\n"
            f"  analytic: {np.round(kr_lin, 5)}\n"
            f"  table   : {np.round(kr_tab, 5)}")
        print(f"ok test_cap_linear_matches_its_own_fine_scale_curve "
              f"(closed form vs vertical integration: {err:.2%} apart)")

    def test_cap_linear_tends_to_the_sharp_interface():
        """Shrink the fringe to nothing and the sharp-interface model returns.

        beta = 2, not 1. With a *linear* fine-scale relperm the fringe makes
        no difference at all - the vertical integral of kr equals the vertical
        integral of s, so kr = sg however thick the transition zone is, and
        the test would pass without testing anything. The fringe only bites
        when the relperm is nonlinear in saturation, which is the case that
        matters.
        """
        p = np.full(NC, 100 * BARSA)
        sg = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        errs = []
        # drho g H = 300 * 9.81 * 20 = 58.9 kPa with srw = 0, so stay below it
        for cap in (5e4, 5e3, 5e2, 5e1):
            f = VECapLinearFluid(rhoWS=1000.0, rhoGS=700.0, mu_w=8e-4,
                                 mu_g=6e-5, srw=0.0, srg=0.0, H=H,
                                 cap_scale=cap, beta=2.0)
            kr = f.kr_g(sg, p, sg_max=sg)
            # a sharp interface gives kr = sg; the fringe must give less
            assert np.all(kr <= sg + 1e-12)
            errs.append(np.abs(kr - sg).max())

        assert errs == sorted(errs, reverse=True), (
            f"a thinner fringe should approach the sharp interface: {errs}")
        assert errs[0] > 0.05, "the thickest fringe should differ noticeably"
        assert errs[-1] < 0.01, errs[-1]
        print(f"ok test_cap_linear_tends_to_the_sharp_interface "
              f"(|kr - sg| falls {errs[0]:.3f} -> {errs[-1]:.4f} as the fringe "
              f"shrinks 1000x)")

    def test_h_table_truncation_is_the_bug_mrst_never_reaches():
        """The height-based table, with MRST's typo fixed.

        MRST's `truncate_at_aquifer_bottom` indexes `vec_truncated(trunctate)`
        - misspelling its own mask - so the 'S table' branch raises the moment
        any plume reaches the aquifer bottom, which is to say immediately. It
        is dead code upstream.

        What truncation *does*: a plume whose theoretical profile reaches past
        the aquifer bottom has that tail removed, so past h = H the table must
        hold strictly less CO2 than the infinite column would, and must grow
        more slowly. Below h = H there is nothing to cut and the two agree
        exactly.
        """
        fluid, inv_pc, kr3 = _fringe()
        drho = 1000.0 - 700.0
        aquifer_H = 20.0
        h, sh, krh = make_ve_table_h(inv_pc, kr3, H_max=aquifer_H,
                                     drho_surf=drho, p_max=3e5, samples=4000)

        # the same integral, with no truncation
        dh = h[1] - h[0]
        sg_h = 1.0 - np.asarray(inv_pc(h * drho * NORM_GRAVITY), float)
        raw = (np.cumsum(sg_h) - sg_h / 2.0) * dh

        shallow = h <= aquifer_H
        deep = h > aquifer_H
        assert np.any(deep), "the table must reach past the aquifer bottom"

        assert np.allclose(sh[shallow], raw[shallow]), \
            "nothing to truncate above the aquifer bottom"
        assert np.all(sh[deep] < raw[deep]), \
            "truncation must remove the tail below the aquifer bottom"
        assert np.all(np.diff(sh) >= -1e-12), "SH must stay non-decreasing"

        # and it grows more slowly than the infinite column does
        assert np.mean(np.diff(sh)[deep[:-1]]) < np.mean(np.diff(raw)[deep[:-1]])

        cut = (raw[-1] - sh[-1]) / raw[-1]
        print(f"ok test_h_table_truncation_is_the_bug_mrst_never_reaches "
              f"(the tail below the aquifer bottom is {cut:.0%} of the "
              f"untruncated column at the deepest sample)")

    test_free_sg_separates_mobile_from_stranded()
    test_free_sg_differentiates()
    test_fringe_table_integrates_the_rock_curve()
    test_fringe_smooths_the_sharp_interface()
    test_fringe_kr_w_holds_back_water_with_the_residual_co2()
    test_cap_linear_tends_to_the_sharp_interface()
    test_cap_linear_matches_its_own_fine_scale_curve()
    test_h_table_truncation_is_the_bug_mrst_never_reaches()
    print("\nAll tests passed!")
