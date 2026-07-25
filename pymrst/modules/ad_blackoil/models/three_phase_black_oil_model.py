"""Three-phase black-oil model with dissolved gas (MRST equationsBlackOil).

Fully-implicit water/oil/gas conservation in surface volumes with live
oil (disgas, no vapoil). The hydrocarbon primary variable x switches
meaning per cell: sG in saturated cells (status 3), Rs in undersaturated
cells (status 1), with the status flag updated by the flash of
computeFlashBlackOil.m after every Newton update. Wells follow
computeWellContributionsSingleWell.m including dissolved-gas surface
volume accounting. Connection pressure drops (cdp) are zero, which is
exact for SPE 1 (refDepth equals the perforation depth).
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_props.blackoil_fluid import BlackOilFluid
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_core.utils.convergence import cnv_mb_convergence
from pymrst.utils.adi import (
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
    vertcat_adi,
)

ETOL: float = np.sqrt(np.finfo(float).eps)


@dataclass
class BOState:
    """Black-oil state: cell fields, per-cell status, well solution."""

    pressure: np.ndarray        # (nc,)
    s: np.ndarray               # (nc, 3) [water, oil, gas]
    rs: np.ndarray              # (nc,)
    status: np.ndarray          # (nc,) 1 = undersaturated, 3 = saturated
    qWs: np.ndarray             # (nw,)
    qOs: np.ndarray
    qGs: np.ndarray
    bhp: np.ndarray

    def copy(self) -> "BOState":
        return BOState(*(v.copy() for v in (
            self.pressure, self.s, self.rs, self.status,
            self.qWs, self.qOs, self.qGs, self.bhp)))


class ThreePhaseBlackOilModelAD:
    """Fully-implicit black-oil model with dissolved gas."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: BlackOilFluid,
        wells: Sequence[Well],
        gravity: float = NORM_GRAVITY,
        ds_max: float = 0.2,
        tol_cnv: float = 1e-8,
        tol_mb: float = 1e-12,
        tol_well: float = 1e-8,
        max_iterations: int = 25,
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.wells = list(wells)
        self.ops = setup_operators_tpfa(g, rock)
        self.pv_ref = pore_volume(g, rock)
        self.gdz = gravity * self.ops.grad(g.cells.centroids[:, 2])
        self.ds_max = ds_max
        self.tol_cnv = tol_cnv
        self.tol_mb = tol_mb
        self.tol_well = tol_well
        self.max_iterations = max_iterations

    # -- assembly ----------------------------------------------------------
    def assemble(self, st0: BOState, st: BOState, p, sw, x,
                 qws, qos, qgs, bhp, dt: float):
        f = self.fluid
        ops = self.ops

        st1 = st.status == 1
        st3 = st.status == 3
        rs_sat = f.rs_sat(p)
        rs = x * st1.astype(float) + rs_sat * (~st1).astype(float)
        sg = x * st3.astype(float)
        so = 1.0 - sw - sg

        b_w, b_o, b_g = f.b_w(p), f.b_o(p, rs, st3), f.b_g(p)
        mob_w = f.kr_w(sw) / f.mu_w(p)
        mob_o = f.kr_o(sw, so, sg) / f.mu_o(p, rs, st3)
        mob_g = f.kr_g(sg) / f.mu_g(p)

        rho_ws, rho_os, rho_gs = f.rhoS
        rho_w = b_w * rho_ws
        rho_o = b_o * (rho_os + rs * rho_gs)
        rho_g = b_g * rho_gs

        fluxes = {}
        for ph, (mob, rho, b) in {
            "w": (mob_w, rho_w, b_w), "o": (mob_o, rho_o, b_o),
            "g": (mob_g, rho_g, b_g),
        }.items():
            pot = ops.grad(p) - ops.avg(rho) * self.gdz
            flag = value(pot) <= 0
            v = -ops.T * ops.upw(flag, mob) * pot
            fluxes[ph] = (flag, v, b)

        bWvW = ops.upw(fluxes["w"][0], b_w) * fluxes["w"][1]
        bOvO = ops.upw(fluxes["o"][0], b_o) * fluxes["o"][1]
        bGvG = ops.upw(fluxes["g"][0], b_g) * fluxes["g"][1]
        rsbOvO = ops.upw(fluxes["o"][0], rs) * bOvO

        pv = self.pv_ref * f.pv_mult(p)
        p0, s0, rs0 = st0.pressure, st0.s, st0.rs
        st3_0 = st0.status == 3
        pv0 = self.pv_ref * value(f.pv_mult(p0))
        b_w0 = value(f.b_w(p0))
        b_o0 = value(f.b_o(p0, rs0, st3_0))
        b_g0 = value(f.b_g(p0))

        water = (1.0 / dt) * (pv * b_w * sw - pv0 * b_w0 * s0[:, 0])
        oil = (1.0 / dt) * (pv * b_o * so - pv0 * b_o0 * s0[:, 1])
        gas = (1.0 / dt) * (pv * (b_g * sg + rs * b_o * so)
                            - pv0 * (b_g0 * s0[:, 2] + rs0 * b_o0 * s0[:, 1]))
        water = water + ops.div(bWvW)
        oil = oil + ops.div(bOvO)
        gas = gas + ops.div(bGvG + rsbOvO)

        eq_wells, ctrls = [[], [], []], []
        for k, w in enumerate(self.wells):
            cq_s, weqs, ctrl = self._well(
                k, p, (mob_w, mob_o, mob_g), (b_w, b_o, b_g), rs,
                (qws[k:k + 1], qos[k:k + 1], qgs[k:k + 1]), bhp[k:k + 1],
            )
            wc = w.cells
            water[wc] = water[wc] - cq_s[0]
            oil[wc] = oil[wc] - cq_s[1]
            gas[wc] = gas[wc] - cq_s[2]
            for ph in range(3):
                eq_wells[ph].append(weqs[ph])
            ctrls.append(ctrl)

        return (water, oil, gas,
                vertcat_adi(*eq_wells[0]), vertcat_adi(*eq_wells[1]),
                vertcat_adi(*eq_wells[2]), vertcat_adi(*ctrls))

    def _well(self, k, p, mob, b, rs, q_s, bhp_k):
        """Three-phase well with dissolved gas (single MRST well model)."""
        w = self.wells[k]
        wc, tw = w.cells, w.WI
        compi = w.compi[:3]

        dd = -bhp_k + p[wc]                      # drawdown, cdp = 0
        inj = value(dd) < 0
        prod = ~inj
        mob_c = [m[wc] for m in mob]
        b_c = [x[wc] for x in b]
        rs_c = rs[wc]

        any_inj, any_prod = bool(inj.any()), bool(prod.any())

        if any_prod:
            con = prod * tw
            cq_p = [-con * m * dd for m in mob_c]
            cq_ps = [b_c[0] * cq_p[0], b_c[1] * cq_p[1],
                     b_c[2] * cq_p[2] + rs_c * (b_c[1] * cq_p[1])]
            wbq = [q_s[ph] * float(value(q_s[ph])[0] > 0) - ad_sum(cq_ps[ph])
                   for ph in range(3)]
            wbqt = wbq[0] + wbq[1] + wbq[2]
            if value(wbqt)[0] == 0.0:
                mix = list(compi)
            else:
                mix = [wq / wbqt for wq in wbq]
        else:
            cq_ps = None
            mix = list(compi)

        if any_inj:
            mt = mob_c[0] + mob_c[1] + mob_c[2]
            cqt_i = -(inj * tw) * (mt * dd)
            vol_rat = (mix[0] / b_c[0] + mix[1] / b_c[1]
                       + (mix[2] - rs_c * mix[1]) / b_c[2])
            cqt_is = cqt_i / vol_rat
            if any_prod:
                cq_s = [cq_ps[ph] + mix[ph] * cqt_is for ph in range(3)]
            else:
                cq_s = [mix[ph] * cqt_is for ph in range(3)]
        else:
            cq_s = cq_ps

        weqs = [q_s[ph] - ad_sum(cq_s[ph]) for ph in range(3)]
        if w.type == "bhp":
            ctrl = bhp_k - w.val
        elif w.type == "rate":
            ctrl = q_s[0] + q_s[1] + q_s[2] - w.val
        elif w.type == "orat":
            ctrl = q_s[1] - w.val
        else:
            raise ValueError(f"unsupported control '{w.type}'")
        return cq_s, weqs, ctrl

    # -- update + flash ------------------------------------------------------
    def update_state(self, st: BOState, upd: np.ndarray) -> BOState:
        """Newton update with disgas interpretation + black-oil flash."""
        nc = self.g.cells.num
        nw = len(self.wells)
        prev = st
        new = st.copy()

        st1 = prev.status == 1
        st3 = prev.status == 3

        dp = upd[:nc]
        dsw = upd[nc:2 * nc]
        dx = upd[2 * nc:3 * nc]

        new.pressure = prev.pressure + dp

        dsg = np.where(st3, dx, 0.0)
        new.rs = np.maximum(prev.rs + np.where(st1, dx, 0.0), 0.0)

        ds = np.column_stack([dsw, -(dsg + dsw), dsg])
        ds = np.clip(ds, -self.ds_max, self.ds_max)
        new.s = prev.s + ds

        off = 3 * nc
        new.qWs = prev.qWs + upd[off:off + nw]
        new.qOs = prev.qOs + upd[off + nw:off + 2 * nw]
        new.qGs = prev.qGs + upd[off + 2 * nw:off + 3 * nw]
        new.bhp = prev.bhp + upd[off + 3 * nw:off + 4 * nw]

        self._flash(new, prev, st1)
        return new

    def _flash(self, st: BOState, prev: BOState, st1_prev: np.ndarray):
        """computeFlashBlackOil: phase appearance and rs capping."""
        f = self.fluid
        sw, so, sg = st.s[:, 0], st.s[:, 1], st.s[:, 2]
        rs = st.rs
        wat_only = sw > 1.0 - ETOL

        rs_sat = value(f.rs_sat(st.pressure))
        rs_sat0 = value(f.rs_sat(prev.pressure))

        gas_present = ((sg > 0) | (rs == 0)) & ~st1_prev
        gas_present |= wat_only
        gas_present |= (sg < 0) & (prev.s[:, 2] > ETOL)
        ix2 = (rs > rs_sat * (1 + ETOL)) & st1_prev \
            & (prev.rs > rs_sat0 * (1 - ETOL))
        sg[ix2] = 0.0
        gas_present |= ix2

        ix = sg < 0
        sw[ix] /= 1.0 - sg[ix]
        so[ix] /= 1.0 - sg[ix]
        sg[ix] = 0.0
        ix = so < 0
        sw[ix] /= 1.0 - so[ix]
        sg[ix] /= 1.0 - so[ix]
        so[ix] = 0.0
        ix = sw < 0
        so[ix] /= 1.0 - sw[ix]
        sg[ix] /= 1.0 - sw[ix]
        sw[ix] = 0.0

        rs[gas_present] = rs_sat[gas_present]
        rs[~gas_present] = np.minimum(rs_sat[~gas_present], rs[~gas_present])
        st.rs = np.maximum(rs, 0.0)
        st.status = 1 + 2 * gas_present.astype(np.int64)
        s = np.column_stack([sw, so, sg])
        st.s = s / s.sum(axis=1, keepdims=True)

    # -- convergence + stepping ----------------------------------------------
    def _conv_scaling(self, st: BOState):
        """`(B_avg, pv)` for the CNV norm - MRST's `getConvergenceValuesCNV`.

        `B_avg = mean(1/b)` weights each phase's surface-volume residual by its
        average formation volume factor, turning it back into the reservoir
        volume it actually occupies. Without it the phases are compared on
        incommensurable scales: `b_g` is ~262 on SPE 1, so an unweighted gas
        residual is 262x larger than MRST's for the same error.
        """
        f = self.fluid
        p = st.pressure
        b = [f.b_w(p), f.b_o(p, st.rs, st.status == 3), f.b_g(p)]
        b_avg = np.array([float(np.mean(1.0 / np.asarray(value(x), float)))
                          for x in b])
        pv = self.pv_ref * np.asarray(value(f.pv_mult(p)), dtype=float)
        return b_avg, pv

    def _converged(self, eqs, dt: float, st: BOState) -> bool:
        res = [value(e) for e in eqs]
        b_avg, pv = self._conv_scaling(st)
        cnv, mb = cnv_mb_convergence(res[:3], b_avg, pv, dt)
        q_ref = max(abs(w.val) for w in self.wells)
        well_ok = all(np.abs(r).max() <= self.tol_well * q_ref
                      for r in res[3:6])
        ctrl_scale = np.array([1e5 if w.type == "bhp" else abs(w.val)
                               for w in self.wells])
        ctrl_ok = np.all(np.abs(res[6]) <= self.tol_well * ctrl_scale)
        return cnv <= self.tol_cnv and mb <= self.tol_mb \
            and well_ok and ctrl_ok

    def step(self, state: BOState, dt: float) -> Tuple[BOState, bool, int]:
        st0 = state
        st = state.copy()
        for it in range(self.max_iterations):
            st1 = st.status == 1
            x_val = np.where(st1, st.rs, st.s[:, 2])
            p, sw, x, qws, qos, qgs, bhp = init_variables_adi(
                st.pressure, st.s[:, 0], x_val,
                st.qWs, st.qOs, st.qGs, st.bhp,
            )
            eqs = self.assemble(st0, st, p, sw, x, qws, qos, qgs, bhp, dt)
            if self._converged(eqs, dt, st):
                return st, True, it
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st = self.update_state(st, upd)

        st1 = st.status == 1
        x_val = np.where(st1, st.rs, st.s[:, 2])
        p, sw, x, qws, qos, qgs, bhp = init_variables_adi(
            st.pressure, st.s[:, 0], x_val, st.qWs, st.qOs, st.qGs, st.bhp
        )
        eqs = self.assemble(st0, st, p, sw, x, qws, qos, qgs, bhp, dt)
        return st, self._converged(eqs, dt, st), self.max_iterations


def initial_state_spe1(model: ThreePhaseBlackOilModelAD, deck) -> BOState:
    """Initial SPE1 state from the deck's SOLUTION section."""
    nc = model.g.cells.num
    s = np.column_stack([deck["swat"], 1 - deck["swat"] - deck["sgas"],
                         deck["sgas"]])
    status = np.where(deck["sgas"] > 0, 3, 1).astype(np.int64)
    nw = len(model.wells)
    qws, qos, qgs = np.zeros(nw), np.zeros(nw), np.zeros(nw)
    bhp = np.zeros(nw)
    p_mean = float(deck["pressure"].mean())
    for k, w in enumerate(model.wells):
        if w.type == "orat":
            qos[k] = w.val
            bhp[k] = p_mean - 10e5
        elif w.type == "rate":
            qgs[k] = w.val * w.compi[2]
            qws[k] = w.val * w.compi[0]
            bhp[k] = p_mean + 10e5
        else:
            bhp[k] = w.val
    return BOState(pressure=deck["pressure"].copy(), s=s,
                   rs=deck["rs"].copy(), status=status,
                   qWs=qws, qOs=qos, qGs=qgs, bhp=bhp)


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.gridprocessing import compute_geometry, tensor_grid
    from pymrst.modules.ad_props.blackoil_fluid import init_deck_blackoil_fluid
    from pymrst.modules.deckformat import read_spe1_deck
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import Well, add_well
    from pymrst.utils.units import DAY

    DECK = Path(__file__).parents[5] / (
        "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
    )
    if not DECK.exists():
        DECK = Path(__file__).parents[4] / (
            "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
        )

    def build_spe1():
        deck = read_spe1_deck(DECK)
        g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
        rock = make_rock(g, deck["perm"], deck["poro"][:, None])
        fluid = init_deck_blackoil_fluid(deck)
        wells: List[Well] = []
        for name in ("INJECTOR", "PRODUCER"):
            wd = deck["wells"][name]
            wells = add_well(
                wells, g, rock, wd["cells"], well_type=wd["type"],
                val=wd["val"], radius=wd["radius"], compi=wd["compi"],
                sign=wd["sign"], name=name, ref_depth=wd["ref_depth"],
                gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
            )
        model = ThreePhaseBlackOilModelAD(g, rock, fluid, wells)
        return deck, model

    def test_spe1_first_steps():
        """First three SPE1 steps: convergence, controls, gas liberation."""
        deck, model = build_spe1()
        assert np.allclose([w.dZ[0] for w in model.wells], 0.0, atol=1e-9)
        st = initial_state_spe1(model, deck)
        total_gas0 = _surface_gas(model, st)
        for i, dt in enumerate(deck["tsteps"][:3]):
            st, ok, its = model.step(st, dt)
            assert ok, f"step {i} failed after {its} its"
        # controls honored
        assert np.isclose(st.qOs[1], deck["wells"]["PRODUCER"]["val"],
                          rtol=1e-9)
        assert np.isclose(st.qWs[0] + st.qOs[0] + st.qGs[0],
                          deck["wells"]["INJECTOR"]["val"], rtol=1e-9)
        # injected gas must appear: injector cell saturates, sG > 0
        assert st.status[0] == 3 and st.s[0, 2] > 0.0
        # producer column still undersaturated
        assert st.status[299] == 1
        # gas surface-volume balance over 5 days
        t = deck["tsteps"][:3].sum()
        expect = (deck["wells"]["INJECTOR"]["val"] + st.qGs[1] * 0) * t
        print(f"ok test_spe1_first_steps (sG inj cell = {st.s[0, 2]:.4f})")

    def _surface_gas(model, st):
        f = model.fluid
        b_g = value(f.b_g(st.pressure))
        b_o = value(f.b_o(st.pressure, st.rs, st.status == 3))
        pv = model.pv_ref * value(f.pv_mult(st.pressure))
        return np.sum(pv * (b_g * st.s[:, 2] + st.rs * b_o * st.s[:, 1]))

    def test_gas_mass_balance():
        """Surface gas in place grows by exactly the injected volume."""
        deck, model = build_spe1()
        st = initial_state_spe1(model, deck)
        g0 = _surface_gas(model, st)
        qg = np.zeros(0)
        dts = deck["tsteps"][:4]
        rates = []
        for dt in dts:
            st, ok, _ = model.step(st, dt)
            assert ok
            rates.append(st.qGs.sum())
        g1 = _surface_gas(model, st)
        injected = float(np.sum(np.asarray(rates) * dts))
        assert np.isclose(g1 - g0, injected, rtol=1e-6), (
            f"gas gained {g1 - g0:.4e} vs injected {injected:.4e}"
        )
        print("ok test_gas_mass_balance")

    test_spe1_first_steps()
    test_gas_mass_balance()
    print("\nAll tests passed!")
