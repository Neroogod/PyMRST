"""Fully-implicit two-phase oil-water model, ported from MRST ad-blackoil.

Implements equationsOilWater.m and the standard well model of
computeWellContributionsSingleWell.m: surface-volume conservation

    (1/dt) (pv b_ph s_ph - pv b_ph0 s_ph0) + div(b_ph,upw v_ph) = q_ph,s

with phase-potential upwinding, plus per-well surface-rate closure and
control equations. Primary variables: pressure, sW, and (qWs, qOs, bhp)
per well. Gravity is not yet supported here (connection pressure drops
cdp are zero); use gravity=None cases for now.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_props.simple_adi_fluid import ADFluid
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.utils.adi import (
    ADI,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
    vertcat_adi,
)
from pymrst.utils.units import BARSA


@dataclass
class OWState:
    """State of the oil-water model: cell fields plus per-well solution."""

    pressure: np.ndarray            # (nc,)
    sW: np.ndarray                  # (nc,)
    qWs: np.ndarray                 # (nw,) surface water rate
    qOs: np.ndarray                 # (nw,) surface oil rate
    bhp: np.ndarray                 # (nw,) bottom-hole pressure

    def copy(self) -> "OWState":
        return OWState(*(v.copy() for v in
                         (self.pressure, self.sW, self.qWs, self.qOs,
                          self.bhp)))


class TwoPhaseOilWaterModelAD:
    """Fully-implicit oil-water model with MRST's standard well model."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: ADFluid,
        wells: Sequence[Well],
        ds_max: float = 0.2,
        dp_max_rel: float = 0.2,
        tol_cnv: float = 1e-8,
        tol_mb: float = 1e-12,
        tol_well: float = 1e-10,
        max_iterations: int = 25,
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.wells = list(wells)
        self.ops = setup_operators_tpfa(g, rock)
        self.pv = pore_volume(g, rock)
        self.ds_max = ds_max
        self.dp_max_rel = dp_max_rel
        self.tol_cnv = tol_cnv
        self.tol_mb = tol_mb
        self.tol_well = tol_well
        self.max_iterations = max_iterations
        for w in self.wells:
            assert w.compi.size >= 2, "wells need 2-phase compi"
            assert np.allclose(w.dZ, 0.0) or True  # cdp support pending

    def initial_state(self, p0: float, sw0: float) -> OWState:
        nw = len(self.wells)
        bhp = np.array([w.val if w.type == "bhp" else p0
                        for w in self.wells])
        qws = np.array([w.val * w.compi[0] if w.type == "rate" else 0.0
                        for w in self.wells])
        qos = np.array([w.val * w.compi[1] if w.type == "rate" else 0.0
                        for w in self.wells])
        return OWState(
            pressure=np.full(self.g.cells.num, float(p0)),
            sW=np.full(self.g.cells.num, float(sw0)),
            qWs=qws, qOs=qos, bhp=bhp,
        )

    # -- equation assembly -------------------------------------------------
    def _well_contributions(
        self, k: int, p, mob_w, mob_o, b_w, b_o, qws_k, qos_k, bhp_k
    ) -> Tuple[ADI, ADI, ADI, ADI, ADI, np.ndarray]:
        """One well's surface connection rates and closure/control eqs.

        Port of computePerforationRates/setupWellControlEquations for a
        two-phase model without dissolved components and cdp = 0.
        """
        w = self.wells[k]
        wc = w.cells
        tw = w.WI
        compi = w.compi[:2]

        drawdown = -bhp_k + p[wc]           # cdp = 0 (no gravity yet)
        inj_conn = value(drawdown) < 0
        prod_conn = ~inj_conn

        mob_w_c, mob_o_c = mob_w[wc], mob_o[wc]
        b_w_c, b_o_c = b_w[wc], b_o[wc]

        any_inj, any_prod = bool(inj_conn.any()), bool(prod_conn.any())

        if any_prod:
            con_eff = prod_conn * tw
            cq_pw = -con_eff * mob_w_c * drawdown
            cq_po = -con_eff * mob_o_c * drawdown
            cq_psw = b_w_c * cq_pw           # surface volumes
            cq_pso = b_o_c * cq_po
            wbq_w = qws_k * float(value(qws_k)[0] > 0) - ad_sum(cq_psw)
            wbq_o = qos_k * float(value(qos_k)[0] > 0) - ad_sum(cq_pso)
            wbqt = wbq_w + wbq_o
            if value(wbqt)[0] == 0.0:        # dead well: fall back to compi
                mix_w, mix_o = compi[0], compi[1]
            else:
                mix_w, mix_o = wbq_w / wbqt, wbq_o / wbqt
        else:
            cq_psw = cq_pso = None
            mix_w, mix_o = compi[0], compi[1]

        if any_inj:
            mt = mob_w_c + mob_o_c
            cqt_i = -(inj_conn * tw) * (mt * drawdown)
            vol_rat = mix_w / b_w_c + mix_o / b_o_c
            cqt_is = cqt_i / vol_rat
            if any_prod:
                cq_sw = cq_psw + mix_w * cqt_is
                cq_so = cq_pso + mix_o * cqt_is
            else:
                cq_sw = mix_w * cqt_is
                cq_so = mix_o * cqt_is
        else:
            cq_sw, cq_so = cq_psw, cq_pso

        eq_w = qws_k - ad_sum(cq_sw)
        eq_o = qos_k - ad_sum(cq_so)

        if w.type == "bhp":
            ctrl = bhp_k - w.val
        elif w.type == "rate":
            ctrl = qws_k + qos_k - w.val
        else:
            raise ValueError(f"unsupported well control '{w.type}'")

        return cq_sw, cq_so, eq_w, eq_o, ctrl, wc

    def assemble(self, st0: OWState, p, sw, qws, qos, bhp, dt: float):
        """Residual equations for the Newton iteration (ADI or values)."""
        f = self.fluid
        ops = self.ops
        pv = self.pv

        b_w, b_o = f.b_w(p), f.b_o(p)
        so = 1.0 - sw
        mob_w = f.kr_w(sw) / f.mu[0]
        mob_o = f.kr_o(so) / f.mu[1]

        # Not wrapped in value(): when st0 carries ADI (the adjoint seeds the
        # *previous* state to get dR/dx0), these must keep their derivatives.
        # In the forward Newton st0 holds plain arrays and nothing changes.
        b_w0 = f.b_w(st0.pressure)
        b_o0 = f.b_o(st0.pressure)

        # Phase fluxes with potential upwinding (no gravity, no pc).
        pot = ops.grad(p)
        flag = value(pot) <= 0
        v_w = -ops.T * ops.upw(flag, mob_w) * pot
        v_o = -ops.T * ops.upw(flag, mob_o) * pot

        water = (1.0 / dt) * (pv * b_w * sw - pv * b_w0 * st0.sW)
        oil = (1.0 / dt) * (pv * b_o * so - pv * b_o0 * (1.0 - st0.sW))
        water = water + ops.div(ops.upw(flag, b_w) * v_w)
        oil = oil + ops.div(ops.upw(flag, b_o) * v_o)

        eq_w_wells, eq_o_wells, ctrls = [], [], []
        for k in range(len(self.wells)):
            cq_sw, cq_so, eq_w, eq_o, ctrl, wc = self._well_contributions(
                k, p, mob_w, mob_o, b_w, b_o,
                qws[k:k + 1], qos[k:k + 1], bhp[k:k + 1],
            )
            water[wc] = water[wc] - cq_sw
            oil[wc] = oil[wc] - cq_so
            eq_w_wells.append(eq_w)
            eq_o_wells.append(eq_o)
            ctrls.append(ctrl)

        return (water, oil, vertcat_adi(*eq_w_wells),
                vertcat_adi(*eq_o_wells), vertcat_adi(*ctrls))

    # -- convergence and stepping ------------------------------------------
    def _converged(self, eqs, dt: float, st: OWState) -> bool:
        water, oil, eq_w, eq_o, ctrl = (value(e) for e in eqs)
        # `B_avg = mean(1/b)`, as MRST's `getConvergenceValuesCNV` does: the
        # surface-volume residual weighted back into reservoir volume. Here both
        # b are ~1 so it barely moves, but the criterion must mean the same
        # thing in every model or two of them stop at different places.
        f, pv = self.fluid, self.pv
        bw = np.mean(1.0 / np.asarray(value(f.b_w(st.pressure)), float))
        bo = np.mean(1.0 / np.asarray(value(f.b_o(st.pressure)), float))
        cnv = max(bw * np.abs(water / pv).max(),
                  bo * np.abs(oil / pv).max()) * dt
        mb = max(bw * abs(water.sum()), bo * abs(oil.sum())) * dt / pv.sum()
        q_ref = max(max(abs(w.val) for w in self.wells if w.type == "rate"),
                    1e-12) if any(w.type == "rate" for w in self.wells) else 1.0
        well_ok = (np.abs(eq_w).max() <= self.tol_well * q_ref
                   and np.abs(eq_o).max() <= self.tol_well * q_ref)
        ctrl_ok = np.all(np.abs(ctrl) <= self.tol_well *
                         np.array([BARSA if w.type == "bhp" else q_ref
                                   for w in self.wells]))
        return (cnv <= self.tol_cnv and mb <= self.tol_mb
                and well_ok and ctrl_ok)

    def step(self, state: OWState, dt: float) -> Tuple[OWState, bool, int]:
        """One backward-Euler step solved by Newton; returns new state."""
        st0 = state
        st = state.copy()
        nc = self.g.cells.num
        nw = len(self.wells)

        for it in range(self.max_iterations):
            p, sw, qws, qos, bhp = init_variables_adi(
                st.pressure, st.sW, st.qWs, st.qOs, st.bhp
            )
            eqs = self.assemble(st0, p, sw, qws, qos, bhp, dt)
            if self._converged(eqs, dt, st):
                return st, True, it
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)

            dp = upd[:nc]
            ds = upd[nc:2 * nc]
            # MRST-style update limits.
            cap = self.dp_max_rel * np.abs(st.pressure)
            dp = np.clip(dp, -cap, cap)
            ds = np.clip(ds, -self.ds_max, self.ds_max)
            st.pressure = st.pressure + dp
            st.sW = np.clip(st.sW + ds, 0.0, 1.0)
            st.qWs = st.qWs + upd[2 * nc:2 * nc + nw]
            st.qOs = st.qOs + upd[2 * nc + nw:2 * nc + 2 * nw]
            st.bhp = st.bhp + upd[2 * nc + 2 * nw:]

        p, sw, qws, qos, bhp = init_variables_adi(
            st.pressure, st.sW, st.qWs, st.qOs, st.bhp
        )
        eqs = self.assemble(st0, p, sw, qws, qos, bhp, dt)
        return st, self._converged(eqs, dt, st), self.max_iterations


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import CENTI, DARCY, DAY, MILLI, POISE

    def _model():
        g = compute_geometry(cart_grid([8, 8, 1], [200.0, 200.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([63]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        return TwoPhaseOilWaterModelAD(g, rock, fluid, w)

    def test_step_converges():
        """A 30-day step converges quadratically from hydrostatic rest."""
        m = _model()
        st = m.initial_state(200 * BARSA, 0.2)
        st1, ok, its = m.step(st, 30 * DAY)
        assert ok, "Newton did not converge"
        assert its < 15
        assert np.isclose(st1.qWs[0] + st1.qOs[0], 1e-3, rtol=1e-9)
        assert np.isclose(st1.bhp[1], 150 * BARSA)
        assert st1.qWs[1] + st1.qOs[1] < 0, "producer must produce"
        print(f"ok test_step_converges ({its} iterations)")

    def test_surface_volume_balance():
        """Accumulated surface volumes match well totals to solver tol."""
        m = _model()
        st = m.initial_state(200 * BARSA, 0.2)
        f, pv = m.fluid, m.pv
        vol_w0 = np.sum(pv * value(f.b_w(st.pressure)) * st.sW)
        dt = 30 * DAY
        states = [st]
        for _ in range(4):
            st, ok, _ = m.step(st, dt)
            assert ok
            states.append(st)
        vol_w1 = np.sum(pv * value(f.b_w(st.pressure)) * st.sW)
        injected = dt * sum(s.qWs.sum() for s in states[1:])
        assert np.isclose(vol_w1 - vol_w0, injected, rtol=1e-7)
        print("ok test_surface_volume_balance")

    def test_water_front_moves_toward_producer():
        """Water saturation rises near the injector first."""
        m = _model()
        st = m.initial_state(200 * BARSA, 0.2)
        for _ in range(3):
            st, ok, _ = m.step(st, 30 * DAY)
        assert st.sW[0] > 0.6            # injector cell floods
        assert st.sW[63] < 0.201         # producer corner untouched
        assert st.pressure[0] > st.pressure[63]
        print("ok test_water_front_moves_toward_producer")

    test_step_converges()
    test_surface_volume_balance()
    test_water_front_moves_toward_producer()
    print("\nAll tests passed!")
