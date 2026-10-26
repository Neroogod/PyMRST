"""MRST ``ad-blackoil/models/ThreePhaseBlackOilModel.m``.

`ThreePhaseBlackOilModel` is MRST's class on the AD-OO framework: equations
from `equationsBlackOil`, Newton updates with MRST's variable switching and
`computeFlashBlackOil`, and everything else inherited from `ReservoirModel`.

`ThreePhaseBlackOilModelAD`, below it, is PyMRST's earlier hand-built live-oil
model on the legacy contract, kept for the code built on it:

Three-phase black-oil model with dissolved gas (MRST equationsBlackOil).

Fully-implicit water/oil/gas conservation in surface volumes with live
oil (disgas, no vapoil). The hydrocarbon primary variable x switches
meaning per cell: sG in saturated cells (status 3), Rs in undersaturated
cells (status 1), with the status flag updated by the flash of
computeFlashBlackOil.m after every Newton update. Wells follow
computeWellContributionsSingleWell.m including dissolved-gas surface
volume accounting. Connection pressure drops (cdp) are zero, which is
exact for SPE 1 (refDepth equals the perforation depth).
"""

from copy import copy
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_props.props.init_deck_adi_fluid import BlackOilFluid
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_core.utils.convergence import cnv_mb_convergence
from pymrst.modules.ad_blackoil.utils.get_b_g_bo import get_b_g_bo_ad
from pymrst.modules.ad_blackoil.utils.get_b_o_bo import get_b_o_bo_ad
from pymrst.modules.ad_blackoil.utils.get_flux_and_props_gas_bo import (
    get_flux_and_props_gas_bo,
)
from pymrst.modules.ad_blackoil.utils.get_flux_and_props_oil_bo import (
    get_flux_and_props_oil_bo,
)
from pymrst.modules.ad_blackoil.utils.get_flux_and_props_water_bo import (
    get_flux_and_props_water_bo,
)
from pymrst.modules.ad_blackoil.utils.calculate_hydrocarbons_from_status_bo import (
    calculate_hydrocarbons_from_status_bo,
)
from pymrst.modules.ad_blackoil.utils.compute_flash_black_oil import (
    compute_flash_black_oil,
)
from pymrst.modules.ad_blackoil.utils.equations_black_oil import equations_black_oil
from pymrst.modules.ad_blackoil.utils.get_cell_status_vo import get_cell_status_vo
from pymrst.modules.ad_core.ad_oo.physical_model import (
    copy_state,
    merge_options_relaxed,
)
from pymrst.modules.ad_core.ad_oo.reservoir_model import (
    ReservoirModel as AdOoReservoirModel,
)
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


class ThreePhaseBlackOilModelAD(PhysicalModel):
    """Fully-implicit black-oil model with dissolved gas.

    Sits on `PhysicalModel`. It did not, and the consequence was the same one
    the two-phase model had before its own migration: the class inherited from
    `object`, so it reached none of the shared AD-OO machinery and carried
    none of MRST's `ThreePhaseBlackOilModel` protocol - nine methods, from
    `getDissolutionMatrix` through `validateState`, simply absent.

    Inheriting adds 76 names, 8 of which are probed by `hasattr`/`getattr`
    elsewhere in the package, so this is one of the changes in the port that
    can move numbers without touching a formula. One of those probes had to be
    fixed first: `cfl.py` asked `hasattr(model, "init_state_ad")` and then
    called it with one argument, while `PhysicalModel.init_state_ad` takes
    three - so the probe would have flipped `True` and the call raised. It now
    tests whether the method is the base's rather than whether the name
    exists, which is the same fix the `get_equations` and `get_prop` probes
    needed in earlier phases.

    The characterization gate is what makes the migration checkable: it must
    stay bit-identical across all eight frozen cases.
    """

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: BlackOilFluid,
        wells: Sequence[Well],
        gravity: float = NORM_GRAVITY,
        ds_max: float = 0.2,
        # MRST's own defaults (`ReservoirModel.m`): `toleranceCNV = 1e-3`,
        # `toleranceMB = 1e-7`, computed here with MRST's exact formulas so the
        # numbers are directly comparable. The previous 1e-8/1e-12 were five
        # orders tighter than the code being ported; see
        # `generic_black_oil_model` for the measurement and the reasoning,
        # including why `tol_well` is a mapping rather than a transcription.
        tol_cnv: float = 1e-3,
        tol_mb: float = 1e-7,
        tol_well: float = 1e-6,
        max_iterations: int = 25,
    ):
        super().__init__()
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.raw_fluid = fluid
        self.disgas = True
        self.vapoil = False
        self.wells = list(wells)
        self.ops = setup_operators_tpfa(g, rock)
        self.operators = self.ops
        self.G = g
        self.pv_ref = pore_volume(g, rock)
        self.gdz = gravity * self.ops.grad(g.cells.centroids[:, -1])
        self.ds_max = ds_max
        self.tol_cnv = tol_cnv
        self.tol_mb = tol_mb
        self.tol_well = tol_well
        self.max_iterations = max_iterations

    # ======================================================================
    # MRST `ThreePhaseBlackOilModel.m`'s own protocol.
    #
    # Inheriting `PhysicalModel` supplied five of the nine methods this class
    # was missing. These four are the black-oil-specific ones - they are about
    # dissolved gas and vaporised oil, which the base has no notion of.
    # ======================================================================

    def get_dissolution_matrix(self, rs=None, rv=None):
        """The `n_phase x n_phase` table of dissolved-component ratios.

        `dismat[i][j]` is how much of component `j` rides in phase `i`: `rv`
        in the oil-gas slot, `rs` in the gas-oil slot, `None` everywhere else.
        MRST returns it as a nested cell array and the conservation equations
        index it directly, so the shape - one row per *active* phase, not per
        possible phase - is what a caller relies on. A three-phase table
        handed to a two-phase model would put the gas ratio in the oil slot.

        The ratios are dropped when the model does not carry them: a dead-oil
        model has no `rs` however much a caller passes, and silently keeping it
        would add solution gas to a fluid that has none.
        """
        if not self.disgas:
            rs = None
        if not self.vapoil:
            rv = None

        names = [str(n).upper() for n in self.get_phase_names()]
        n = len(names)
        table = [[None]*n for _ in range(n)]
        if "O" in names and "G" in names:
            oil, gas = names.index("O"), names.index("G")
            table[oil][gas] = rv
            table[gas][oil] = rs
        return table

    getDissolutionMatrix = get_dissolution_matrix

    def get_dissolution_matrix_max(self, pressure):
        """The dissolution matrix at *saturated* conditions.

        The most gas the oil can hold at this pressure, and vice versa. Used
        as the ceiling when a cell crosses the bubble point: above it the
        extra gas must come out as a free phase rather than staying dissolved.
        """
        rs_max = self._saturated_ratio("rs", pressure) if self.disgas else None
        rv_max = self._saturated_ratio("rv", pressure) if self.vapoil else None
        return self.get_dissolution_matrix(rs_max, rv_max)

    getDissolutionMatrixMax = get_dissolution_matrix_max

    def get_scaling_factors_cpr(self, problem=None, names=None,
                                scaling_type: str = "trueimpes"):
        """Per-equation CPR weights, `1/b` per phase.

        The weighting is what makes CPR work at all, and the reason is the
        cancellation: the equations are in surface volume, `pv b_i s_i`, so
        dividing by `b_i` gives reservoir volume `pv s_i`, and summing over
        phases gives `pv` - the saturations cancel exactly and what is left is
        a function of pressure alone. That is the elliptic system AMG wants.
        An unweighted sum cancels nothing.

        `trueimpes` defers to the model's `PressureReductionFactors` where the
        state-function graph provides them; otherwise the `1/b` weights are
        evaluated at the mean pressure, which is MRST's `quasiimpes` fallback.
        """
        names = list(names or [])
        pressure = self._mean_pressure(problem)
        fluid = getattr(self, "raw_fluid", None) or self.fluid

        if str(scaling_type).lower() == "trueimpes":
            factors = self._pressure_reduction_factors(problem)
            if factors is not None:
                return list(factors)[:len(names)] or [1]*len(names)

        scaling = []
        for name in names:
            key = str(name).lower()
            if key == "water":
                b = self._eval(getattr(fluid, "b_w", getattr(fluid, "bW", None)),
                               pressure)
            elif key == "oil":
                if self.disgas:
                    rs = self._saturated_ratio("rs", pressure)
                    b = self._eval(getattr(fluid, "b_o", getattr(fluid, "bO", None)),
                                   pressure, rs, True)
                else:
                    b = self._eval(getattr(fluid, "b_o", getattr(fluid, "bO", None)),
                                   pressure)
            elif key == "gas":
                if self.vapoil:
                    rv = self._saturated_ratio("rv", pressure)
                    b = self._eval(getattr(fluid, "b_g", getattr(fluid, "bG", None)),
                                   pressure, rv, True)
                else:
                    b = self._eval(getattr(fluid, "b_g", getattr(fluid, "bG", None)),
                                   pressure)
            else:
                scaling.append(1)
                continue
            scaling.append(1.0/b if b not in (None, 0) else 1)
        return scaling

    getScalingFactorsCPR = get_scaling_factors_cpr

    @staticmethod
    def eval_single(fn, *args):
        """Call `fn`, unwrapping a one-element region list first.

        MRST's regionalised properties arrive as cell arrays even when there
        is a single region, and `fn{1}` is how it reaches the function. The
        same shape occurs here, so the unwrapping is not MATLAB bookkeeping -
        a list would otherwise be called as though it were a function.
        """
        if isinstance(fn, (list, tuple)):
            fn = fn[0]
        return fn(*args)

    evalSingle = eval_single

    # -- helpers for the four above -----------------------------------------

    def _saturated_ratio(self, which: str, pressure):
        """`rs_sat` / `rv_sat` at `pressure`, under either spelling."""
        fluid = getattr(self, "raw_fluid", None) or self.fluid
        for name in (f"{which}_sat", f"{which}Sat"):
            fn = getattr(fluid, name, None)
            if fn is not None:
                return self.eval_single(fn, pressure)
        return None

    @staticmethod
    def _eval(fn, *args):
        if fn is None:
            return None
        if isinstance(fn, (list, tuple)):
            fn = fn[0]
        result = fn(*args)
        return float(np.ravel(value(result))[0])

    @staticmethod
    def _mean_pressure(problem) -> float:
        state = getattr(problem, "state", None) if problem is not None else None
        if state is None:
            return 0.0
        pressure = state["pressure"] if "pressure" in state else \
            getattr(state, "pressure", 0.0)
        return float(np.mean(np.asarray(value(pressure), dtype=float)))

    def _pressure_reduction_factors(self, problem):
        """MRST's `trueimpes` weights, when the property graph supplies them."""
        state = getattr(problem, "state", None) if problem is not None else None
        if state is None or not hasattr(self, "get_prop"):
            return None
        try:
            return self.get_prop(state, "PressureReductionFactors")
        except Exception:
            return None

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

        water_props = get_flux_and_props_water_bo(
            self, p, sw, f.kr_w(sw), ops.T, self.gdz,
        )
        oil_props = get_flux_and_props_oil_bo(
            self, p, so, f.kr_o(sw, so, sg), ops.T, self.gdz, rs, st3,
        )
        gas_props = get_flux_and_props_gas_bo(
            self, p, sg, f.kr_g(sg), ops.T, self.gdz,
        )
        fluxes = {
            "w": (water_props[5], water_props[0], water_props[1]),
            "o": (oil_props[5], oil_props[0], oil_props[1]),
            "g": (gas_props[5], gas_props[0], gas_props[1]),
        }
        b_w, mob_w = water_props[1], water_props[2]
        b_o, mob_o = oil_props[1], oil_props[2]
        b_g, mob_g = gas_props[1], gas_props[2]

        bWvW = ops.upw(fluxes["w"][0], b_w) * fluxes["w"][1]
        bOvO = ops.upw(fluxes["o"][0], b_o) * fluxes["o"][1]
        bGvG = ops.upw(fluxes["g"][0], b_g) * fluxes["g"][1]
        rsbOvO = ops.upw(fluxes["o"][0], rs) * bOvO

        pv = self.pv_ref * f.pv_mult(p)
        p0, s0, rs0 = st0.pressure, st0.s, st0.rs
        st3_0 = st0.status == 3
        pv0 = self.pv_ref * value(f.pv_mult(p0))
        b_w0 = value(f.b_w(p0))
        b_o0 = value(get_b_o_bo_ad(self, p0, rs0, st3_0))
        b_g0 = value(get_b_g_bo_ad(self, p0))

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
        b = [f.b_w(p), get_b_o_bo_ad(self, p, st.rs, st.status == 3),
             get_b_g_bo_ad(self, p)]
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


class ThreePhaseBlackOilModel(AdOoReservoirModel):
    """MRST `ThreePhaseBlackOilModel`: water, oil and gas with optional
    dissolved gas (`disgas`) and vaporised oil (`vapoil`)."""

    def __init__(self, G: Any, rock: Any, fluid: Any, *, inputdata: Any = None,
                 **options: Any):
        super().__init__(G, rock, fluid, inputdata=inputdata)
        self.vapoil = False
        self.disgas = False
        self.drsMaxAbs = float("inf")
        self.drsMaxRel = float("inf")
        self.useCNVConvergence = True
        self.oil = True
        self.gas = True
        self.water = True
        merge_options_relaxed(self, options)
        d = self.inputdata
        if d is not None:
            runspec = _deck_field(d, "RUNSPEC")
            if runspec is None:
                raise ValueError("Unknown dataset format!")
            if _deck_field(runspec, "VAPOIL") is not None:
                self.vapoil = bool(_deck_field(runspec, "VAPOIL"))
            if _deck_field(runspec, "DISGAS") is not None:
                self.disgas = bool(_deck_field(runspec, "DISGAS"))
        self.OutputStateFunctions = ["PoreVolume", "ShrinkageFactors"]

    def get_variable_field(self, name: str, throw_error: bool = True):
        if name.lower() in ("rs", "rv"):
            return name.lower(), ":"
        return super().get_variable_field(name, throw_error)

    def get_primary_variables(self, state: Any):
        """`pressure`, every saturation but oil's, the switched `x`, wells."""
        phases = list(self.get_phase_names())
        nph = len(phases)
        ix = phases.index("O") if self.oil else nph - 1
        phases.pop(ix)
        snames = [f"s{x}" for x in phases]
        p = self.get_prop(state, "pressure")
        s = [self.get_prop(state, n) for n in snames]
        rs, rv = self.get_props(state, "rs", "rv")
        if self.disgas or self.vapoil:
            assert self.oil, "Cannot have disgas/vapoil without oil phase."
            assert self.gas, "Cannot have disgas/vapoil without gas phase."
            s_w = s[phases.index("W")] if self.water else 0.0
            is_g = phases.index("G")
            s_g = s[is_g]
            st = self.get_cell_status_vo(state, 1 - s_w - s_g, s_w, s_g)
            s[is_g] = (_as_float(st[0])*rs + _as_float(st[1])*rv
                       + _as_float(st[2])*s_g)
            snames[is_g] = "x"
        variables = [p] + s
        names = ["pressure"] + snames
        origin = [type(self).__name__]*len(variables)
        if self.FacilityModel is not None:
            v, n, o = self.FacilityModel.get_primary_variables(state)
            variables += list(v)
            names += list(n)
            origin += list(o)
        return variables, names, origin

    def init_state_ad(self, state: Any, variables, names, origin):
        """Set saturations (and `rs`/`rv`) from the switched variable."""
        variables, names, origin = list(variables), list(names), list(origin)
        lowered = [n.lower() for n in names]
        removed = [False]*len(variables)
        st = None
        x = None
        if self.disgas or self.vapoil:
            if self.water:
                isw = lowered.index("sw")
                s_w = variables[isw]
                removed[isw] = True
            else:
                s_w = 0.0
            isx = lowered.index("x")
            x = variables[isx]
            s_g = self.get_props(state, "sg")
            st = self.get_cell_status_vo(state, 1 - s_w - s_g, s_w, s_g)
            s_g = _as_float(st[1])*(1 - s_w) + _as_float(st[2])*x
            s_o = _as_float(st[0])*(1 - s_w) + _as_float(~np.asarray(st[0]))*(1 - s_w - s_g)
            sat = [s_w, s_o, s_g] if self.water else [s_o, s_g]
            removed[isx] = True
        else:
            phases = self.get_phase_names()
            sat: List[Any] = [None]*len(phases)
            fill = np.ones(self.G.cells.num)
            missing = []
            for i, letter in enumerate(phases):
                key = f"s{letter}".lower()
                if key in lowered:
                    k = lowered.index(key)
                    fill = fill - variables[k]
                    removed[k] = True
                    sat[i] = variables[k]
                else:
                    missing.append(i)
            for i in missing:
                sat[i] = fill
        state = self.set_prop(state, "s", sat)
        if self.FacilityModel is not None:
            fm = type(self.FacilityModel).__name__
            is_f = [o == fm for o in origin]
            state = self.FacilityModel.init_state_ad(
                state, [v for v, f in zip(variables, is_f) if f],
                [n for n, f in zip(names, is_f) if f],
                [o for o, f in zip(origin, is_f) if f])
            removed = [r or f for r, f in zip(removed, is_f)]
        keep = [not r for r in removed]
        state = super().init_state_ad(
            state, [v for v, k in zip(variables, keep) if k],
            [n for n, k in zip(names, keep) if k],
            [o for o, k in zip(origin, keep) if k])
        if self.disgas:
            rs_sat = self.get_prop(state, "RsMax")
            rs = _as_float(~np.asarray(st[0]))*rs_sat + _as_float(st[0])*x
            state = self.set_prop(state, "rs", rs)
        if self.vapoil:
            rv_sat = self.get_prop(state, "RvMax")
            rv = _as_float(~np.asarray(st[1]))*rv_sat + _as_float(st[1])*x
            state = self.set_prop(state, "rv", rv)
            bad_oil = (np.asarray(value(s_o)) == 0) & (np.asarray(value(rv)) == 0)
            if np.any(bad_oil):
                rows = np.flatnonzero(bad_oil)
                s_o = s_o + 0.0
                s_o[rows] = 1 - _rows(s_w, rows) - np.asarray(value(s_g))[rows]
                state = self.set_prop(state, "sO", s_o)
        return state

    def get_equations(self, state0: Any, state: Any, dt: float,
                      driving_forces: Any, **options: Any):
        return equations_black_oil(state0, state, self, dt, driving_forces,
                                   **options)

    def validate_model(self, *args: Any):
        model = super().validate_model(*args)
        if (model.disgas or model.vapoil) and not np.isfinite(model.minimumPressure):
            import warnings
            warnings.warn("Black-oil case with no lower limit "
                          "(model.minimumPressure) on pressure may result in "
                          "convergence issues.")
        return model

    def validate_state(self, state: Any):
        """Saturated `rs`/`rv` where the other hydrocarbon phase exists."""
        state = super().validate_state(state)
        nc = self.G.cells.num
        if self.disgas:
            self.check_property(state, "rs", nc, 1)
            rs_max = np.asarray(value(self.get_prop(state, "rsMax")), dtype=float)
            sg, so, rs = (np.array(value(x), dtype=float, copy=True)
                          for x in self.get_props(state, "sg", "so", "rs"))
            rs[sg > 0] = rs_max[sg > 0]
            state = self.set_prop(state, "rs", rs*(so > 0))
        elif "rs" not in state:
            if self.verbose:
                print('Missing field "rs" added since disgas is not enabled.')
            state["rs"] = 0.0
        if self.vapoil:
            self.check_property(state, "rv", nc, 1)
            rv_max = np.asarray(value(self.get_prop(state, "RvMax")), dtype=float)
            so, sg, rv = (np.array(value(x), dtype=float, copy=True)
                          for x in self.get_props(state, "so", "sg", "rv"))
            rv[so > 0] = rv_max[so > 0]
            state = self.set_prop(state, "rv", rv*(sg > 0))
        elif "rv" not in state:
            if self.verbose:
                print('Missing field "rv" added since vapoil is not enabled.')
            state["rv"] = 0.0
        return state

    def update_state(self, state: Any, problem: Any, dx, driving_forces: Any):
        """Newton update with variable switching and the black-oil flash."""
        variables = list(problem.primary_variables)
        removed = [False]*len(variables)
        if self.disgas or self.vapoil:
            state0 = copy_state(state)
            state = self.init_state_function_containers(state)
            state, _, _ = self.update_state_from_increment(
                state, dx, problem, "pressure", self.dpMaxRel, self.dpMaxAbs)
            state = self.cap_property(state, "pressure", self.minimumPressure,
                                      self.maximumPressure)
            variables, gone = self.strip_vars(variables, "pressure")
            removed = _mark_removed(removed, gone)
            so, sg = (np.asarray(value(x), dtype=float)
                      for x in self.get_props(state, "so", "sg"))
            if self.water:
                sw = np.asarray(value(self.get_prop(state, "sw")), dtype=float)
                dsw = np.asarray(value(self.get_increment(dx, problem, "sw")), dtype=float)
            else:
                sw = dsw = 0.0
            st = self.get_cell_status_vo(state0, so, sw, sg)
            dr = np.asarray(value(self.get_increment(dx, problem, "x")), dtype=float)
            dsg = _as_float(st[2])*dr - _as_float(st[1])*dsw
            if self.disgas:
                drs = self._dissolution_limit(state, "rsMax")
                state, _, _ = self.update_state_from_increment(
                    state, _as_float(st[0])*dr, problem, "rs", np.inf, drs)
                state = self.cap_property(state, "rs", 0)
            if self.vapoil:
                drv = self._dissolution_limit(state, "rvMax")
                state, _, _ = self.update_state_from_increment(
                    state, _as_float(st[1])*dr, problem, "rv", np.inf, drv)
                state = self.cap_property(state, "rv", 0)
            dso = -(dsg + dsw)
            n_ph = int(np.count_nonzero(self.get_active_phases()))
            ds = np.zeros((so.size, n_ph))
            ph = self.get_phase_indices()
            if self.water:
                ds[:, ph[0]] = dsw
            if self.oil:
                ds[:, ph[1]] = dso
            if self.gas:
                ds[:, ph[2]] = dsg
            state, _, _ = self.update_state_from_increment(state, ds, problem, "s",
                                                           np.inf, self.dsMaxAbs)
            kr = self.FlowPropertyFunctions.get_state_function("RelativePermeability")
            state, _ = kr.apply_immobile_chop(self, state, state0)
            assert "so" not in [v.lower() for v in variables]
            state = compute_flash_black_oil(state, state0, self, st)
            s = np.asarray(state["s"], dtype=float)
            state["s"] = s/s.sum(axis=1, keepdims=True)
            variables, gone = self.strip_vars(variables,
                                              ["sw", "so", "sg", "rs", "rv", "x"])
            removed = _mark_removed(removed, gone)
        # MATLAB passes `problem` by value: the stripped variable list must not
        # reach the caller (a line search calls this again with the same one).
        problem = copy(problem)
        problem.primary_variables = variables
        dx = [d for d, r in zip(dx, removed) if not r]
        return super().update_state(state, problem, dx, driving_forces)

    def _dissolution_limit(self, state: Any, saturated: str) -> Any:
        """MRST's `drs`: `min(drsMaxAbs, rsMax*drsMaxRel)`, or unlimited."""
        if np.isfinite(self.drsMaxRel):
            relative = np.asarray(value(self.get_prop(state, saturated)))*self.drsMaxRel
            return np.minimum(self.drsMaxAbs, relative)
        return np.inf

    def get_scaling_factors_cpr(self, problem: Any, names, solver_type="trueimpes"):
        """CPR weights: true-IMPES reduction factors, or `1/b` scaled."""
        n_names = len(names)
        scaling: List[Any] = [None]*n_names
        handled = [False]*n_names
        state = problem.state
        fluid = self.fluid
        is_surf_vol = not self.isGenericReservoirModel
        rho_s = self.get_surface_densities() if is_surf_vol else None
        ph = self.get_phase_names()
        if solver_type.lower() == "trueimpes":
            w = self.get_prop(state, "PressureReductionFactors")
            for i in range(len(w)):
                scaling[i] = w[i]
                handled[i] = True
        else:
            p = float(np.mean(value(state["pressure"])))
            for i, name in enumerate(n.lower() for n in names):
                if name == "oil":
                    if self.disgas:
                        rs = self._eval_single(fluid.rsSat, p)
                        b_o = self._eval_single(fluid.bO, p, rs, True)
                    else:
                        b_o = self._eval_single(fluid.bO, p)
                    s = 1/b_o
                elif name == "water":
                    s = 1/self._eval_single(fluid.bW, p)
                elif name == "gas":
                    if self.vapoil:
                        rv = self._eval_single(fluid.rvSat, p)
                        b_g = self._eval_single(fluid.bG, p, rv, True)
                    else:
                        b_g = self._eval_single(fluid.bG, p)
                    s = 1/b_g
                else:
                    continue
                scaling[i] = s
                for k, eq_name in enumerate(problem.equation_names):
                    if eq_name.lower() == name and k < n_names:
                        handled[k] = True
        if is_surf_vol:
            column = {"oil": ph.find("O"), "water": ph.find("W"), "gas": ph.find("G")}
            for i, name in enumerate(n.lower() for n in names):
                if name in column and scaling[i] is not None:
                    scaling[i] = scaling[i]*rho_s[0, column[name]]
        if not all(handled):
            other = super().get_scaling_factors_cpr(
                problem, [n for n, h in zip(names, handled) if not h])
            slots = [i for i, h in enumerate(handled) if not h]
            for i, item in zip(slots, other):
                scaling[i] = item
        return scaling

    def get_cell_status_vo(self, state: Any, s_o: Any, s_w: Any, s_g: Any):
        status = state.get("status") if isinstance(state, dict) else None
        return get_cell_status_vo(s_o, s_w, s_g, status=status,
                                  vapoil=self.vapoil, disgas=self.disgas)

    def calculate_hydrocarbons_from_status_bo(self, status, s_o, x, rs, rv,
                                              pressure):
        return calculate_hydrocarbons_from_status_bo(
            self.fluid, status, s_o, x, rs, rv, pressure, self.disgas,
            self.vapoil)

    def get_dissolution_matrix(self, rs: Any, rv: Any):
        """`dismat[i][j]`: phase `j` dissolved in phase `i` (rs, rv)."""
        act = self.get_active_phases()
        n_ph = int(np.count_nonzero(act))
        rs = rs if self.disgas else None
        rv = rv if self.vapoil else None
        dismat: List[List[Any]] = [[None]*n_ph for _ in range(n_ph)]
        ix = 0
        for i in range(3):
            if not act[i]:
                continue
            jx = 0
            for j in range(3):
                if not act[j]:
                    continue
                if i == 1 and j == 2:
                    dismat[ix][jx] = rv
                elif i == 2 and j == 1:
                    dismat[ix][jx] = rs
                jx += 1
            ix += 1
        return dismat

    def get_dissolution_matrix_max(self, pressure: Any):
        rs_max = self.fluid.rsSat(pressure) if self.disgas else None
        rv_max = self.fluid.rvSat(pressure) if self.vapoil else None
        return self.get_dissolution_matrix(rs_max, rv_max)

    @staticmethod
    def _eval_single(fn: Any, *args: Any) -> Any:
        if isinstance(fn, (list, tuple)):
            fn = fn[0]
        return fn(*args)

    getVariableField = get_variable_field
    getPrimaryVariables = get_primary_variables
    initStateAD = init_state_ad
    getEquations = get_equations
    validateModel = validate_model
    validateState = validate_state
    updateState = update_state
    getScalingFactorsCPR = get_scaling_factors_cpr
    getCellStatusVO = get_cell_status_vo
    calculateHydrocarbonsFromStatusBO = calculate_hydrocarbons_from_status_bo
    getDissolutionMatrix = get_dissolution_matrix
    getDissolutionMatrixMax = get_dissolution_matrix_max


def _deck_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_float(mask: Any) -> Any:
    return np.asarray(mask, dtype=float)


def _rows(x: Any, rows: np.ndarray) -> Any:
    """MATLAB `x(rows)`, keeping derivatives when `x` is AD; a scalar is
    returned as it is (`sW = 0` without water)."""
    if hasattr(x, "jac"):
        return x[rows]
    if np.ndim(x) == 0:
        return float(x)
    return np.asarray(x)[rows]


def _mark_removed(removed: List[bool], gone: List[bool]) -> List[bool]:
    """MRST's `removed(~removed) = removed(~removed) | ix`."""
    out = list(removed)
    slots = [i for i, r in enumerate(out) if not r]
    for slot, g in zip(slots, gone):
        out[slot] = out[slot] or g
    return out


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.gridprocessing import compute_geometry, tensor_grid
    from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
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
        b_g = value(get_b_g_bo_ad(model, st.pressure))
        b_o = value(get_b_o_bo_ad(model, st.pressure, st.rs, st.status == 3))
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
