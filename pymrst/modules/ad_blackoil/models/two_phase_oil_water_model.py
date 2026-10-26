"""MRST ``ad-blackoil/models/TwoPhaseOilWaterModel.m``.

`TwoPhaseOilWaterModel` is MRST's class on the AD-OO framework: a
`ThreePhaseBlackOilModel` with water and oil only, its equations from
`equationsOilWater`.

`TwoPhaseOilWaterModelAD`, below it, is PyMRST's earlier hand-built model on
the legacy contract, kept for the code built on it:

Fully-implicit two-phase oil-water model, ported from MRST ad-blackoil.

Implements equationsOilWater.m and the standard well model of
computeWellContributionsSingleWell.m: surface-volume conservation

    (1/dt) (pv b_ph s_ph - pv b_ph0 s_ph0) + div(b_ph,upw v_ph) = q_ph,s

with phase-potential upwinding, plus per-well surface-rate closure and
control equations. Primary variables: pressure, sW, and (qWs, qOs, bhp)
per well. Gravity is optional (`gravity=0.0` by default, matching every
existing caller): pass a nonzero magnitude to get MRST's per-phase
gravity-adjusted potentials in the flux and a per-perforation wellbore
hydrostatic `cdp` in the well equations, computed once per timestep by
`_compute_cdp_per_well` via the same topology walk as MRST's
`SimpleWell.updateConnectionPressureDrop` - see that method's docstring
for how the previous timestep's converged well composition is
reconstructed without adding new per-perforation state to `OWState`.
"""

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Sequence, Tuple

from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    ThreePhaseBlackOilModel,
)
from pymrst.modules.ad_blackoil.utils.equations_oil_water import equations_oil_water
from pymrst.modules.ad_core.ad_oo.physical_model import merge_options

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_props.init_simple_adi_fluid import ADFluid
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_core.utils.convergence import cnv_mb_convergence
from pymrst.modules.ad_blackoil.utils.get_b_o_bo import get_b_o_bo_ad
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.utils.adi import (
    ADI,
    ad_matmul,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
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


class TwoPhaseOilWaterModelAD(PhysicalModel):
    """Fully-implicit oil-water model with MRST's standard well model.

    Sits on `PhysicalModel` (Phase 3 of the ad_core port). It did not, and that
    was the structural finding of the original audit: the model most of this
    port's two-phase work runs on - the adjoint gradient, every poromechanics
    model - stood outside the AD-OO hierarchy and so could reach none of the
    shared machinery.

    The consequences were not cosmetic. `CPRSolver` is written against the
    ad_core protocol and checks `hasattr(model, "G")`; this model stored the
    grid as `self.g`, so CPR *silently fell back to a direct solve* - measured
    at 91.5% of the runtime of a Newton step, scaling as n^1.71 where a
    preconditioned solve is roughly linear.

    Inheriting adds 77 names, 8 of which are probed by `hasattr`/`getattr`
    elsewhere, so this is the one change in the port that can move numbers.
    The characterization gate is what makes it checkable: it must stay
    bit-identical across all eight frozen cases.
    """

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: ADFluid,
        wells: Sequence[Well],
        gravity: float = 0.0,
        ds_max: float = 0.2,
        dp_max_rel: float = 0.2,
        # MRST's own defaults (`ReservoirModel.m`): `toleranceCNV = 1e-3`,
        # `toleranceMB = 1e-7`. PyMRST computes both with MRST's exact formulas
        # - `B_avg*dt*max|R|/pv` and `dt*|B_avg*sum(R)|/sum(pv)` - so the
        # numbers are directly comparable, and the previous 1e-8/1e-12 were
        # five orders tighter than the code being ported.
        #
        # Tighter is not safer here. CNV is a *normalized local mass error*:
        # 1e-3 means the worst cell is off by 0.1% of its pore volume over the
        # step. Asking for 1e-8 demands accuracy the TPFA/upstream
        # discretization does not have. Measured on SPE1 over ten steps, the
        # old defaults cost 49 Newton iterations against 36 and moved the
        # answer by 1.6e-6 in pressure. On a hard case the residual stalls at
        # the achievable floor, a converged step is reported as failed, and the
        # solver cuts until the ministeps degenerate - which is exactly how the
        # Norne field model failed.
        #
        # `tol_well` is a *considered* mapping rather than a transcription.
        # MRST compares the **unscaled** well residual against
        # `toleranceWellRate = 1/day`; PyMRST divides by a reference rate
        # first, so its tolerance is relative. At field-scale rates MRST's
        # 1/day is of order 1e-6 relative, and 1e-6 recovers all but one of
        # the iterations that 1e-3 does.
        tol_cnv: float = 1e-3,
        tol_mb: float = 1e-7,
        tol_well: float = 1e-6,
        max_iterations: int = 25,
        linear_solver=None,
    ):
        super().__init__()
        # None means the sparse LU this model has always used. Opting in is a
        # decision with a measured trade-off in `_solve_increment`; defaulting
        # to CPR would make every existing caller's answers move by the linear
        # solver's tolerance without anyone asking for it.
        self.linear_solver = linear_solver
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.raw_fluid = fluid
        self.disgas = False
        self.vapoil = False
        self.wells = list(wells)
        self.ops = setup_operators_tpfa(g, rock)
        self.pv = pore_volume(g, rock)
        # MRST's `gdz`: `GenericBlackOilModelAD.__init__` here does the same
        # `gravity*(z[N[:,1]] - z[N[:,0]])` (`ReservoirModel.getGravityGradient`
        # restricted to a purely-vertical gravity vector, which is what every
        # PyMRST caller of this constructor passes). Zero gravity leaves this
        # identically zero, so every existing caller's equations are unchanged
        # bit-for-bit - this is additive, not a redefinition of the no-gravity
        # path.
        self.gravity = gravity
        if gravity:
            z = g.cells.centroids[:, -1]
            self.gdz = gravity * (z[self.ops.N[:, 1]] - z[self.ops.N[:, 0]])
        else:
            self.gdz = np.zeros(self.ops.T.size)
        self.ds_max = ds_max
        self.dp_max_rel = dp_max_rel
        self.tol_cnv = tol_cnv
        self.tol_mb = tol_mb
        self.tol_well = tol_well
        self.max_iterations = max_iterations
        for w in self.wells:
            assert w.compi.size >= 2, "wells need 2-phase compi"

    #: Water and oil. Read by `CPRSolver._weighting` to size the block rows.
    n_phases = 2

    # -- the ad_core attribute contract ----------------------------------------
    #
    # `G` is the spelling every ad_core consumer uses; this model was written
    # with `g` and so was invisible to all of them. It is a property rather
    # than a second assignment in `__init__` on purpose: two attributes holding
    # one grid can be made to disagree by a later `model.g = ...`, and a model
    # whose `G` and `g` are different grids fails in a way nobody would think
    # to look for. Binding both names to one slot makes that unrepresentable.
    @property
    def G(self) -> Grid:
        return self.g

    @G.setter
    def G(self, grid: Grid) -> None:
        self.g = grid

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

    def _segment_mix_fractions(self, w: Well, cq_w: np.ndarray, cq_o: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """MRST's `wb2in`/`mixs`: per-perforation net rates -> per-segment
        cumulative wellbore composition, for the simple top-to-bottom
        perforation chain `add_well`/`vertical_well` always build (MRST's
        own fallback when a well carries no explicit `topo`).

        `C\\qs` for that chain's `C` (identity plus one `-1` per
        connection) is exactly a reverse cumulative sum: the flow past
        segment i must carry everything perforated at or below it. Dead
        segments (net zero) fall back to `compi`, matching MRST's own
        zero-division guard.
        """
        seg_w = np.cumsum(cq_w[::-1])[::-1]
        seg_o = np.cumsum(cq_o[::-1])[::-1]
        seg_t = seg_w + seg_o
        dead = seg_t == 0.0
        safe_t = np.where(dead, 1.0, seg_t)
        mix_w = np.where(dead, w.compi[0], seg_w / safe_t)
        mix_o = np.where(dead, w.compi[1], seg_o / safe_t)
        return mix_w, mix_o

    def _topology_cdp(self, w: Well, mix_w: np.ndarray, mix_o: np.ndarray,
                      b_w_c: np.ndarray, b_o_c: np.ndarray) -> np.ndarray:
        """MRST's `SimpleWell.updateConnectionPressureDrop`: per-perforation
        mixture density from surface composition (`compVolRat`'s
        dissolution-free case, `sum(mix/b)`, converted to reservoir
        density), then `g*dz*rhoMix` cumulated down the chain from the
        reference depth. One density per perforation, not one averaged
        value for the whole well - the thing this replaces
        (`simplePressureDrop`) collapsed to a single average instead.
        """
        f = self.fluid
        vol_rat = mix_w / b_w_c + mix_o / b_o_c
        rho_mix = (mix_w * f.rhoS[0] + mix_o * f.rhoS[1]) / vol_rat
        dz = np.diff(np.concatenate(([0.0], w.dZ)))
        ddp = self.gravity * rho_mix * dz
        return np.cumsum(ddp)

    def _compute_cdp_per_well(self, st0: OWState) -> List[np.ndarray]:
        """`wellSol.cdp` for every well, computed once per timestep from
        `st0` (the previous converged state) - MRST: "computed once per
        nonlinear loop to avoid dense linear systems," i.e. frozen for
        every Newton iteration of the timestep this feeds, not
        recomputed per iteration.

        The per-perforation composition MRST reads from `wellSol.cqs`
        (the previous timestep's converged well fluxes) is reconstructed
        here by re-evaluating `_well_contributions` at `st0` itself, using
        the *previous* call's `cdp` (`self._last_cdp`, zero on the first
        call) - the same equations that produced `st0`'s own converged
        well rates, so this recovers what actually flowed rather than
        approximating it from cell mobilities alone.
        """
        if not self.gravity:
            return [np.zeros(w.dZ.size) for w in self.wells]
        f = self.fluid
        b_w0 = np.asarray(value(f.b_w(st0.pressure)), dtype=float)
        b_o0 = np.asarray(value(get_b_o_bo_ad(self, st0.pressure)), dtype=float)
        so0 = 1.0 - st0.sW
        mob_w0 = np.asarray(value(f.kr_w(st0.sW)), dtype=float) / f.mu[0]
        mob_o0 = np.asarray(value(f.kr_o(so0)), dtype=float) / f.mu[1]
        prev_cdp = getattr(self, "_last_cdp", None)
        if prev_cdp is None:
            prev_cdp = [np.zeros(w.dZ.size) for w in self.wells]

        new_cdp = []
        for k, w in enumerate(self.wells):
            cq_sw, cq_so, *_ = self._well_contributions(
                k, st0.pressure, mob_w0, mob_o0, b_w0, b_o0,
                st0.qWs[k:k + 1], st0.qOs[k:k + 1], st0.bhp[k:k + 1],
                prev_cdp[k],
            )
            mix_w, mix_o = self._segment_mix_fractions(
                w, np.asarray(value(cq_sw), dtype=float),
                np.asarray(value(cq_so), dtype=float))
            new_cdp.append(self._topology_cdp(
                w, mix_w, mix_o, b_w0[w.cells], b_o0[w.cells]))
        self._last_cdp = new_cdp
        return new_cdp

    # -- equation assembly -------------------------------------------------
    def _well_contributions(
        self, k: int, p, mob_w, mob_o, b_w, b_o, qws_k, qos_k, bhp_k, cdp
    ) -> Tuple[ADI, ADI, ADI, ADI, ADI, np.ndarray]:
        """One well's surface connection rates and closure/control eqs.

        Port of computePerforationRates/setupWellControlEquations for a
        two-phase model without dissolved components. `cdp` (one value
        per perforation) comes from `_compute_cdp_per_well`, computed once
        per timestep rather than here, matching MRST's own gating.
        """
        w = self.wells[k]
        wc = w.cells
        tw = w.WI
        compi = w.compi[:2]

        mob_w_c, mob_o_c = mob_w[wc], mob_o[wc]
        b_w_c, b_o_c = b_w[wc], b_o[wc]

        # MRST: `drawdown = -(pBH + cdp) + p`.
        drawdown = -(bhp_k + cdp) + p[wc]
        inj_conn = value(drawdown) < 0
        prod_conn = ~inj_conn

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

    def _well_contributions_batched(self, p, mob_w, mob_o, b_w, b_o, qws, qos,
                                    bhp, cdp_per_well):
        """All wells' contributions at once - same equations as
        `_well_contributions`, called once per well from a Python loop, but
        batched across every perforation of every well as single vectorized
        `ADI` operations. P2-3 (see ROADMAP.md): a real, profiled cost
        (`_well_contributions` measured at ~16% of a field-scale SPE10 AD
        run), addressed by removing the per-well Python-dispatch overhead
        rather than the per-perforation arithmetic, which is unchanged.

        Two of `_well_contributions`'s per-well branches are genuine
        semantic choices, not just skip-the-zero-work optimizations, and are
        reproduced exactly rather than only approximated by masking:

        - `any_prod` (does this well have at least one producing
          perforation?) selects *which formula* computes the wellbore
          mixture fraction `mix_w`/`mix_o` - the well's fixed injection
          composition `compi` for an all-injector well, the produced-fluid
          mixture otherwise. A well's `con_eff`-masked producer terms are
          identically zero regardless of `any_prod`, so this cannot be left
          to masking alone the way `any_inj` can.
        - The dead-well fallback (`wbqt == 0`) is folded into the same
          `compi`-vs-computed blend, using the same "shift the zero
          denominator away from zero, then blend by mask" pattern
          `_diag_mul`-based masking already relies on elsewhere in this
          port (see `ad_max`/`ad_min`): `wbqt_safe = wbqt + dead_mask`
          avoids computing `x/0` even in the branch that gets discarded,
          matching how `ad_max`/`ad_min` differentiate through only the
          winning branch without contaminating it with the other branch's
          singularities.

        `any_inj` (any injecting perforation?), by contrast, only gates
        whether the injection-mixing term is worth computing - a producer
        perforation's `con_eff_inj` is exactly zero either way, so its
        contribution is correctly zero under masking with no special case
        needed.

        Returns `(water_adjustment, oil_adjustment, eq_w, eq_o, ctrl)` where
        the first two are scattered per-cell arrays (to be subtracted from
        the reservoir equations, MRST-style) and the last three are
        per-well `ADI` vectors, matching `combine_equations`'s expectations
        the same way the per-well loop's `vertcat_adi(*eq_w_wells)` did.
        """
        wells = self.wells
        n_wells = len(wells)
        n_cells = len(p)
        if n_wells == 0:
            zero_cells = ADI(np.zeros(n_cells), [
                sp.csr_array((n_cells, j.shape[1])) for j in p.jac])
            zero_wells = ADI(np.zeros(0), [
                sp.csr_array((0, j.shape[1])) for j in p.jac])
            return (zero_cells, zero_cells, zero_wells, zero_wells, zero_wells)

        bad_types = {w.type for w in wells} - {"bhp", "rate"}
        if bad_types:
            raise ValueError(f"unsupported well control(s) {sorted(bad_types)!r}")

        perf_counts = np.array([w.cells.size for w in wells])
        wc_all = np.concatenate([w.cells for w in wells]).astype(np.int64)
        tw_all = np.concatenate([w.WI for w in wells])
        cdp_all = np.concatenate([np.asarray(c, dtype=float) for c in cdp_per_well])
        well_of_perf = np.repeat(np.arange(n_wells), perf_counts)
        n_perf = wc_all.size
        compi_w_perf = np.repeat([w.compi[0] for w in wells], perf_counts)
        compi_o_perf = np.repeat([w.compi[1] for w in wells], perf_counts)

        bhp_perf = bhp[well_of_perf]
        p_c = p[wc_all]
        mob_w_c, mob_o_c = mob_w[wc_all], mob_o[wc_all]
        b_w_c, b_o_c = b_w[wc_all], b_o[wc_all]

        drawdown = -(bhp_perf + cdp_all) + p_c
        inj_conn = value(drawdown) < 0
        prod_conn = ~inj_conn

        con_eff_prod = prod_conn.astype(float) * tw_all
        cq_pw = -con_eff_prod * mob_w_c * drawdown
        cq_po = -con_eff_prod * mob_o_c * drawdown
        cq_psw = b_w_c * cq_pw
        cq_pso = b_o_c * cq_po

        # (n_wells, n_perf): row k marks well k's own perforations - a
        # grouped sum (`grouping @ x`) and a per-cell scatter (`scatter @
        # x`, below) below are the batched analogues of the per-well
        # `ad_sum` and per-well `water[wc] = water[wc] - cq_sw` the
        # original loop does one well/cell at a time.
        grouping = sp.csr_array(
            (np.ones(n_perf), (well_of_perf, np.arange(n_perf))),
            shape=(n_wells, n_perf))
        sum_cq_psw = ad_matmul(grouping, cq_psw)
        sum_cq_pso = ad_matmul(grouping, cq_pso)

        any_prod_perwell = np.zeros(n_wells, dtype=bool)
        np.logical_or.at(any_prod_perwell, well_of_perf, prod_conn)

        qws_pos = (value(qws) > 0).astype(float)
        qos_pos = (value(qos) > 0).astype(float)
        wbq_w = qws * qws_pos - sum_cq_psw
        wbq_o = qos * qos_pos - sum_cq_pso
        wbqt = wbq_w + wbq_o

        dead = (value(wbqt) == 0.0) | (~any_prod_perwell)
        dead_f = dead.astype(float)
        wbqt_safe = wbqt + dead_f
        compi_w_well = np.array([w.compi[0] for w in wells])
        compi_o_well = np.array([w.compi[1] for w in wells])
        mix_w = (wbq_w / wbqt_safe) * (1.0 - dead_f) + compi_w_well * dead_f
        mix_o = (wbq_o / wbqt_safe) * (1.0 - dead_f) + compi_o_well * dead_f
        mix_w_perf = mix_w[well_of_perf]
        mix_o_perf = mix_o[well_of_perf]

        mt = mob_w_c + mob_o_c
        con_eff_inj = inj_conn.astype(float) * tw_all
        cqt_i = -con_eff_inj * (mt * drawdown)
        vol_rat = mix_w_perf / b_w_c + mix_o_perf / b_o_c
        cqt_is = cqt_i / vol_rat
        cq_sw = cq_psw + mix_w_perf * cqt_is
        cq_so = cq_pso + mix_o_perf * cqt_is

        # (n_cells, n_perf): scatters each perforation's contribution into
        # its cell, summing automatically where perforations of different
        # wells (or the same well) share a cell - correct even in that
        # case, unlike a fancy-index assignment would be.
        scatter = sp.csr_array(
            (np.ones(n_perf), (wc_all, np.arange(n_perf))),
            shape=(n_cells, n_perf))
        water_adj = ad_matmul(scatter, cq_sw)
        oil_adj = ad_matmul(scatter, cq_so)

        eq_w = qws - ad_matmul(grouping, cq_sw)
        eq_o = qos - ad_matmul(grouping, cq_so)

        bhp_mask = np.array([w.type == "bhp" for w in wells], dtype=float)
        well_vals = np.array([w.val for w in wells])
        ctrl = (bhp - well_vals) * bhp_mask + (qws + qos - well_vals) * (1.0 - bhp_mask)

        return water_adj, oil_adj, eq_w, eq_o, ctrl

    def assemble(self, st0: OWState, p, sw, qws, qos, bhp, dt: float,
                cdp_per_well: Optional[List[np.ndarray]] = None):
        """Residual equations for the Newton iteration (ADI or values).

        `cdp_per_well` is frozen for the whole timestep by the caller
        (`step`), matching MRST; `None` means "no gravity, or a caller
        that hasn't computed it" and is filled with zeros here so that
        direct callers (this file's own hydrostatic self-test) don't need
        to know about well cdp bookkeeping to exercise the flux term.
        """
        f = self.fluid
        ops = self.ops
        pv = self.pv

        b_w, b_o = f.b_w(p), get_b_o_bo_ad(self, p)
        so = 1.0 - sw
        mob_w = f.kr_w(sw) / f.mu[0]
        mob_o = f.kr_o(so) / f.mu[1]

        # Not wrapped in value(): when st0 carries ADI (the adjoint seeds the
        # *previous* state to get dR/dx0), these must keep their derivatives.
        # In the forward Newton st0 holds plain arrays and nothing changes.
        b_w0 = f.b_w(st0.pressure)
        b_o0 = get_b_o_bo_ad(self, st0.pressure)

        # Phase fluxes with potential upwinding (no capillary pressure).
        # MRST's `PhasePotentialDifference`: `dp - faceAvg(rho_phase)*gdz`,
        # one potential and one upwind flag per phase since the two
        # densities differ - `self.gdz` is identically zero without
        # gravity, so `pot_w == pot_o == dp` and both flags coincide with
        # the old shared `flag`, reproducing the no-gravity path exactly.
        dp = ops.grad(p)
        rho_w, rho_o = f.rhoS[0] * b_w, f.rhoS[1] * b_o
        pot_w = dp - ops.avg(rho_w) * self.gdz
        pot_o = dp - ops.avg(rho_o) * self.gdz
        flag_w = value(pot_w) <= 0
        flag_o = value(pot_o) <= 0
        v_w = -ops.T * ops.upw(flag_w, mob_w) * pot_w
        v_o = -ops.T * ops.upw(flag_o, mob_o) * pot_o

        water = (1.0 / dt) * (pv * b_w * sw - pv * b_w0 * st0.sW)
        oil = (1.0 / dt) * (pv * b_o * so - pv * b_o0 * (1.0 - st0.sW))
        water = water + ops.div(ops.upw(flag_w, b_w) * v_w)
        oil = oil + ops.div(ops.upw(flag_o, b_o) * v_o)

        if cdp_per_well is None:
            cdp_per_well = [np.zeros(w.dZ.size) for w in self.wells]

        water_adj, oil_adj, eq_w, eq_o, ctrl = self._well_contributions_batched(
            p, mob_w, mob_o, b_w, b_o, qws, qos, bhp, cdp_per_well)
        water = water - water_adj
        oil = oil - oil_adj

        return water, oil, eq_w, eq_o, ctrl

    # -- convergence and stepping ------------------------------------------
    def _converged(self, eqs, dt: float, st: OWState) -> bool:
        water, oil, eq_w, eq_o, ctrl = (value(e) for e in eqs)
        # `B_avg = mean(1/b)`, as MRST's `getConvergenceValuesCNV` does: the
        # surface-volume residual weighted back into reservoir volume. Here both
        # b are ~1 so it barely moves, but the criterion must mean the same
        # thing in every model or two of them stop at different places.
        f, pv = self.fluid, self.pv
        bw = np.mean(1.0 / np.asarray(value(f.b_w(st.pressure)), float))
        bo = np.mean(1.0 / np.asarray(value(get_b_o_bo_ad(self, st.pressure)), float))
        cnv, mb = cnv_mb_convergence([water, oil], np.array([bw, bo]), pv, dt)
        q_ref = max(max(abs(w.val) for w in self.wells if w.type == "rate"),
                    1e-12) if any(w.type == "rate" for w in self.wells) else 1.0
        well_ok = (np.abs(eq_w).max() <= self.tol_well * q_ref
                   and np.abs(eq_o).max() <= self.tol_well * q_ref)
        ctrl_ok = np.all(np.abs(ctrl) <= self.tol_well *
                         np.array([BARSA if w.type == "bhp" else q_ref
                                   for w in self.wells]))
        return (cnv <= self.tol_cnv and mb <= self.tol_mb
                and well_ok and ctrl_ok)

    def _solve_increment(self, eq, state: OWState) -> np.ndarray:
        """Solve `J dx = -r` with the configured solver, direct by default.

        `linear_solver = None` is a sparse LU and is what this model has always
        done; it stays the default so that not choosing is not a change. The
        measurements below are why choosing is worth it, and why the default is
        still LU: which one wins depends entirely on the problem.

        On 2D grids LU is *better* until roughly 100k unknowns - nested
        dissection orders a 2D mesh well and there is little fill-in, so CPR's
        setup is pure overhead (0.10x at 3.2k). In 3D that advantage is gone:

            n         LU        CPR      speed-up
            16k      133 ms    49 ms       2.7x
            64k     2043 ms   120 ms      17.0x
            196k   17181 ms   335 ms      51.4x

        LU scales n^1.94 there while CPR scales n^0.77 over this range, and the
        reason is one number: **the iteration count does not move**. Nine or ten
        GMRES iterations at every size, in 2D and 3D alike. That is the whole
        property - the AMG pressure stage is mesh-independent, so work grows
        with the cost of an iteration rather than with the conditioning.

        A caller who passes a solver gets it and its convergence tolerance,
        not this model's Newton tolerance. That matters: an inexact linear
        solve inside an exact Newton loop converges to the same answer but not
        by the same path, so iteration counts may differ from the direct run.
        """
        A = eq.jac[0]
        b = -eq.val
        if self.linear_solver is None:
            return spla.spsolve(A.tocsc(), b)
        # CPR's weighting reads `state["pressure"]`; OWState is a dataclass.
        return self.linear_solver.solve(A.tocsr(), b, self,
                                        {"pressure": state.pressure})

    def step(self, state: OWState, dt: float) -> Tuple[OWState, bool, int]:
        """One backward-Euler step solved by Newton; returns new state."""
        st0 = state
        st = state.copy()
        nc = self.g.cells.num
        nw = len(self.wells)
        # MRST computes `cdp` once per timestep, not per Newton iteration
        # (`SimpleWell.updateConnectionPressureDrop`'s `iteration ~= 1`
        # gate) - `_compute_cdp_per_well` also updates `self._last_cdp`,
        # the state that reconstruction needs on the *next* call.
        cdp_per_well = self._compute_cdp_per_well(st0)

        for it in range(self.max_iterations):
            p, sw, qws, qos, bhp = init_variables_adi(
                st.pressure, st.sW, st.qWs, st.qOs, st.bhp
            )
            eqs = self.assemble(st0, p, sw, qws, qos, bhp, dt, cdp_per_well)
            if self._converged(eqs, dt, st):
                return st, True, it
            eq = combine_equations(*eqs)
            upd = self._solve_increment(eq, st)

            dp = upd[:nc]
            ds = upd[nc:2 * nc]
            # MRST's update limits, and this *is* them rather than a local
            # approximation - which was not obvious, because this model sits
            # outside the AD-OO hierarchy and looked like it had grown its own
            # rule.
            #
            # `PhysicalModel.limit_update_relative` scales a row by
            # `min(cap/|dv|, 1)`. For a single-column update that is
            # `sign(dv)*min(|dv|, cap)` - exactly a clip. The two agree to
            # within an ulp (measured: 78 of 2000 entries differ, all at
            # relative 1e-15), and `np.clip` is the more accurate of the two
            # because it returns `cap` exactly instead of `dv*(cap/|dv|)`.
            #
            # So this stays. Swapping in the shared helper would move the last
            # bit of every limited update for no gain. The equivalence is
            # pinned by tests/ad_blackoil/test_two_phase_update_limits.py.
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
        eqs = self.assemble(st0, p, sw, qws, qos, bhp, dt, cdp_per_well)
        return st, self._converged(eqs, dt, st), self.max_iterations


class TwoPhaseOilWaterModel(ThreePhaseBlackOilModel):
    """MRST `TwoPhaseOilWaterModel`."""

    def __init__(self, G: Any, rock: Any, fluid: Any, **options: Any):
        super().__init__(G, rock, fluid)
        self.oil = True
        self.gas = False
        self.water = True
        merge_options(self, options)

    def get_equations(self, state0: Any, state: Any, dt: float,
                      driving_forces: Any, **options: Any):
        return equations_oil_water(state0, state, self, dt, driving_forces,
                                   **options)

    getEquations = get_equations


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
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

    def test_hydrostatic_water_column_has_zero_flux():
        """A single-phase water column at hydrostatic equilibrium must carry
        zero flux across every face - the standard check that the gravity
        term added to `assemble`'s phase potential exactly cancels a
        hydrostatic pressure gradient. `st0 == st` here, so the accumulation
        term is exactly zero and the residual is the flux divergence alone;
        reverting the gravity term (`pot = grad(p)` again) would make this
        fail, since a hydrostatic `p` is then no longer a zero-flux state."""
        from pymrst.params import NORM_GRAVITY

        nz = 5
        g = compute_geometry(cart_grid([1, 1, nz], [10.0, 10.0, 10.0 * nz]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        one = lambda p, *a, **k: np.ones_like(np.asarray(p, dtype=float))
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 700.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA,
        )
        # A zero-WI "null" well contributes exactly zero to every term
        # below regardless of drawdown - it exists only to keep `assemble`
        # callable without perturbing the column.
        wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                         val=0.0, compi=np.array([1.0, 0.0]), sign=1,
                         wi=np.array([0.0]))
        model = TwoPhaseOilWaterModelAD(g, rock, fluid, wells, gravity=NORM_GRAVITY)
        z = g.cells.centroids[:, 2]
        p = 200 * BARSA + fluid.rhoS[0] * NORM_GRAVITY * z
        sw = np.ones(g.cells.num)          # sW=1 makes mob_o == 0 everywhere,
        st0 = OWState(p.copy(), sw.copy(), np.array([0.0]), np.array([0.0]),
                      np.array([p[0]]))
        water, oil, _, _, _ = model.assemble(
            st0, p, sw, np.array([0.0]), np.array([0.0]), np.array([p[0]]),
            dt=1.0,
        )
        assert np.allclose(value(water), 0.0, atol=1e-8), value(water)
        print("ok test_hydrostatic_water_column_has_zero_flux")

    def test_gravity_speeds_denser_water_injection_downward():
        """Water denser than oil, injected at the top of a vertical column
        with a producer at the bottom: gravity should assist the downward
        front, so the bottom cell's water saturation after N steps with
        gravity on must exceed the same run with gravity off. Falsifiable:
        reverting `self.gdz`/`_compute_cdp_per_well` to zero collapses the
        two runs to identical, and this assertion catches that."""
        from pymrst.params import NORM_GRAVITY

        g = compute_geometry(cart_grid([1, 1, 6], [10.0, 10.0, 60.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 700.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-6 / BARSA], p_ref=200 * BARSA,
        )
        gvec = np.array([0.0, 0.0, NORM_GRAVITY])

        def bottom_sw_after_injection(gravity):
            wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                             val=2e-6, compi=np.array([1.0, 0.0]), sign=1,
                             gravity=gvec)
            wells = add_well(wells, g, rock, np.array([5]), well_type="bhp",
                             val=190 * BARSA, compi=np.array([0.5, 0.5]),
                             sign=-1, gravity=gvec)
            m = TwoPhaseOilWaterModelAD(g, rock, fluid, wells, gravity=gravity)
            st = m.initial_state(200 * BARSA, 0.2)
            for _ in range(8):
                st, ok, _ = m.step(st, 5 * DAY)
                assert ok
            return st.sW[-1]

        sw_gravity = bottom_sw_after_injection(NORM_GRAVITY)
        sw_no_gravity = bottom_sw_after_injection(0.0)
        assert sw_gravity > sw_no_gravity + 0.01, (sw_gravity, sw_no_gravity)
        print("ok test_gravity_speeds_denser_water_injection_downward "
             f"(gravity sW={sw_gravity:.4f}, no-gravity sW={sw_no_gravity:.4f})")

    def test_segment_mix_fractions_matches_hand_computed_reverse_cumsum():
        """`_segment_mix_fractions`: the flow through a segment near the top
        of a simple chain must carry everything perforated at or below it
        (MRST's `wb2in`/`C\\qs` for the trivial identity-chain topology),
        with a `compi` fallback where the cumulative segment flow is zero."""
        from pymrst.params import NORM_GRAVITY

        g = compute_geometry(cart_grid([1, 1, 3], [10.0, 10.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 700.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA,
        )
        gvec = np.array([0.0, 0.0, NORM_GRAVITY])
        wells = add_well(None, g, rock, np.array([0, 1, 2]), well_type="bhp",
                         val=190 * BARSA, compi=np.array([0.3, 0.7]),
                         sign=-1, gravity=gvec)
        model = TwoPhaseOilWaterModelAD(g, rock, fluid, wells,
                                        gravity=NORM_GRAVITY)
        w = model.wells[0]

        cq_w = np.array([1.0, 2.0, 0.0])
        cq_o = np.array([0.5, 0.0, 0.0])       # bottom perforation is dead
        mix_w, mix_o = model._segment_mix_fractions(w, cq_w, cq_o)
        # segment i carries perforation i plus everything below it
        seg_w_expected = np.array([3.0, 2.0, 0.0])
        seg_o_expected = np.array([0.5, 0.0, 0.0])
        seg_t_expected = seg_w_expected + seg_o_expected
        expected_mix_w = np.where(seg_t_expected == 0, w.compi[0],
                                  seg_w_expected / np.where(seg_t_expected == 0, 1, seg_t_expected))
        expected_mix_o = np.where(seg_t_expected == 0, w.compi[1],
                                  seg_o_expected / np.where(seg_t_expected == 0, 1, seg_t_expected))
        assert np.allclose(mix_w, expected_mix_w), (mix_w, expected_mix_w)
        assert np.allclose(mix_o, expected_mix_o), (mix_o, expected_mix_o)
        assert mix_w[-1] == w.compi[0] and mix_o[-1] == w.compi[1], (
            "dead bottom segment must fall back to compi")
        print("ok test_segment_mix_fractions_matches_hand_computed_reverse_cumsum")

    def test_topology_cdp_matches_hand_computed_value():
        """`_topology_cdp`'s per-perforation `rhoMix = sum(mix*rhoS)/sum(mix/b)`
        then `cumsum(g*rhoMix*dz)`, checked against the same computation done
        independently here with two perforations carrying *different*
        mixtures - the case a single averaged density (the old
        `simplePressureDrop` formula this replaced) could not represent."""
        from pymrst.params import NORM_GRAVITY

        g = compute_geometry(cart_grid([1, 1, 2], [10.0, 10.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 700.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA,
        )
        gvec = np.array([0.0, 0.0, NORM_GRAVITY])
        wells = add_well(None, g, rock, np.array([0, 1]), well_type="bhp",
                         val=190 * BARSA, compi=np.array([0.5, 0.5]),
                         sign=-1, gravity=gvec)
        model = TwoPhaseOilWaterModelAD(g, rock, fluid, wells,
                                        gravity=NORM_GRAVITY)
        w = model.wells[0]
        assert w.dZ.size == 2 and not np.allclose(w.dZ, 0.0)

        mix_w, mix_o = np.array([0.8, 0.3]), np.array([0.2, 0.7])
        b_w_c, b_o_c = np.array([1.0, 1.0]), np.array([1.0, 1.0])
        cdp = model._topology_cdp(w, mix_w, mix_o, b_w_c, b_o_c)

        rho_w, rho_o = fluid.rhoS[0], fluid.rhoS[1]
        vol_rat = mix_w / b_w_c + mix_o / b_o_c
        rho_mix = (mix_w * rho_w + mix_o * rho_o) / vol_rat
        dz = np.diff(np.concatenate(([0.0], w.dZ)))
        expected = np.cumsum(NORM_GRAVITY * rho_mix * dz)
        assert np.allclose(cdp, expected), (cdp, expected)
        # the two perforations' densities differ (0.8/0.2 vs 0.3/0.7 water/oil
        # mix), so a single averaged density could not reproduce this cdp.
        assert not np.isclose(rho_mix[0], rho_mix[1])

        zero = TwoPhaseOilWaterModelAD(
            g, rock, fluid, wells, gravity=0.0
        )._topology_cdp(model.wells[0], mix_w, mix_o, b_w_c, b_o_c)
        assert np.allclose(zero, 0.0), "gravity=0 must give cdp==0 exactly"
        print("ok test_topology_cdp_matches_hand_computed_value")

    def test_cdp_is_frozen_for_the_whole_timestep():
        """`step` computes `cdp_per_well` once from `st0` and must reuse the
        same array through every Newton iteration of that timestep (MRST's
        `iteration ~= 1` gate) - spies on `_compute_cdp_per_well` and checks
        it is called exactly once per `step`, however many Newton
        iterations that step needs."""
        from pymrst.params import NORM_GRAVITY

        g = compute_geometry(cart_grid([1, 1, 6], [10.0, 10.0, 60.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 700.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-6 / BARSA], p_ref=200 * BARSA,
        )
        gvec = np.array([0.0, 0.0, NORM_GRAVITY])
        wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                         val=2e-6, compi=np.array([1.0, 0.0]), sign=1,
                         gravity=gvec)
        wells = add_well(wells, g, rock, np.array([5]), well_type="bhp",
                         val=190 * BARSA, compi=np.array([0.5, 0.5]),
                         sign=-1, gravity=gvec)
        m = TwoPhaseOilWaterModelAD(g, rock, fluid, wells, gravity=NORM_GRAVITY)

        calls = []
        _orig = TwoPhaseOilWaterModelAD._compute_cdp_per_well
        def spy(self, st0):
            calls.append(1)
            return _orig(self, st0)
        TwoPhaseOilWaterModelAD._compute_cdp_per_well = spy
        try:
            st = m.initial_state(200 * BARSA, 0.2)
            st, ok, its = m.step(st, 5 * DAY)
        finally:
            TwoPhaseOilWaterModelAD._compute_cdp_per_well = _orig
        assert ok and its >= 1, its
        assert len(calls) == 1, (
            f"expected exactly 1 call for {its} Newton iterations, got {len(calls)}")
        print(f"ok test_cdp_is_frozen_for_the_whole_timestep ({its} iterations, "
             f"{len(calls)} cdp computation)")

    def test_batched_well_contributions_matches_the_per_well_loop():
        """P2-3 (ROADMAP.md): `_well_contributions_batched` replaced a
        per-well Python loop calling `_well_contributions` once per well -
        a real, profiled ~16% of a field-scale SPE10 AD run. Two of that
        loop's branches are genuine semantic choices (which formula
        computes the wellbore mixture fraction), not just skip-the-
        zero-work shortcuts, so this checks the batched result against the
        original loop directly - not just "the model still converges" -
        across scenarios chosen to exercise each one: a plain
        producer+injector pair, a single well with cross-flow perforations
        (one producing, one injecting), a dead well (`qws == qos == 0`),
        two different wells perforating the *same* cell, and all of that
        together with three wells at once.
        """
        def _fluid():
            return init_simple_adi_fluid(
                mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
                n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
            )

        def _compare(model, p_val, sw_val, qws_val, qos_val, bhp_val):
            from pymrst.utils.adi import vertcat_adi

            p, sw, qws, qos, bhp = init_variables_adi(
                p_val, sw_val, qws_val, qos_val, bhp_val)
            f = model.fluid
            so = 1.0 - sw
            mob_w, mob_o = f.kr_w(sw) / f.mu[0], f.kr_o(so) / f.mu[1]
            b_w, b_o = f.b_w(p), get_b_o_bo_ad(model, p)
            cdp_per_well = [np.zeros(w.dZ.size) for w in model.wells]

            n = len(p)
            water_ref = ADI(np.zeros(n), [sp.csr_array((n, j.shape[1])) for j in p.jac])
            oil_ref = ADI(np.zeros(n), [sp.csr_array((n, j.shape[1])) for j in p.jac])
            eq_w_wells, eq_o_wells, ctrls = [], [], []
            for k in range(len(model.wells)):
                cq_sw, cq_so, eq_w, eq_o, ctrl, wc = model._well_contributions(
                    k, p, mob_w, mob_o, b_w, b_o,
                    qws[k:k + 1], qos[k:k + 1], bhp[k:k + 1], cdp_per_well[k])
                water_ref[wc] = water_ref[wc] + cq_sw
                oil_ref[wc] = oil_ref[wc] + cq_so
                eq_w_wells.append(eq_w)
                eq_o_wells.append(eq_o)
                ctrls.append(ctrl)
            eq_w_ref = vertcat_adi(*eq_w_wells)
            eq_o_ref = vertcat_adi(*eq_o_wells)
            ctrl_ref = vertcat_adi(*ctrls)

            water_b, oil_b, eq_w_b, eq_o_b, ctrl_b = model._well_contributions_batched(
                p, mob_w, mob_o, b_w, b_o, qws, qos, bhp, cdp_per_well)

            for name, ref, got in (("water", water_ref, water_b), ("oil", oil_ref, oil_b),
                                   ("eq_w", eq_w_ref, eq_w_b), ("eq_o", eq_o_ref, eq_o_b),
                                   ("ctrl", ctrl_ref, ctrl_b)):
                assert np.allclose(ref.val, got.val, atol=1e-10), (name, "value mismatch")
                for ja, jb in zip(ref.jac, got.jac):
                    assert np.allclose(ja.toarray(), jb.toarray(), atol=1e-10), (name, "jacobian mismatch")

        g = compute_geometry(cart_grid([8, 8, 1], [200.0, 200.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        sw0 = np.full(64, 0.2)
        p0 = np.full(64, 200 * BARSA)

        # Plain producer + injector. The injector's own cell pressure must
        # sit strictly below its bhp so its one perforation is genuinely
        # classified `inj_conn` (drawdown < 0) - this is what exercises
        # `any_prod == False` (skip the wellbore-mixture computation
        # entirely, use `compi` directly), the branch the dead-well
        # scenario below, by construction, cannot reach (there, drawdown
        # comes out exactly zero either way). `compi` is deliberately
        # *not* [1, 0] to match `qws`'s own water fraction - an earlier
        # draft of this test used [1, 0] with an all-water qws and did not
        # notice a real bug (a missing `any_prod` term in the dead-well
        # guard) because `qws/(qws+qos)` coincidentally equalled `compi[0]`
        # anyway, so the broken and correct formulas agreed by accident.
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=1e-3,
                    compi=np.array([0.7, 0.3]), sign=1)
        w = add_well(w, g, rock, np.array([63]), well_type="bhp", val=150 * BARSA,
                    compi=np.array([0.0, 1.0]), sign=-1)
        p_pi = p0.copy()
        p_pi[0] = 150 * BARSA  # below the injector's own bhp -> drawdown < 0
        _compare(TwoPhaseOilWaterModelAD(g, rock, _fluid(), w), p_pi, sw0,
                 np.array([1e-3, -5e-4]), np.array([0.0, -5e-4]),
                 np.array([200 * BARSA, 150 * BARSA]))

        # Single well, cross-flow: one perforation producing, one injecting.
        w = add_well(None, g, rock, np.array([0, 7]), well_type="bhp", val=180 * BARSA,
                    compi=np.array([1.0, 0.0]), sign=0)
        p_cf = p0.copy()
        p_cf[0], p_cf[7] = 220 * BARSA, 150 * BARSA
        _compare(TwoPhaseOilWaterModelAD(g, rock, _fluid(), w), p_cf, sw0,
                 np.array([0.0]), np.array([0.0]), np.array([180 * BARSA]))

        # Dead injector: qws == qos == 0 forces wbqt == 0.
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=0.0,
                    compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([63]), well_type="bhp", val=150 * BARSA,
                    compi=np.array([0.0, 1.0]), sign=-1)
        _compare(TwoPhaseOilWaterModelAD(g, rock, _fluid(), w), p0, sw0,
                 np.array([0.0, -5e-4]), np.array([0.0, -5e-4]),
                 np.array([200 * BARSA, 150 * BARSA]))

        # Two different wells perforating the same cell, each with a
        # genuinely nonzero contribution there (the rate well's own bhp is
        # a free Newton variable, not its control - left equal to the
        # cell's pressure it would give drawdown = 0 and contribute
        # nothing, silently failing to exercise whether contributions from
        # multiple wells at one cell are actually summed rather than one
        # overwriting the other).
        w = add_well(None, g, rock, np.array([10]), well_type="rate", val=5e-4,
                    compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([10]), well_type="bhp", val=150 * BARSA,
                    compi=np.array([0.0, 1.0]), sign=-1)
        _compare(TwoPhaseOilWaterModelAD(g, rock, _fluid(), w), p0, sw0,
                 np.array([5e-4, -5e-4]), np.array([0.0, -5e-4]),
                 np.array([250 * BARSA, 150 * BARSA]))

        # Three wells together: producer, injector, and a cross-flow well.
        # Well 0's compi is mismatched from qws/qos the same way as the
        # producer+injector scenario above, for the same reason.
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=1e-3,
                    compi=np.array([0.7, 0.3]), sign=1)
        w = add_well(w, g, rock, np.array([63]), well_type="bhp", val=150 * BARSA,
                    compi=np.array([0.0, 1.0]), sign=-1)
        w = add_well(w, g, rock, np.array([20, 27]), well_type="bhp", val=180 * BARSA,
                    compi=np.array([0.5, 0.5]), sign=0)
        p5 = p0.copy()
        p5[0] = 150 * BARSA  # below well 0's own bhp -> genuine injection
        p5[20], p5[27] = 220 * BARSA, 150 * BARSA
        _compare(TwoPhaseOilWaterModelAD(g, rock, _fluid(), w), p5, sw0,
                 np.array([1e-3, -5e-4, 0.0]), np.array([0.0, -5e-4, 0.0]),
                 np.array([200 * BARSA, 150 * BARSA, 180 * BARSA]))

        print("ok test_batched_well_contributions_matches_the_per_well_loop "
             "(producer+injector, cross-flow, dead well, shared cell, 3 wells)")

    test_step_converges()
    test_surface_volume_balance()
    test_water_front_moves_toward_producer()
    test_hydrostatic_water_column_has_zero_flux()
    test_gravity_speeds_denser_water_injection_downward()
    test_segment_mix_fractions_matches_hand_computed_reverse_cumsum()
    test_topology_cdp_matches_hand_computed_value()
    test_cdp_is_frozen_for_the_whole_timestep()
    test_batched_well_contributions_matches_the_per_well_loop()
    print("\nAll tests passed!")
