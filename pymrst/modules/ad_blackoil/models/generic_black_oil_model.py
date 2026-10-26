"""Black-oil assembled from the state-function graph (legacy; MRST GenericBlackOilModel).

This is the same physics as `oil_water_model.py` and `blackoil_model.py`,
but nothing here assembles a residual. The reservoir equations come out of
`FlowDiscretization.component_conservation_equations`, which does not know
what physics it is assembling; the wells come out of `FacilityModel`; and
this class only has to say what the primary variables are and how they are
updated.

That is the whole argument for the graph, and the argument is testable:
this model must reproduce the hand-rolled ones - which are validated
against MATLAB to 1e-10 (SPE 1) and 1e-14 (oil-water) - to round-off.
`test_matches_hand_rolled_*` below pins exactly that, and
`examples/validation/validate_phase24.py` re-runs the MATLAB comparisons
of Phases 5a and 5b through this model.

Two phases (water, oil) or three with live oil (water, oil, gas + Rs); the
hydrocarbon variable switches meaning per cell as in `blackoil_model.py`.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_core.utils.boundary_conditions import add_bc_and_sources
from pymrst.modules.ad_core.utils.convergence import (
    cnv_mb_convergence,
    max_extra_residual,
    well_control_residual,
)
from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.facilities.generic_facility_model import (
    GenericFacilityModel,
)
from pymrst.modules.ad_core.models.facilities.multisegment import (
    assemble_multisegment_equations,
)
from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.models.generic_reservoir_model import (
    GenericReservoirModel,
)
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    blackoil_phase_fluid,
    simple_phase_fluid,
    simple_three_phase_fluid,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_blackoil.utils.compute_flash_black_oil import (
    compute_flash_black_oil_ad,
)
from pymrst.modules.ad_blackoil.utils.get_cell_status_vo import (
    get_cell_status_vo,
)
from pymrst.modules.ad_blackoil.utils.calculate_hydrocarbons_from_status_bo import (
    calculate_hydrocarbons_from_status_bo,
)
from pymrst.modules.ad_blackoil.utils.update_state_black_oil_generic import (
    update_state_black_oil_generic,
)
from pymrst.modules.ad_blackoil.components.gas_component import GasComponent
from pymrst.modules.ad_blackoil.components.oil_component import OilComponent
from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    ThreePhaseBlackOilModel,
)
from pymrst.modules.ad_core.ad_oo.generic_facility_model import (
    GenericFacilityModel as AdOoGenericFacilityModel,
)
from pymrst.modules.ad_core.ad_oo.generic_reservoir_model import (
    GenericReservoirModel as AdOoGenericReservoirModel,
)
from pymrst.modules.ad_core.ad_oo.physical_model import (
    PhysicalModel as AdOoPhysicalModel,
    copy_state,
)
from pymrst.modules.ad_core.ad_oo.reservoir_model import (
    ReservoirModel as AdOoReservoirModel,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.components.immiscible_component import (
    ImmiscibleComponent,
)
from pymrst.modules.ad_core.utils.get_boundary_condition_fluxes_ad import (
    get_boundary_condition_fluxes_ad,
)
from pymrst.modules.ad_core.utils.expand_matrix_to_cell import expand_matrix_to_cell
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import init_variables_adi, value, vertcat_adi
from pymrst.modules.ad_core.models.aquifers.aquifer_model import add_aquifers_contribution

ETOL: float = np.sqrt(np.finfo(float).eps)


class GenericBlackOilModelAD(GenericReservoirModel):
    """Two- or three-phase black-oil, assembled from the state-function graph."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid,
        wells: Sequence[Well],
        disgas: bool = False,
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
        bc=None,
        src=None,
        vapoil: bool = False,
        aquifer=None,
        mass_basis: bool = False,
        operators=None,
    ):
        # The phase count comes from the *fluid*, not from disgas/vapoil.
        # MRST's `GenericBlackOilModel(G, rock, fluid, 'disgas', false,
        # 'vapoil', false)` on a three-phase fluid is a three-phase dead-oil,
        # dry-gas model; deciding on the dissolution flags instead dropped its
        # gas and left a two-phase model that runs.
        # `n_phases_declared` is stated by every fluid constructor; the
        # `rhoS` fallback is for a hand-built one. "Has a `b_g`" is *not* a
        # usable test - `deck_fluid_object` fills the gas slots of a two-phase
        # deck fluid with a refusing placeholder.
        declared = int(getattr(fluid, "n_phases_declared", 0)) or int(
            np.size(getattr(fluid, "rhoS", ())))
        if disgas or vapoil:
            phase_fluid = blackoil_phase_fluid(fluid, disgas=disgas,
                                               vapoil=vapoil)
        elif declared == 3:
            phase_fluid = simple_three_phase_fluid(fluid)
        else:
            phase_fluid = simple_phase_fluid(fluid)
        operators = (setup_operators_tpfa(g, rock)
                     if operators is None else operators)

        n_if = operators.T.size
        if gravity:
            z = g.cells.centroids[:, -1]
            gdz = gravity * (z[operators.N[:, 1]] - z[operators.N[:, 0]])
        else:
            gdz = np.zeros(n_if)

        super().__init__(g, rock, phase_fluid, operators,
                         pore_volume(g, rock), gdz, gravity=gravity,
                         mass_basis=mass_basis)
        self.raw_fluid = fluid
        self.disgas = disgas
        self.vapoil = vapoil
        # MRST's "driving forces": a model can be driven by wells, by its
        # boundary, or by sources - and most real cases use more than one.
        self.bc = bc
        self.src = src
        # An analytic aquifer (ad_core.aquifer.AquiferModel) is another driving
        # force: it feeds water in through the boundary cells it is attached to.
        # Its pressure and volume ride on the state (aquifer_p/aquifer_V), so
        # `aquifer.init_state` must seed the initial state before simulating.
        self.aquifer = aquifer
        # MRST gives a `GenericReservoirModel` a `GenericFacilityModel` and
        # everything else a plain `FacilityModel` (ReservoirModel.m:104-111),
        # and this is the generic model. The subclass overrides *nothing* - all
        # sixteen of its names are new - so this adds the reservoir-volume
        # control conversion without changing a single existing behaviour.
        self.facility = GenericFacilityModel(wells, disgas=disgas,
                                             vapoil=vapoil)
        self.ds_max = ds_max
        self.dp_max_rel = dp_max_rel
        self.tol_cnv = tol_cnv
        self.tol_mb = tol_mb
        self.tol_well = tol_well
        self.validate_model()

    @property
    def wells(self) -> List[Well]:
        return self.facility.wells

    def _ms_well_indices(self) -> List[int]:
        return [
            i for i, w in enumerate(self.wells)
            if getattr(w, "multisegment", None) is not None
        ]

    def _simple_well_indices(self) -> List[int]:
        ms = set(self._ms_well_indices())
        return [i for i in range(len(self.wells)) if i not in ms]

    def _ensure_ms_state(self, st: State) -> State:
        """Seed aggregate multisegment primary arrays on a state if absent."""
        nph = self.n_phases
        p_parts, v_parts = [], []
        comp_parts = [[] for _ in range(nph)]
        for wi in self._ms_well_indices():
            w = self.wells[wi]
            n_nodes = len(w.multisegment.segments)
            n_seg = max(n_nodes - 1, 0)
            if n_nodes <= 1:
                continue
            p_ref = float(st["bhp"][wi])
            p_parts.append(np.full(n_nodes - 1, p_ref))
            node_comp = self._initial_ms_node_comp(w, st, n_nodes)
            for ph in range(nph):
                comp_parts[ph].append(node_comp[:, ph])
            rates = np.array([
                abs(float(st[q][wi]))
                for q in ["qWs", "qOs", "qGs"][:nph]
            ])
            rho_s = np.asarray(self.fluid.rho_s[:nph], dtype=float)
            total_mass_rate = float(np.dot(rates, rho_s))
            # Segment flux is mass rate along parent -> child. Distribute the
            # initial well mass rate evenly over completions and accumulate it
            # along every completion-to-root path. This gives the correct sign
            # and branch loading for both injectors and producers.
            v0 = np.zeros(n_seg)
            perf_nodes = np.asarray(
                w.multisegment.perforation_segment, dtype=np.int64
            )
            if perf_nodes.size:
                branch_rate = total_mass_rate / perf_nodes.size
                direction = 1.0 if w.sign >= 0 else -1.0
                for node in perf_nodes:
                    for child in w.multisegment.path_to_root(int(node))[:-1]:
                        if child > 0:
                            v0[child - 1] += direction * branch_rate
            v_parts.append(v0)
        if p_parts and "ms_pN" not in st:
            st["ms_pN"] = np.concatenate(p_parts)
        if v_parts and "ms_vmix" not in st:
            st["ms_vmix"] = np.concatenate(v_parts)
        for ph, name in enumerate(["ms_rW", "ms_rO", "ms_rG"][:nph]):
            if comp_parts[ph] and name not in st:
                st[name] = np.concatenate(comp_parts[ph])
        return st

    def _initial_ms_node_comp(self, w: Well, st: State, n_nodes: int) -> np.ndarray:
        """MRST-style initial non-root MS node component fractions."""
        nph = self.n_phases
        base = np.zeros(nph)
        wi = next((i for i, candidate in enumerate(self.wells) if candidate is w), None)
        if wi is not None:
            rates = np.array([
                abs(float(st[name][wi]))
                for name in ["qWs", "qOs", "qGs"][:nph]
            ])
            base = rates * np.asarray(self.fluid.rho_s[:nph], dtype=float)
        if base.sum() <= 0.0:
            base[:min(nph, w.compi.size)] = np.asarray(w.compi[:nph], dtype=float)
        if base.sum() <= 0.0:
            base[:] = 1.0 / nph
        else:
            base /= base.sum()
        out = np.tile(base, (n_nodes - 1, 1))
        if self.disgas or "sW" not in st or nph != 2:
            return out

        wc = np.asarray(w.cells, dtype=np.int64)
        if wc.size == 0:
            return out
        p = np.asarray(st["pressure"], dtype=float)[wc]
        sw = np.asarray(st["sW"], dtype=float)[wc]
        so = np.maximum(1.0 - sw, 0.0)
        bw = np.asarray(value(self.raw_fluid.b_w(p)), dtype=float)
        from pymrst.modules.ad_blackoil.utils.get_b_o_bo import get_b_o_bo_ad

        bo = np.asarray(value(get_b_o_bo_ad(self, p)), dtype=float)
        mw = np.asarray(value(self.raw_fluid.kr_w(sw)), dtype=float) / self.raw_fluid.mu[0]
        mo = np.asarray(value(self.raw_fluid.kr_o(so)), dtype=float) / self.raw_fluid.mu[1]
        weights = np.column_stack([
            self.raw_fluid.rhoS[0] * bw * mw,
            self.raw_fluid.rhoS[1] * bo * mo,
        ])
        denom = weights.sum(axis=1)
        good = denom > 0.0
        frac = np.tile(base, (wc.size, 1))
        frac[good] = weights[good] / denom[good, None]

        perf_node = np.asarray(w.multisegment.perforation_segment, dtype=np.int64)
        if perf_node.size == 0:
            perf_node = np.arange(wc.size, dtype=np.int64) + 1
        perf_node = np.clip(perf_node, 1, n_nodes - 1)
        counts = np.zeros(n_nodes - 1)
        acc = np.zeros((n_nodes - 1, nph))
        for j, node in enumerate(perf_node):
            acc[node - 1] += frac[j]
            counts[node - 1] += 1.0
        hit = counts > 0.0
        out[hit] = acc[hit] / counts[hit, None]
        return out

    def _ms_slices(self):
        """Return per-MS-well slices into aggregate MS state arrays."""
        out, p0, v0, r0 = [], 0, 0, 0
        for wi in self._ms_well_indices():
            w = self.wells[wi]
            n_nodes = len(w.multisegment.segments)
            n_seg = max(n_nodes - 1, 0)
            out.append((wi, slice(p0, p0 + n_nodes - 1),
                        slice(r0, r0 + n_nodes - 1), slice(v0, v0 + n_seg)))
            p0 += n_nodes - 1
            r0 += n_nodes - 1
            v0 += n_seg
        return out

    # -- primary variables ---------------------------------------------------
    @staticmethod
    def _status_masks(status):
        """MRST's `getCellStatusVO`: which of the three things `x` means here.

        `status = oil_present + 2*gas_present`, so:

        - **1** - oil but no free gas. The oil is *undersaturated*: how much
          gas it holds is a free variable, so `x` is `rs`.
        - **2** - gas but no free oil. The gas is undersaturated in the mirror
          sense, and `x` is `rv`.
        - **3** - both phases present. Neither dissolution is free (both sit on
          their saturation curves), and what is free is *how much gas there
          is*: `x` is `sG`.

        This is the whole of black-oil's variable switching, and it is a switch
        rather than a fixed choice because at a phase boundary one unknown
        vanishes and another appears in its place.
        """
        st1 = status == 1
        st2 = status == 2
        st3 = status == 3
        return st1, st2, st3

    def init_state_ad(self, st: State) -> State:
        """Seed the AD variables and resolve the hydrocarbon switch.

        `sG`, `rs` and `rv` are all *derived* from the single primary variable
        `x` here rather than read off the state - which is exactly why
        `Saturations` is a state function and not a lookup.
        """
        nph = self.n_phases
        q_names = ["qWs", "qOs", "qGs"][:nph]
        self._ensure_ms_state(st)
        ms_names = []
        if self._ms_well_indices():
            ms_names = ["ms_pN", *["ms_rW", "ms_rO", "ms_rG"][:nph],
                        "ms_vmix"]

        if not (self.disgas or self.vapoil):
            # A three-phase model without dissolution still has two independent
            # saturations. There is no hydrocarbon *switch* - `sG` is simply a
            # second primary variable rather than a derived one - but omitting
            # it leaves the gas equation with no unknown of its own, and the
            # gas component with no phase to live in.
            gas = nph > 2
            extra = [st["sG"]] if gas else []
            prim = init_variables_adi(st["pressure"], st["sW"], *extra,
                                      *[st[q] for q in q_names], st["bhp"],
                                      *[st[n] for n in ms_names])
            offset = 2 + int(gas)
            p, sw = prim[0], prim[1]
            out = State(pressure=p, sW=sw, bhp=prim[offset + nph])
            if gas:
                out["sG"] = prim[2]
                out["sO"] = 1.0 - sw - prim[2]
            for i, q in enumerate(q_names):
                out[q] = prim[offset + i]
            for i, name in enumerate(ms_names):
                out[name] = prim[offset + nph + 1 + i]
            if "implicit" in st:
                out["implicit"] = np.asarray(st["implicit"], dtype=bool).copy()
            return out

        st1, st2, st3 = self._status_masks(st["status"])
        x_val = np.where(st1, st["rs"], np.where(st2, st.get("rv", 0.0),
                                                st["sG"]))
        prim = init_variables_adi(st["pressure"], st["sW"], x_val,
                                  *[st[q] for q in q_names], st["bhp"],
                                  *[st[n] for n in ms_names])
        p, sw, x = prim[0], prim[1], prim[2]
        sg, rs, rv, _, _ = calculate_hydrocarbons_from_status_bo(
            self.raw_fluid, (st1, st2, st3), 1.0-sw, x,
            st["rs"], st.get("rv", np.zeros_like(st["rs"])), p,
            self.disgas, self.vapoil,
        )

        out = State(pressure=p, sW=sw, sG=sg, rs=rs, status=st["status"],
                    bhp=prim[3 + nph])
        if self.vapoil:
            out["rv"] = rv
        for i, q in enumerate(q_names):
            out[q] = prim[3 + i]
        for i, name in enumerate(ms_names):
            out[name] = prim[3 + nph + 1 + i]
        if "implicit" in st:
            out["implicit"] = np.asarray(st["implicit"], dtype=bool).copy()
        return out

    def _state0_props(self, st0: State) -> State:
        """The previous state, put through *the same reconstruction* as the new
        one (MRST's `calculateHydrocarbonsFromStatusBO`, applied to `state0`).

        This is not bookkeeping. The accumulation term differences two states,
        so they must mean the same thing - and a cell's stored `rs`/`rv` are
        only half the story. Where the oil is undersaturated the *gas* has no
        free oil to be undersaturated against, so `rv` is not a free variable
        there at all: it is `rvSat(p)`, whatever number happens to be sitting
        in the array. Take the array at face value and the previous state
        silently claims the gas held no condensate while the new one says it
        held a saturated amount - a difference the solver then dutifully tries
        to conserve.

        MRST reconstructs; copying the raw fields only *looked* right for
        dissolved gas alone, because there the flash already leaves them
        consistent. An initial state, which has never been flashed, does not.
        """
        # `well_flux` must survive the reduction. It is the previous step's
        # perforation fluxes, and it is what the wellbore column is weighed
        # with - drop it and the well silently falls back to the no-history
        # branch on *every* step, which is a different (and wrong) column that
        # no single-perforation case could ever have noticed.
        def _carry(out: State) -> State:
            if "well_flux" in st0:
                out["well_flux"] = st0["well_flux"]
            return out

        if not (self.disgas or self.vapoil):
            out = State(pressure=st0["pressure"], sW=st0["sW"])
            if self.n_phases > 2:
                # A three-phase model without dissolution has no hydrocarbon
                # switch to reconstruct, but it still has two independent
                # saturations - and the accumulation term differences this
                # state against the new one, so it must carry the same ones.
                out["sG"] = st0["sG"]
                out["sO"] = 1.0 - np.asarray(st0["sW"]) - np.asarray(st0["sG"])
            return _carry(out)

        p0, sw0 = st0["pressure"], st0["sW"]
        st1, st2, st3 = self._status_masks(st0["status"])
        x0 = np.where(st1, st0["rs"],
                      np.where(st2, st0.get("rv", 0.0), st0["sG"]))
        sg0, rs0, rv0, _, _ = calculate_hydrocarbons_from_status_bo(
            self.raw_fluid, (st1, st2, st3), 1.0-sw0, x0,
            st0["rs"], st0.get("rv", np.zeros_like(st0["rs"])), p0,
            self.disgas, self.vapoil,
        )
        out = State(pressure=p0, sW=sw0, sG=sg0, rs=rs0,
                    status=st0["status"])
        if self.vapoil:
            out["rv"] = rv0
        return _carry(out)

    # -- equations -----------------------------------------------------------
    def get_equations(self, state0: State, state: State, dt: float):
        """Reservoir equations from the graph, well equations from the facility."""
        # `update_state` may have switched a well to a violated limit and left
        # `within_limits=False` to force one more assembly.  This *is* that
        # assembly: all equations below use the new controls, so the stale
        # flag must not reject a converged residual and provoke an unnecessary
        # extra Newton/linear solve.  Any further violation is detected after
        # the next update and will set the flag again.
        self.facility.within_limits = True
        st = self.init_state_ad(state)
        # remembered for `get_convergence`, which needs b and pv to form
        # MRST's CNV norm and is handed only the residuals
        self._conv_state = st
        st0 = self._state0_props(state0)
        nph = self.n_phases

        eqs = self.conservation_equations(st0, st, dt)
        eqs = add_bc_and_sources(self, eqs, st, self.bc, self.src)
        if self.aquifer is not None:
            # p_aq/V_aq are held explicit over the Newton loop (MRST does the
            # same): read them from the numeric state, differentiate only the
            # reservoir side through the AD state `st`.
            keys = getattr(self.aquifer, "state_arg_keys",
                           ("aquifer_p", "aquifer_V"))
            eqs = self.aquifer.add_contribution(
                self, eqs, st, state[keys[0]], state[keys[1]], dt)

        q_s = [st[q] for q in ["qWs", "qOs", "qGs"][:nph]]
        simple = self._simple_well_indices()
        ms = self._ms_well_indices()
        well_eqs = []
        ctrl_parts = []
        if simple:
            sfac = FacilityModel([self.wells[i] for i in simple],
                                 disgas=self.disgas, vapoil=self.vapoil)
            q_simple = [q[simple] for q in q_s]
            bhp_simple = st["bhp"][simple]
            cq_s, we, ctrl = sfac.well_equations(
                self, st, q_simple, bhp_simple, state0=st0
            )
            for local, wi in enumerate(simple):
                for ph, eq in enumerate(eqs):
                    q = self._basis_scale()[ph] * cq_s[local][ph]
                    eq[self.wells[wi].cells] = eq[self.wells[wi].cells] - q
            well_eqs.extend(we)
            ctrl_parts.append(ctrl)
            self.facility.last_flux = sfac.last_flux
        else:
            well_eqs.extend([np.zeros(0) for _ in range(nph)])

        ms_eqs, ms_ctrls, extra = self._multisegment_equations(state0, st, q_s, dt, eqs)
        well_eqs = [vertcat_adi(well_eqs[i], ms_eqs[i]) for i in range(nph)]
        ctrl_parts.extend(ms_ctrls)
        ctrl = vertcat_adi(*ctrl_parts) if ctrl_parts else np.zeros(0)
        return eqs + well_eqs + [ctrl] + extra

    def _multisegment_equations(self, state0: State, st: State, q_s, dt, eqs):
        nph = self.n_phases
        q_names = ["qWs", "qOs", "qGs"][:nph]
        ms_well_eqs = [[] for _ in range(nph)]
        ms_ctrls = []
        extra = []
        if not self._ms_well_indices():
            return [np.zeros(0) for _ in range(nph)], [], []
        mob = self.get_props(st, "Mobility")
        b = self.get_props(st, "ShrinkageFactors")
        for wi, p_slice, r_slice, v_slice in self._ms_slices():
            w = self.wells[wi]
            res = assemble_multisegment_equations(
                w, self,
                q_s=[q[wi:wi + 1] for q in q_s],
                bhp=st["bhp"][wi:wi + 1],
                node_pressure=st["ms_pN"][p_slice],
                node_comp=[st[n][r_slice] for n in ["ms_rW", "ms_rO", "ms_rG"][:nph]],
                segment_flux=st["ms_vmix"][v_slice],
                res_props={"pressure": st["pressure"], "mob": mob, "b": b,
                           "rs": st.get("rs")},
                dt=dt,
                previous_node_comp=[
                    state0[n][r_slice]
                    for n in ["ms_rW", "ms_rO", "ms_rG"][:nph]
                ] if all(n in state0 for n in ["ms_rW", "ms_rO", "ms_rG"][:nph]) else None,
            )
            for ph in range(nph):
                q = self._basis_scale()[ph] * res.connection_surface_rates[ph]
                eqs[ph][w.cells] = eqs[ph][w.cells] - q
                ms_well_eqs[ph].append(res.well_rate_equations[ph])
                extra.append(res.node_mass_equations[ph])
            extra.append(res.segment_pressure_equations)
            extra.append(res.composition_equation)
            ms_ctrls.append(FacilityModel._control_equation(
                w, [q[wi:wi + 1] for q in q_s], st["bhp"][wi:wi + 1], nph
            ))
        for ph in range(nph):
            ms_well_eqs[ph] = vertcat_adi(*ms_well_eqs[ph]) if ms_well_eqs[ph] else np.zeros(0)
        return ms_well_eqs, ms_ctrls, extra

    def after_convergence(self, state: State, state0: State,
                          dt: float) -> State:
        """Explicit post-step updates (MRST's `updateAfterConvergence`).

        The analytic aquifer advances its own pressure and volume here, from
        the flux it delivered over the step just taken. `state0` holds the
        aquifer state at the start of the step, so a re-solve after a timestep
        cut restarts from the right values.

        The wells hand on their **perforation fluxes** (MRST's `wellSol.flux`),
        which is what the *next* step weighs its wellbore column with. Freezing
        them at the previous step is not an approximation of convenience: making
        the column an implicit function of the unknowns would couple every
        perforation of a well to every other and fill in the Jacobian, and MRST
        declines to do it for the same reason.
        """
        if self.aquifer is not None:
            keys = getattr(self.aquifer, "state_arg_keys",
                           ("aquifer_p", "aquifer_V"))
            a_new, b_new = self.aquifer.after_convergence(
                self, state, state0[keys[0]], state0[keys[1]], dt)
            state[keys[0]] = a_new
            state[keys[1]] = b_new
        if self.facility.last_flux is not None:
            state["well_flux"] = [q.copy() for q in self.facility.last_flux]
        # The explicit and AIM builders estimate the next step from the flux
        # just accepted. Keeping it on the state also mirrors MRST's simulator
        # state payload and makes ministep reporting reproducible.
        ad_state = self.init_state_ad(state)
        state["flux"] = np.column_stack([
            np.asarray(value(v), dtype=float)
            for v in self.get_props(ad_state, "PhaseFlux")
        ])
        return state

    def _convergence_scaling(self, nph: int):
        r"""`(B_avg, pv)` for the CNV/MB norms - MRST's `getConvergenceValuesCNV`.

        Two things here were missing, and both make the residual mean something
        other than what MRST means by it.

        **`B_avg = mean(1/b)`.** MRST weights each phase's residual by its
        average *formation volume factor*, which converts a surface-volume
        imbalance back into the reservoir volume it actually occupies. That is
        the point of the criterion: a cubic metre of surface gas is not a cubic
        metre of anything down here. Leave it out and the phases are compared on
        incommensurable scales - on SPE 1, `b_g ~ 262`, so an unweighted gas
        residual is **262x** larger than MRST's for the identical error, and the
        tolerance is silently ~262x stricter on gas than on water.

        (This port assembles its equations in *surface volumes*, as MRST's own
        classic `equationsBlackOil.m` does. MRST's newer state-function path
        uses masses instead - and then divides the surface density straight back
        out here, `eq = eq./rhoS`, precisely so the criterion is computed on the
        surface-volume residual either way. The two conventions are a constant
        row scaling apart, which leaves the Newton step invariant; the
        *criterion* is where the difference would have shown, and this is it.)

        **The pore volume is the current one**, not the reference: MRST reads
        `PoreVolume` from the state, so it carries the rock compressibility
        multiplier. A small effect, and free to get right.
        """
        st = getattr(self, "_conv_state", None)
        if st is None:               # a caller that never assembled equations
            return np.ones(nph), self.pv_ref
        b = self.get_props(st, "ShrinkageFactors")
        pv = np.asarray(value(self.get_props(st, "PoreVolume")), dtype=float)
        b_avg = np.array([float(np.mean(1.0 / np.asarray(value(b[i]), float)))
                          for i in range(nph)])
        return b_avg, pv

    def get_convergence(self, eqs, dt: float) -> Tuple[bool, Dict[str, float]]:
        """MRST's CNV / material-balance / well residual criteria.

        The numeric kernels live in `ad_core.utils.convergence`; this method is
        only the orchestration - which residuals are which, and the one basis
        unscaling that is specific to how *this* model assembled its equations.
        """
        nph = self.n_phases
        res = [value(e) for e in eqs]
        cell_eqs, well_eqs, ctrl = res[:nph], res[nph:2 * nph], res[2 * nph]
        extra_eqs = res[2 * nph + 1:]

        b_avg, pv = self._convergence_scaling(nph)
        # MRST's `isMass` branch: whatever currency the equations were assembled
        # in, the criterion is defined on the *surface-volume* residual, so take
        # the surface density straight back out. This is the one place the two
        # bases have to be told apart - and it is why they converge identically
        # rather than merely similarly.
        scale = self._basis_scale()
        cell_eqs = [cell_eqs[i] / scale[i] for i in range(len(cell_eqs))]

        cnv, mb = cnv_mb_convergence(cell_eqs, b_avg, pv, dt)
        well, ctrl_res = well_control_residual(well_eqs, ctrl, self.wells)
        extra_res = max_extra_residual(extra_eqs)

        converged = (cnv <= self.tol_cnv and mb <= self.tol_mb
                     and well <= self.tol_well and ctrl_res <= self.tol_well
                     and extra_res <= self.tol_well
                     # A well that just switched control is solving a
                     # different problem from the one this residual measures.
                     and self.facility.within_limits)
        report = {"CNV": cnv, "MB": mb, "well": well, "control": ctrl_res}
        if extra_eqs:
            report["extra"] = extra_res
        # `within_limits` is part of the decision above but is not a residual,
        # so it was absent from the report - and a step failing only on it
        # reported every residual at ~1e-11 alongside "did not converge", which
        # reads as a solver that cannot count. Report it when it is the reason.
        if not self.facility.within_limits:
            report["within_limits"] = 0.0
        return converged, report

    # -- Newton update -------------------------------------------------------
    def update_state(self, st: State, increment: np.ndarray) -> State:
        return update_state_black_oil_generic(self, st, increment)

    def _flash(self, st: State, prev: State) -> None:
        """MRST's `computeFlashBlackOil`: which phases exist, and the capping.

        The flash is what makes the variable switch legal. Newton is allowed to
        drive a saturation negative or push `rs` past its bubble point; this is
        where that is interpreted as *a phase appearing or disappearing* rather
        than as nonsense, and the status is set for the next iteration.

        The gas and oil halves are mirror images, and the second one only runs
        with vaporised oil - without it the oil phase never disappears, which
        is precisely the assumption `vapoil` removes.
        """
        compute_flash_black_oil_ad(st, prev, self)

    # -- MRST `updateAfterConvergence` ---------------------------------------

    def update_after_convergence(self, state0, state, dt, forces=None):
        """Store the converged fluxes on the state.

        The fluxes are not a by-product: streamlines, flow diagnostics and any
        post-processing read `state.flux`, and they cannot be recovered from
        pressure and saturation afterwards without redoing the discretization.
        This is the one point they are available for free, because the
        converged state functions still hold them.

        Two details are MRST's and both matter:

        * the array is sized by **all** faces and written only at the
          *internal* connections. A boundary face keeps its zero, because the
          interior flux operator has no value there and writing one would put
          an interior number on an exterior face;
        * boundary fluxes are added separately and only when the step actually
          had boundary conditions. They carry their own sign convention, which
          `store_boundary_fluxes` resolves.

        Off unless `output_fluxes` is set, as in MRST - it costs an extra
        property evaluation per step.
        """
        base = getattr(super(), "update_after_convergence", None)
        state, report = (base(state0, state, dt, forces) if base is not None
                         else (state, {}))

        if not getattr(self, "output_fluxes",
                       getattr(self, "outputFluxes", False)):
            return state, report

        flow_state = self.FlowDiscretization.build_flow_state(
            self, state, state0, dt)
        flux = self.get_props(flow_state, "PhaseFlux")

        internal = self._internal_connections()
        stored = np.zeros((int(self.G.faces.num), len(flux)))
        for i, f in enumerate(flux):
            stored[internal, i] = np.asarray(value(f), dtype=float)
        state["flux"] = stored

        bc = (forces.get("bc") if isinstance(forces, dict)
              else getattr(forces, "bc", None))
        if bc is not None:
            state = self._store_boundary_fluxes_after_convergence(state, forces)
        return state, report

    updateAfterConvergence = update_after_convergence

    def _internal_connections(self):
        internal = getattr(self.operators, "internalConn", None)
        if internal is None and isinstance(self.operators, dict):
            internal = self.operators.get("internalConn")
        if internal is None:
            return np.all(np.asarray(self.G.faces.neighbors) >= 0, axis=1)
        return np.asarray(internal)

    def _store_boundary_fluxes_after_convergence(self, state, forces):
        """Boundary fluxes, expanded to the fixed `[W, O, G]` triple.

        `boundary_reservoir_fluxes` returns one entry per *active* phase;
        `store_boundary_fluxes` indexes the triple. Passing the active list
        straight through would put the gas flux in the oil slot on a
        two-phase model.
        """
        from pymrst.modules.ad_core.utils.boundary_conditions import (
            boundary_reservoir_fluxes,
        )
        bc = (forces.get("bc") if isinstance(forces, dict) else forces.bc)
        fluxes = boundary_reservoir_fluxes(self, state, bc)
        presence = self.get_phase_presence()
        triple = [None, None, None]
        for slot, f in zip([i for i, on in enumerate(presence) if on], fluxes):
            triple[slot] = f
        return self.store_boundary_fluxes(state, *triple, forces=forces)


class GenericBlackOilModel(AdOoGenericReservoirModel, ThreePhaseBlackOilModel):
    """MRST `GenericBlackOilModel`: black-oil as component mass conservation.

    The flow discretization writes `(M_i - M_i^0)/dt` and the component
    fluxes, the facility adds the well sources and its own equations, and
    `ThreePhaseBlackOilModel` supplies the variables and their update. MRST
    lists `ThreePhaseBlackOilModel` first; the marker comes first here so its
    `isGenericReservoirModel = True` is found before `ReservoirModel`'s False.
    """

    def __init__(self, G: Any, rock: Any, fluid: Any, **options: Any):
        super().__init__(G, rock, fluid, **options)
        self.OutputStateFunctions = ["ComponentTotalMass", "Density"]

    def get_equations(self, state0: Any, state: Any, dt: float,
                      driving_forces: Any, **options: Any):
        return AdOoPhysicalModel.get_equations(self, state0, state, dt,
                                               driving_forces, **options)

    def get_model_equations(self, state0: Any, state: Any, dt: float,
                            driving_forces: Any):
        eqs, flux, names, types = self.FlowDiscretization.component_conservation_equations(
            self, state, state0, dt)
        if self.FacilityModel is not None:
            src = self.FacilityModel.get_component_sources(state)
        if _has_force(driving_forces, "bc") or _has_force(driving_forces, "src"):
            pressures, sat, mob, rho, rs, rv = self.get_props(
                state, "PhasePressures", "s", "Mobility", "Density", "Rs", "Rv")
            dissolved = self.get_dissolution_matrix(rs, rv)
            # MRST keeps only the equations; the state it stores fluxes in is
            # a copy that is discarded.
            eqs, _, _ = self.add_boundary_conditions_and_sources(
                eqs, names, types, copy_state(state), pressures, sat, mob, rho,
                dissolved, [], driving_forces)
        if self.AquiferModel is not None:
            eqs = add_aquifers_contribution(self.AquiferModel, eqs, names, state, dt)
        if self.FacilityModel is not None:
            eqs = self.insert_sources(eqs, src)
        eqs = [self.operators.AccDiv(eq, f) for eq, f in zip(eqs, flux)]
        if self.FacilityModel is not None:
            weqs, wnames, wtypes, state = self.FacilityModel.get_model_equations(
                state0, state, dt, driving_forces)
            eqs, names, types = eqs + weqs, names + wnames, types + wtypes
        return eqs, names, types, state

    def get_component_names(self) -> List[str]:
        return [c.name for c in self.Components]

    def update_state(self, state: Any, problem: Any, dx, driving_forces: Any):
        state, report = super().update_state(state, problem, dx, driving_forces)
        if self.FacilityModel is not None:
            state = self.FacilityModel.apply_well_limits(state)
        return state, report

    def validate_model(self, *args: Any) -> "GenericBlackOilModel":
        """Default components (one per phase) before the base validation."""
        assert self.FacilityModel is None or isinstance(
            self.FacilityModel, AdOoGenericFacilityModel), (
            "Generic model can only be used with GenericFacilityModel.")
        if not self.Components:
            disgas, vapoil = self.disgas, self.vapoil
            components = []
            for ph, letter in enumerate(self.get_phase_names()):
                if letter == "W":
                    c = ImmiscibleComponent("water", ph)
                elif letter == "O":
                    c = (OilComponent("oil", ph, disgas, vapoil) if disgas or vapoil
                         else ImmiscibleComponent("oil", ph))
                elif letter == "G":
                    c = (GasComponent("gas", ph, disgas, vapoil) if disgas or vapoil
                         else ImmiscibleComponent("gas", ph))
                else:
                    raise ValueError("Unknown phase")
                components.append(c)
            self.Components = components
        return super().validate_model(*args)

    def update_after_convergence(self, state0: Any, state: Any, dt: float,
                                 driving_forces: Any):
        """Store the converged interface fluxes (and boundary fluxes)."""
        state, report = AdOoReservoirModel.update_after_convergence(
            self, state0, state, dt, driving_forces)
        if self.outputFluxes:
            state_flow = self.FlowDiscretization.build_flow_state(self, state,
                                                                  state0, dt)
            f = self.get_prop(state_flow, "PhaseFlux")
            flux = np.zeros((self.G.faces.num, len(f)))
            flux[self.operators.internalConn, :] = np.column_stack(
                [np.asarray(value(x), dtype=float) for x in f])
            state["flux"] = flux
            if _has_force(driving_forces, "bc"):
                p, s, mob, r, b = self.get_props(state, "PhasePressures", "s",
                                                 "Mobility", "Density",
                                                 "ShrinkageFactors")
                sat = expand_matrix_to_cell(s)
                rho = expand_matrix_to_cell(r)
                bc = driving_forces["bc"] if isinstance(driving_forces, dict) \
                    else driving_forces.bc
                *_, f_res = get_boundary_condition_fluxes_ad(self, p, sat, mob,
                                                             rho, b, bc)
                f_wog: List[Any] = [None, None, None]
                for slot, q in zip(np.flatnonzero(self.get_active_phases()), f_res):
                    f_wog[slot] = q
                state = self.store_boundary_fluxes(state, *f_wog, driving_forces)
        return state, report

    getEquations = get_equations
    getModelEquations = get_model_equations
    getComponentNames = get_component_names
    updateState = update_state
    validateModel = validate_model
    updateAfterConvergence = update_after_convergence


def _has_force(forces: Any, name: str) -> bool:
    """MRST `~isempty(drivingForces.(name))`."""
    item = forces.get(name) if isinstance(forces, dict) else getattr(forces, name, None)
    if item is None:
        return False
    if isinstance(item, (list, tuple)):
        return len(item) > 0
    return True


def blackoil_status(sw, so, sg, disgas: bool = True,
                    vapoil: bool = False) -> np.ndarray:
    """Which phases a cell starts with (MRST's `getCellStatusVO`, no `status`).

    An initial state says what saturations a cell has, not which variable the
    solver should carry there - that follows from the saturations, and MRST
    derives it silently the first time it is asked. Deriving it explicitly is
    better: an initial state whose `status` disagrees with its saturations is a
    contradiction the solver will resolve by quietly discarding one of them.

    Water-only cells are declared to have *both* hydrocarbon phases: there is
    no free oil or gas to speak of, so nothing constrains `rs` or `rv`, and
    pinning them to their saturation curves is the only well-posed choice.
    """
    st1, st2, st3 = get_cell_status_vo(so, sw, sg, disgas=disgas, vapoil=vapoil)
    # MRST's `oilPresent + 2*gasPresent`, recovered from the masks: the
    # gating in `getCellStatusVO` only ever zeroes a status the saturations
    # could not have produced, so the round trip is exact.
    return 1*np.asarray(st1) + 2*np.asarray(st2) + 3*np.asarray(st3)


def initial_state_from_deck(model: GenericBlackOilModelAD, deck) -> State:
    """Create a black-oil state from imported deck pressure/composition."""
    sw, sg = deck["swat"], deck["sgas"]
    nw = len(model.wells)
    qws, qos, qgs = np.zeros(nw), np.zeros(nw), np.zeros(nw)
    bhp = np.zeros(nw)
    p_mean = float(deck["pressure"].mean())
    for k, w in enumerate(model.wells):
        if w.type in ("orat", "grat", "wrat", "lrat"):
            if w.type == "orat":
                qos[k] = w.val
            elif w.type == "grat":
                qgs[k] = w.val
            elif w.type == "wrat":
                qws[k] = w.val
            bhp[k] = p_mean - 10e5
        elif w.type == "rate":
            qgs[k] = w.val * w.compi[2]
            qws[k] = w.val * w.compi[0]
            bhp[k] = p_mean + 10e5
        else:
            bhp[k] = w.val
    state = State(
        pressure=deck["pressure"].copy(),
        sW=sw.copy(), sO=1.0 - sw - sg, sG=sg.copy(),
        rs=deck["rs"].copy(),
        status=blackoil_status(sw, 1.0 - sw - sg, sg,
                               disgas=model.disgas, vapoil=model.vapoil),
        qWs=qws, qOs=qos, qGs=qgs, bhp=bhp,
    )
    if model.vapoil:
        state["rv"] = np.asarray(deck.get("rv", np.zeros_like(sw)), float).copy()
        # Deck RV values are commonly rounded at the dew-point envelope (SPE3
        # is a canonical example).  Seed those cells on the two-phase side and
        # cap the rounded value, instead of forcing the first Newton iteration
        # to discover hundreds of simultaneous phase appearances.
        rv_sat = np.asarray(value(model.raw_fluid.rv_sat(state["pressure"])), float)
        at_dew = state["rv"] >= rv_sat * (1.0 - ETOL)
        state["rv"][at_dew] = rv_sat[at_dew]
        state["status"][at_dew] = 3
    return state


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.gridprocessing import (
        cart_grid,
        compute_geometry,
        tensor_grid,
    )
    from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
        ThreePhaseBlackOilModelAD,
        initial_state_spe1,
    )
    from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
        TwoPhaseOilWaterModelAD,
    )
    from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
    from pymrst.modules.deckformat import read_spe1_deck
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _ow_case():
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
        return g, rock, fluid, w

    def _ow_initial(g, wells, p0=200 * BARSA, sw0=0.2) -> State:
        nw = len(wells)
        return State(
            pressure=np.full(g.cells.num, float(p0)),
            sW=np.full(g.cells.num, float(sw0)),
            qWs=np.array([w.val * w.compi[0] if w.type == "rate" else 0.0
                          for w in wells]),
            qOs=np.array([w.val * w.compi[1] if w.type == "rate" else 0.0
                          for w in wells]),
            bhp=np.array([w.val if w.type == "bhp" else float(p0)
                          for w in wells]),
        )

    def test_graph_model_converges():
        """The graph-assembled model takes a step like any other model."""
        g, rock, fluid, wells = _ow_case()
        m = GenericBlackOilModelAD(g, rock, fluid, wells)
        st, rep = NonLinearSolver().solve_ministep(
            m, _ow_initial(g, wells), _ow_initial(g, wells), 30 * DAY)
        assert rep.converged, rep.residuals[-1]
        assert np.isclose(st["qWs"][0] + st["qOs"][0], 1e-3, rtol=1e-9)
        assert np.isclose(st["bhp"][1], 150 * BARSA)
        print(f"ok test_graph_model_converges ({rep.iterations} its)")

    def test_matches_hand_rolled_oil_water():
        """THE PROOF: the graph reproduces the hand-rolled model exactly.

        `oil_water_model` is validated against MATLAB ad-blackoil to ~1e-14
        (Phase 5a). If the graph-assembled model reproduces it to round-off
        over several steps, the graph inherits that validation.

        Both models are built with the **tight** tolerances rather than the
        MRST defaults the constructors now carry. This is an
        implementation-equivalence claim: at `tol_cnv = 1e-3` the two stop at
        whatever point each happens to reach first and agree only to 7.5e-7,
        which compares two approximations rather than two implementations.
        Converging both to round-off is what makes "the same code path" a
        testable statement - the same reason the adjoint case keeps them.
        """
        g, rock, fluid, wells = _ow_case()
        tight = dict(tol_cnv=1e-8, tol_mb=1e-12, tol_well=1e-10)
        graph = GenericBlackOilModelAD(g, rock, fluid, wells, **tight)
        hand = TwoPhaseOilWaterModelAD(g, rock, fluid, wells, **tight)

        st_g = _ow_initial(g, wells)
        st_h = hand.initial_state(200 * BARSA, 0.2)
        solver = NonLinearSolver()

        worst = 0.0
        for _ in range(4):
            st_g, rep = solver.solve_ministep(graph, st_g, st_g.copy(),
                                              30 * DAY)
            st_h, ok, _ = hand.step(st_h, 30 * DAY)
            assert rep.converged and ok
            for key, ref in (("pressure", st_h.pressure), ("sW", st_h.sW),
                             ("qWs", st_h.qWs), ("qOs", st_h.qOs),
                             ("bhp", st_h.bhp)):
                scale = max(np.abs(ref).max(), 1e-30)
                worst = max(worst, np.abs(st_g[key] - ref).max() / scale)
        assert worst < 1e-11, f"graph and hand-rolled diverged by {worst:.2e}"
        print(f"ok test_matches_hand_rolled_oil_water (max rel diff {worst:.1e})")

    def test_every_property_evaluated_once_per_state():
        """The graph caches: each property is evaluated once per state, not
        once per place it is used.

        Mobility is wanted in three places while assembling a residual - the
        face mobility, the well inflow, and the wellbore density - across two
        states (the current one, and the previous one the connection pressure
        drops are frozen at). So the count that proves the cache works is
        exactly *one per state*, not one overall.
        """
        g, rock, fluid, wells = _ow_case()
        m = GenericBlackOilModelAD(g, rock, fluid, wells)
        st = _ow_initial(g, wells)

        seen = []
        mob = m.FlowPropertyFunctions.get_state_function("Mobility")
        inner = mob.evaluate_on_domain

        def counted(model, state):
            seen.append(id(state))
            return inner(model, state)

        mob.evaluate_on_domain = counted
        m.get_equations(st, st.copy(), 30 * DAY)

        assert len(seen) == len(set(seen)), (
            f"Mobility recomputed on a state it was already evaluated on: "
            f"{len(seen)} evaluations over {len(set(seen))} states"
        )
        assert len(seen) <= 2, f"more states than expected: {len(seen)}"
        print(f"ok test_every_property_evaluated_once_per_state "
              f"({len(seen)} evaluations, {len(set(seen))} distinct states)")

    def test_solver_report_is_populated():
        """A converged step reports its residual history."""
        g, rock, fluid, wells = _ow_case()
        m = GenericBlackOilModelAD(g, rock, fluid, wells)
        _, rep = NonLinearSolver().solve_ministep(
            m, _ow_initial(g, wells), _ow_initial(g, wells), 30 * DAY)
        assert rep.converged
        assert set(rep.residuals[0]) == {"CNV", "MB", "well", "control"}
        assert rep.residuals[-1]["CNV"] < rep.residuals[0]["CNV"]
        print("ok test_solver_report_is_populated")

    def test_matches_hand_rolled_spe1():
        """The same proof for three-phase live oil: SPE 1 through the graph.

        `blackoil_model` is validated against MATLAB ad-blackoil to ~1e-10
        (Phase 5b), variable switching and all. The graph must reproduce it
        - including the flash, which is the part most likely to drift.
        """
        deck_path = Path(__file__).parents[5] / (
            "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
        )
        if not deck_path.exists():
            deck_path = Path(__file__).parents[4] / (
                "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
            )
        if not deck_path.exists():
            print("skip test_matches_hand_rolled_spe1 (SPE1 deck not found)")
            return

        deck = read_spe1_deck(deck_path)
        g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
        rock = make_rock(g, deck["perm"], deck["poro"][:, None])
        fluid = init_deck_blackoil_fluid(deck)
        wells = None
        for name in ("INJECTOR", "PRODUCER"):
            wd = deck["wells"][name]
            wells = add_well(
                wells, g, rock, wd["cells"], well_type=wd["type"],
                val=wd["val"], radius=wd["radius"], compi=wd["compi"],
                sign=wd["sign"], name=name, ref_depth=wd["ref_depth"],
                gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
            )

        # Tight tolerances on both, for the same reason as the oil-water
        # proof above: this compares two *implementations*, and at the MRST
        # defaults the constructors now carry they would agree only to the
        # tolerance rather than to round-off.
        tight = dict(tol_cnv=1e-8, tol_mb=1e-12, tol_well=1e-8)
        graph = GenericBlackOilModelAD(g, rock, fluid, wells, disgas=True,
                                     gravity=NORM_GRAVITY, **tight)
        hand = ThreePhaseBlackOilModelAD(g, rock, fluid, wells, **tight)

        st_g = initial_state_from_deck(graph, deck)
        st_h = initial_state_spe1(hand, deck)
        solver = NonLinearSolver()

        worst = 0.0
        for i, dt in enumerate(deck["tsteps"][:5]):
            st_g, rep = solver.solve_ministep(graph, st_g, st_g.copy(), dt)
            st_h, ok, _ = hand.step(st_h, dt)
            assert rep.converged and ok, f"step {i} failed"
            for key, ref in (("pressure", st_h.pressure), ("sW", st_h.s[:, 0]),
                             ("sG", st_h.s[:, 2]), ("rs", st_h.rs),
                             ("qOs", st_h.qOs), ("qGs", st_h.qGs),
                             ("bhp", st_h.bhp)):
                scale = max(np.abs(ref).max(), 1e-30)
                worst = max(worst, np.abs(st_g[key] - ref).max() / scale)
            assert np.array_equal(st_g["status"], st_h.status), \
                f"flash disagreed at step {i}"
        assert worst < 1e-11, f"graph and hand-rolled diverged by {worst:.2e}"
        print(f"ok test_matches_hand_rolled_spe1 (max rel diff {worst:.1e})")

    def test_injector_switches_from_rate_to_bhp():
        """A rate-controlled injector that would need too much pressure must
        fall back to holding its pressure limit.

        This is what a real schedule says - "inject 5 l/s, but never above
        250 bar" - and without switching the well simply demands whatever
        bottom-hole pressure the rate needs, however unphysical.
        """
        g, rock, fluid, _ = _ow_case()
        p_lim = 250.0 * BARSA
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5e-3,                       # far more than 250 bar buys
                     compi=np.array([1.0, 0.0]), sign=1,
                     lims={"bhp": p_lim})
        w = add_well(w, g, rock, np.array([63]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)

        m = GenericBlackOilModelAD(g, rock, fluid, w)
        st = _ow_initial(g, w)
        st, rep = NonLinearSolver().solve_ministep(m, st, st.copy(), 30 * DAY)

        assert rep.converged
        assert m.wells[0].type == "bhp", "the injector should have switched"
        assert np.isclose(st["bhp"][0], p_lim), "and must now hold its limit"
        assert st["qWs"][0] < 5e-3, "so it injects less than it was asked to"
        assert st["qWs"][0] > 0
        print(f"ok test_injector_switches_from_rate_to_bhp "
              f"(rate {st['qWs'][0]:.2e} m3/s at its {p_lim / BARSA:.0f} bar limit)")

    def test_well_within_its_limits_does_not_switch():
        """A limit that is never reached must leave the well alone."""
        g, rock, fluid, _ = _ow_case()
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-3, compi=np.array([1.0, 0.0]), sign=1,
                     lims={"bhp": 1000.0 * BARSA})     # unreachably high
        w = add_well(w, g, rock, np.array([63]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)

        m = GenericBlackOilModelAD(g, rock, fluid, w)
        st = _ow_initial(g, w)
        st, rep = NonLinearSolver().solve_ministep(m, st, st.copy(), 30 * DAY)

        assert rep.converged and m.wells[0].type == "rate"
        assert np.isclose(st["qWs"][0] + st["qOs"][0], 1e-3, rtol=1e-9)
        print("ok test_well_within_its_limits_does_not_switch")

    def test_producer_switches_on_liquid_rate():
        """A BHP producer that pulls too much liquid switches to lrat.

        Producer rates are negative, so it violates the limit by going
        *below* it - the comparison that looks upside down but is not.
        """
        g, rock, fluid, _ = _ow_case()
        l_lim = -5e-4                       # m3/s, negative = production
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=2e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([63]), well_type="bhp",
                     val=100 * BARSA,        # a hard pull
                     compi=np.array([0.0, 1.0]), sign=-1,
                     lims={"lrat": l_lim})

        m = GenericBlackOilModelAD(g, rock, fluid, w)
        st = _ow_initial(g, w)
        st, rep = NonLinearSolver().solve_ministep(m, st, st.copy(), 30 * DAY)

        assert rep.converged
        assert m.wells[1].type == "lrat", "the producer should have switched"
        liquid = st["qWs"][1] + st["qOs"][1]
        assert np.isclose(liquid, l_lim, rtol=1e-8), (
            "and must now hold exactly its liquid limit"
        )
        assert st["bhp"][1] > 100 * BARSA, "backing off means a higher bhp"
        print(f"ok test_producer_switches_on_liquid_rate "
              f"(bhp rose to {st['bhp'][1] / BARSA:.0f} bar to hold "
              f"{l_lim:.1e} m3/s)")

    test_graph_model_converges()
    test_matches_hand_rolled_oil_water()
    test_matches_hand_rolled_spe1()
    test_every_property_evaluated_once_per_state()
    test_solver_report_is_populated()
    test_injector_switches_from_rate_to_bhp()
    test_well_within_its_limits_does_not_switch()

    def _wet_gas_model(nc=6, ngc=2):
        """A tiny wet-gas column: a gas cap with no oil phase over an oil leg."""
        from pymrst.modules.ad_props.props.init_deck_adi_fluid import BlackOilFluid
        from pymrst.utils.adi import ad_max

        p_ref = 300 * BARSA
        rs_sat = lambda p: 150.0 * (p / p_ref)
        rv_sat = lambda p: 8.0e-4 * (p / p_ref)
        blend = lambda f, a, b: (np.asarray(f, float) * a
                                 + (1.0 - np.asarray(f, float)) * b)
        swcon = 0.1
        fluid = BlackOilFluid(
            rhoS=np.array([1000.0, 800.0, 1.2]),
            b_w=lambda p: 1.0 + 4e-5 / BARSA * (p - p_ref),
            mu_w=lambda p: 0.4 * CENTI * POISE + 0.0 * p,
            b_o=lambda p, rs, f: (1.0 + 1.2e-4 / BARSA * (p - p_ref))
            / (1.0 + 1e-3 * blend(f, rs_sat(p), rs)),
            mu_o=lambda p, rs, f: 2.0 * CENTI * POISE
            / (1.0 + 2e-3 * blend(f, rs_sat(p), rs)),
            rs_sat=rs_sat,
            b_g=lambda p, rv, f: (1.0 + 3e-3 / BARSA * (p - p_ref))
            * (1.0 + 50.0 * blend(f, rv_sat(p), rv)),
            mu_g=lambda p, rv, f: 0.02 * CENTI * POISE
            * (1.0 + 100.0 * blend(f, rv_sat(p), rv)),
            rv_sat=rv_sat, vapoil=True,
            kr_w=lambda sw: ad_max((sw - swcon) / (1.0 - swcon), 0.0) ** 2,
            kr_g=lambda sg: ad_max(sg, 0.0) ** 2,
            kr_ow=lambda so: ad_max(so, 0.0) ** 2,
            kr_og=lambda so: ad_max(so, 0.0) ** 2,
            pv_mult=lambda p: 1.0 + 0.0 * p,
            swcon=swcon,
        )
        g = compute_geometry(cart_grid([1, 1, nc], [10.0, 10.0, 50.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        model = GenericBlackOilModelAD(g, rock, fluid, [], disgas=True,
                                     vapoil=True, gravity=0.0)

        p0 = np.full(nc, p_ref)
        cap = np.arange(nc) < ngc
        sw = np.full(nc, swcon)
        so = np.where(cap, 0.0, 1.0 - swcon)
        sg = np.where(cap, 1.0 - swcon, 0.0)
        st = State(
            pressure=p0, sW=sw, sO=so, sG=sg,
            rs=np.where(cap, 0.0, 0.6 * rs_sat(p0)),
            rv=np.where(cap, 0.6 * rv_sat(p0), 0.0),
            status=blackoil_status(sw, so, sg, disgas=True, vapoil=True),
            qWs=np.zeros(0), qOs=np.zeros(0), qGs=np.zeros(0),
            bhp=np.zeros(0),
        )
        return model, st

    def test_the_variable_switch_means_three_different_things():
        """`x` is `rs`, `rv` or `sG` depending on the cell, and every other
        hydrocarbon quantity is *derived* from it. If that reconstruction is
        wrong the model still runs - it just conserves the wrong thing."""
        model, st = _wet_gas_model()
        assert set(st["status"]) == {1, 2}, st["status"]

        ad = model.init_state_ad(st)
        p = value(ad["pressure"])
        cap = st["status"] == 2
        rs, rv, sg = (value(ad[k]) for k in ("rs", "rv", "sG"))

        # status 2 (gas cap, no oil phase): rv is free, rs sits on its curve,
        # and *all* the hydrocarbon is gas - so sG is 1 - sW, not a variable.
        assert np.allclose(rv[cap], st["rv"][cap])
        assert np.allclose(rs[cap], model.raw_fluid.rs_sat(p)[cap])
        assert np.allclose(sg[cap], 1.0 - st["sW"][cap])

        # status 1 (oil leg, no free gas): the mirror image.
        assert np.allclose(rs[~cap], st["rs"][~cap])
        assert np.allclose(rv[~cap], model.raw_fluid.rv_sat(p)[~cap])
        assert np.allclose(sg[~cap], 0.0)
        print("ok test_the_variable_switch_means_three_different_things")

    def test_a_closed_wet_gas_reservoir_does_not_move():
        """The test that catches a mis-stated conservation law.

        Seal the box - no wells, no boundary, no sources - and nothing may
        happen. It is a sharp test precisely because `rv` puts oil in the gas
        phase: get the oil component's density wrong in *either* phase and the
        oil equation stops balancing, mass appears from nowhere, and a state
        that should be inert starts drifting.
        """
        model, st = _wet_gas_model()
        solver = NonLinearSolver()
        s = st
        for _ in range(5):
            s, rep = solver.solve_ministep(model, s, s.copy(), 100 * DAY)
            assert rep.converged

        moved = max(
            np.abs(np.asarray(s[k], float)
                   - np.asarray(st[k], float)).max()
            / max(np.abs(np.asarray(st[k], float)).max(), 1e-300)
            for k in ("pressure", "sW", "sO", "sG", "rs", "rv")
        )
        assert moved < 1e-12, f"a sealed reservoir moved by {moved:.2e}"
        assert np.array_equal(s["status"], st["status"])
        print(f"ok test_a_closed_wet_gas_reservoir_does_not_move "
              f"(500 days, moved {moved:.1e})")

    def test_vapoil_is_independent_of_disgas():
        """SPE3 is wet gas with dead oil: VAPOIL must work without DISGAS."""
        model, st = _wet_gas_model()
        spe3 = GenericBlackOilModelAD(model.G, model.rock, model.raw_fluid, [],
                                    disgas=False, vapoil=True)
        assert not spe3.disgas and spe3.vapoil
        assert not spe3.fluid.disgas and spe3.fluid.vapoil
        print("ok test_vapoil_is_independent_of_disgas")

    test_producer_switches_on_liquid_rate()

    def test_the_two_bases_give_the_same_answer():
        """Surface volumes or masses: the same simulation, not a similar one.

        `eq_i^mass = rhoS_i * eq_i^surfvol` exactly, so the mass basis is a
        *row scaling* of the residual system - and a row scaling leaves the
        Newton step invariant, `(DJ)^-1 (Dr) = J^-1 r`. The convergence
        criterion is defined on the surface-volume residual either way (MRST's
        `isMass` branch, mirrored in `get_convergence`), so the two do not even
        stop at different places.

        That is a strong claim and it was asserted in this port's docstrings for
        a long time before anyone checked it. This checks it: the two bases must
        agree to *round-off*, and every driving force must be in on it - so the
        case has a well, a boundary condition, a source and an aquifer, because
        each of those is a separate place where a surface-volume rate has to be
        weighed before it can enter a mass equation, and each is a separate
        place to forget.
        """
        from pymrst.modules.ad_core.models.aquifers.aquifer import (
            AquiferConnections,
            AquiferModelAD,
        )
        from pymrst.params.wells_and_bc import add_source, pside

        def build(mass_basis):
            g = compute_geometry(cart_grid([6, 1, 2], [300.0, 50.0, 20.0]))
            nc = g.cells.num
            rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
            fluid = init_simple_adi_fluid(
                mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
                n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)

            bc = pside(None, g, "xmin", 210 * BARSA,
                       sat=np.array([1.0, 0.0]))
            src = add_source(None, np.array([nc // 2]), np.array([1e-5]),
                             sat=np.array([1.0, 0.0]))

            conn = AquiferConnections(
                aquid=np.zeros(1, dtype=int), cells=np.array([nc - 1]),
                j=np.array([3e-11]), c=np.array([1e-5 / BARSA]),
                alpha=np.array([1.0]),
                depth_conn=g.cells.centroids[nc - 1:nc, -1],
                depth_aq=np.array([10.0]), c_aq=np.array([1e-5 / BARSA]))
            aq = AquiferModelAD(conn, np.array([260.0 * BARSA]), np.array([1e7]))

            w = add_well(None, g, rock, np.array([0]), well_type="bhp",
                         val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
            m = GenericBlackOilModelAD(g, rock, fluid, w, bc=bc, src=src,
                                     aquifer=aq, mass_basis=mass_basis)
            st = State(pressure=np.full(nc, 200.0 * BARSA),
                       sW=np.full(nc, 0.3),
                       qWs=np.zeros(1), qOs=np.zeros(1),
                       bhp=np.array([150.0 * BARSA]))
            aq.init_state(st)
            return m, st

        out = {}
        for basis in (False, True):
            m, st = build(basis)
            solver = NonLinearSolver()
            newtons = 0
            for _ in range(4):
                st, rep = solver.solve_ministep(m, st, st.copy(), 30 * DAY)
                assert rep.converged
                newtons += rep.iterations
            out[basis] = (st, newtons)

        (s_surf, n_surf), (s_mass, n_mass) = out[False], out[True]

        # The iteration counts must be *identical*: a row scaling cannot change
        # the Newton path, and if it did, the equivalence claim would be false.
        assert n_surf == n_mass, (
            f"the bases took different Newton paths: {n_surf} vs {n_mass}")

        worst, worst_key = 0.0, None
        for key in ("pressure", "sW", "bhp", "qWs", "qOs", "aquifer_p",
                    "aquifer_V"):
            a = np.asarray(s_surf[key], float)
            b = np.asarray(s_mass[key], float)
            err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
            if err > worst:
                worst, worst_key = err, key
        assert worst < 1e-12, f"{worst_key} differs by {worst:.2e}"
        print(f"ok test_the_two_bases_give_the_same_answer "
              f"(well + bc + source + aquifer; {n_surf} Newtons either way, "
              f"states agree to {worst:.1e})")

    def test_multisegment_well_packs_a_square_system():
        """MS wells add node pressures, node compositions, and segment fluxes
        as actual Newton variables, with matching equations."""
        from pymrst.params.wells_and_bc import MultisegmentWell, WellSegment
        from pymrst.utils.adi import combine_equations

        g = compute_geometry(cart_grid([2, 1, 1], [20.0, 10.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-5 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-4, compi=np.array([1.0, 0.0]), sign=1)
        w[0].multisegment = MultisegmentWell([
            WellSegment(0, -1, depth=0.0, diameter=0.1, volume=1.0),
            WellSegment(1, 0, length=10.0, depth=10.0, diameter=0.1,
                        roughness=1e-5, volume=1.0),
        ], np.array([1]))
        m = GenericBlackOilModelAD(g, rock, fluid, w)
        st = State(pressure=np.full(2, 200 * BARSA), sW=np.full(2, 0.2),
                   qWs=np.array([1e-4]), qOs=np.array([0.0]),
                   bhp=np.array([210 * BARSA]))
        eqs = m.get_equations(st, st.copy(), 1.0)
        eq = combine_equations(*eqs)
        assert eq.val.size == eq.jac[0].shape[0] == eq.jac[0].shape[1]
        assert "extra" in m.get_convergence(eqs, 1.0)[1]
        print(f"ok test_multisegment_well_packs_a_square_system "
              f"({eq.val.size}x{eq.jac[0].shape[1]})")

    test_the_variable_switch_means_three_different_things()
    test_a_closed_wet_gas_reservoir_does_not_move()
    test_vapoil_is_independent_of_disgas()
    test_the_two_bases_give_the_same_answer()
    test_multisegment_well_packs_a_square_system()

    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock import make_rock

    def faithful_live_oil(n_cells: int = 3) -> GenericBlackOilModel:
        _, _, fluid, _, _ = setup_spe1()
        G = compute_geometry(cart_grid([n_cells, 1, 1], [300.0, 10.0, 10.0]))
        return GenericBlackOilModel(G, make_rock(G, 1e-13, 0.3), fluid, disgas=True)

    def test_faithful_model_builds_one_component_per_phase():
        with isolated_gravity():
            live = faithful_live_oil().validate_model()
            _, _, fluid, _, _ = setup_spe1()
            G = compute_geometry(cart_grid([2, 1, 1]))
            dead = GenericBlackOilModel(G, make_rock(G, 1e-13, 0.3), fluid).validate_model()
        assert [type(c).__name__ for c in live.Components] == [
            "ImmiscibleComponent", "OilComponent", "GasComponent"]
        assert [type(c).__name__ for c in dead.Components] == ["ImmiscibleComponent"]*3
        assert live.get_component_names() == ["water", "oil", "gas"]
        assert live.isGenericReservoirModel
        assert type(live.FacilityModel).__name__ == "GenericFacilityModel"
        print("ok test_faithful_model_builds_one_component_per_phase")

    def test_faithful_model_conserves_mass_in_a_closed_box():
        with isolated_gravity():
            model = faithful_live_oil()
            state0 = {"pressure": np.array([250e5, 200e5, 150e5]),
                      "s": np.tile([0.2, 0.8, 0.0], (3, 1)),
                      "rs": np.full(3, 100.0)}
            _, states, _ = simulate_schedule_ad(state0, model,
                                                simple_schedule([10*86400.0]))
            checker = faithful_live_oil().validate_model()
            mass = [checker.get_prop(checker.init_state_function_containers(
                        checker.validate_state(dict(st))), "ComponentTotalMass")
                    for st in (state0, states[0])]
        imbalance = max(abs(np.sum(after) - np.sum(before))/np.sum(before)
                        for before, after in zip(*mass))
        assert imbalance < model.toleranceMB, imbalance
        assert np.ptp(states[0]["pressure"]) < np.ptp(state0["pressure"])
        assert states[0]["flux"].shape == (model.G.faces.num, 3)
        print(f"ok test_faithful_model_conserves_mass_in_a_closed_box "
              f"(imbalance {imbalance:.1e}, toleranceMB {model.toleranceMB:.0e})")

    test_faithful_model_builds_one_component_per_phase()
    test_faithful_model_conserves_mass_in_a_closed_box()
    print("\nAll tests passed!")
