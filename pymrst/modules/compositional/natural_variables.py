"""Natural-variable compositional model (MRST `NaturalVariablesCompositionalModel`).

This native port covers the liquid/vapour formulation, with optional immiscible
water. The primary variables are pressure, liquid composition, water/liquid/gas
saturations, vapour composition and the existing well variables. The residual is
MRST's natural-variable system: water conservation when active, component
conservation, fugacity equality for each component and saturation closure.
"""

from dataclasses import dataclass
from typing import List, Tuple
import warnings

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import MatrixRankWarning

from pymrst.modules.compositional.backend import (
    backend_flash,
    backend_flow_props,
    backend_natural_props,
    backend_natural_props_ad,
)
from pymrst.modules.compositional.flow_props import mass_fraction
from pymrst.modules.compositional.model import CompositionalState, OverallCompositionModel
from pymrst.params.wells_and_bc import BoundaryCondition, Source
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA


@dataclass
class NaturalVariablesState(CompositionalState):
    """Natural-variable state with explicit liquid/vapour phase variables."""

    x: np.ndarray | None = None
    y: np.ndarray | None = None
    sO: np.ndarray | None = None
    sG: np.ndarray | None = None
    flag: np.ndarray | None = None
    K: np.ndarray | None = None
    L: np.ndarray | None = None
    Z_L: np.ndarray | None = None
    Z_V: np.ndarray | None = None
    switched: np.ndarray | None = None
    switchCount: np.ndarray | None = None

    def copy(self) -> "NaturalVariablesState":
        s_w = None if self.sW is None else self.sW.copy()
        q_ws = None if self.qWs is None else self.qWs.copy()
        s_t = None if self.sT is None else self.sT.copy()
        total_flux = None if self.total_flux is None else self.total_flux.copy()
        well_flux = None if self.well_flux is None else self.well_flux.copy()
        well_components = None if self.well_components is None else self.well_components.copy()
        return NaturalVariablesState(
            self.pressure.copy(),
            self.z.copy(),
            self.qOs.copy(),
            self.qGs.copy(),
            self.bhp.copy(),
            s_w,
            q_ws,
            s_t,
            None if self.x is None else self.x.copy(),
            None if self.y is None else self.y.copy(),
            None if self.sO is None else self.sO.copy(),
            None if self.sG is None else self.sG.copy(),
            None if self.flag is None else self.flag.copy(),
            None if self.K is None else self.K.copy(),
            None if self.L is None else self.L.copy(),
            None if self.Z_L is None else self.Z_L.copy(),
            None if self.Z_V is None else self.Z_V.copy(),
            None if self.switched is None else self.switched.copy(),
            None if self.switchCount is None else self.switchCount.copy(),
            total_flux=total_flux,
            well_flux=well_flux,
            well_components=well_components,
        )


class NaturalVariablesCompositionalModel(OverallCompositionModel):
    """Fully implicit natural-variable model for liquid/vapour compositional flow."""

    def __init__(self, *args, saturation_epsilon: float = 1.0e-8,
                 phase_switch_epsilon: float = 1.0e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.saturation_epsilon = float(saturation_epsilon)
        self.phase_switch_epsilon = float(phase_switch_epsilon)
        self.tol_fugacity = 1.0e-7
        self.tol_closure = 1.0e-9

    # -- MRST-style public compatibility surface ---------------------------
    @property
    def allowLargeSaturations(self) -> bool:
        return bool(getattr(self, "allow_large_saturations", False))

    @allowLargeSaturations.setter
    def allowLargeSaturations(self, value: bool) -> None:
        self.allow_large_saturations = bool(value)

    @property
    def saturationEpsilon(self) -> float:
        return self.saturation_epsilon

    @saturationEpsilon.setter
    def saturationEpsilon(self, value: float) -> None:
        self.saturation_epsilon = float(value)

    @property
    def flashFromSinglePhase(self) -> bool:
        return bool(getattr(self, "flash_from_single_phase", False))

    @flashFromSinglePhase.setter
    def flashFromSinglePhase(self, value: bool) -> None:
        self.flash_from_single_phase = bool(value)

    def getComponentNames(self) -> tuple[str, ...]:
        """MRST-style EOS component names."""
        return tuple(str(x) for x in getattr(self.mix, "names", tuple(range(self.mix.n))))

    def getPhaseNames(self) -> tuple[str, ...]:
        """Return active phase names in PyMRST's zero-based phase order."""
        return (("water",) if self.water else tuple()) + ("oil", "gas")

    def getLiquidIndex(self) -> int:
        """Zero-based liquid/oil phase index."""
        return int(bool(self.water))

    def getVaporIndex(self) -> int:
        """Zero-based vapour/gas phase index."""
        return int(bool(self.water)) + 1

    def getNumberOfPhases(self) -> int:
        return len(self.getPhaseNames())

    def getFlag(self, state: NaturalVariablesState):
        """MRST-style ``[pureLiquid, pureVapor, twoPhase]`` flags."""
        return self._get_flag(state)

    def getTwoPhaseFlag(self, state: NaturalVariablesState) -> np.ndarray:
        return self._get_flag(state)[2]

    def updateSecondaryProperties(self, state: NaturalVariablesState) -> NaturalVariablesState:
        """Refresh phase variables from the EOS flash, MRST-style."""
        return self._sync_phase_state_from_flash(state.copy())

    def flashPhases(self, state: NaturalVariablesState, state0=None, s_uncap=None,
                    xyUpdated: bool = False, iteration: int | None = None) -> NaturalVariablesState:
        """Compatibility wrapper for MRST ``flashPhases``."""
        _ = state0, s_uncap, xyUpdated, iteration
        return self._sync_phase_state_from_flash(state.copy())

    def validateState(self, state: NaturalVariablesState) -> NaturalVariablesState:
        """Normalize and refresh a natural-variable state."""
        out = state.copy()
        out.pressure = np.asarray(out.pressure, dtype=float).reshape(-1)
        out.z = np.maximum(np.asarray(out.z, dtype=float), self.min_z)
        out.z = out.z / out.z.sum(axis=1, keepdims=True)
        if out.x is None or out.y is None or out.sO is None or out.sG is None:
            return self.initial_state(out.pressure, out.z, sW=out.sW)
        out.x = np.maximum(np.asarray(out.x, dtype=float), self.min_z)
        out.y = np.maximum(np.asarray(out.y, dtype=float), self.min_z)
        out.x = out.x / out.x.sum(axis=1, keepdims=True)
        out.y = out.y / out.y.sum(axis=1, keepdims=True)
        out.sW, out.sO, out.sG = self._normalize_all_saturations(out.sW, out.sO, out.sG)
        out.z = self._overall_from_natural(out.pressure, out.sO, out.sG, out.x, out.y)
        return self._sync_phase_state_from_flash(out)

    def getPrimaryVariables(self, state: NaturalVariablesState):
        """Return MRST-like primary variables, names and origins.

        The values are ordinary arrays; callers that need AD variables can pass
        these through PyMRST's ``init_variables_adi`` or backend-specific
        initializer.  Names follow MRST's natural-variable convention:
        pressure, liquid composition ``v_*``, optional water saturation, liquid
        saturation on two-phase cells, vapour composition ``w_*`` on two-phase
        cells, vapour saturation on two-phase cells, then well variables.
        """
        st = self.validateState(state)
        ncomp = self.mix.n
        names = ["pressure"]
        vars = [st.pressure.copy()]
        for name, col in zip(self.getComponentNames()[:-1], range(ncomp - 1)):
            names.append(f"v_{name}")
            vars.append(st.x[:, col].copy())
        if self.water:
            names.append("sW")
            vars.append(st.sW.copy())
        two_phase = self.getTwoPhaseFlag(st)
        names.append("sL")
        vars.append(st.sO[two_phase].copy())
        for name, col in zip(self.getComponentNames()[:-1], range(ncomp - 1)):
            names.append(f"w_{name}")
            vars.append(st.y[two_phase, col].copy())
        names.append("sV")
        vars.append(st.sG[two_phase].copy())
        if self.water:
            names.append("qWs")
            vars.append(st.qWs.copy())
        names.extend(["qOs", "qGs", "bhp"])
        vars.extend([st.qOs.copy(), st.qGs.copy(), st.bhp.copy()])
        origin = [self.__class__.__name__] * len(vars)
        return vars, names, origin

    def _flag_from_vapour_fraction(self, V: np.ndarray) -> np.ndarray:
        flag = np.zeros(np.asarray(V).size, dtype=np.int8)
        flag[V <= self.phase_switch_epsilon] = 1
        flag[V >= 1.0 - self.phase_switch_epsilon] = 2
        mid = (V > self.phase_switch_epsilon) & (V < 1.0 - self.phase_switch_epsilon)
        flag[mid] = 0
        return flag

    def _flag_from_flash(self, p, z) -> np.ndarray:
        """MRST EOS phase flag: 0 two-phase, 1 liquid-only, 2 vapour-only."""
        fr = backend_flash(self.mix, self.eos_model, np.asarray(p, dtype=float), self.T, np.asarray(z, dtype=float))
        return self._flag_from_vapour_fraction(fr.V)

    @staticmethod
    def _get_flag(state: NaturalVariablesState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if state.flag is None:
            nc = state.pressure.size
            flag = np.zeros(nc, dtype=np.int8)
        else:
            flag = np.asarray(state.flag, dtype=np.int8)
        pure_liquid = flag == 1
        pure_vapour = flag == 2
        two_phase = flag == 0
        return pure_liquid, pure_vapour, two_phase

    def _sync_phase_state_from_flash(self, state: NaturalVariablesState) -> NaturalVariablesState:
        """Update phase flags and MRST natural-variable flash bookkeeping.

        MRST ``flashPhases`` refreshes the phase-state variables after a
        nonlinear update and carries several fields forward for the next
        assembly: equilibrium ratios ``K``, molar liquid fraction ``L``,
        compressibility roots, switch counters and a per-cell switched mask.
        PyMRST's full switching heuristics live in the Newton update path, but
        this refresh now preserves the same state surface instead of dropping
        those EOS bookkeeping quantities.
        """
        assert state.x is not None and state.y is not None and state.sO is not None and state.sG is not None
        old_two_phase = self._get_flag(state)[2].copy()
        fr = backend_flash(self.mix, self.eos_model, state.pressure, self.T, state.z)
        fp = backend_flow_props(self.mix, self.eos_model, state.pressure, self.T, state.z, flash_result=fr)
        state.flag = self._flag_from_vapour_fraction(fr.V)
        if self.water:
            assert state.sW is not None
            s_o, s_g = self._scale_hydrocarbon_saturations(state.sW, fp.sL, fp.sV)
        else:
            s_o, s_g = self._normalize_saturations(fp.sL, fp.sV)
        state.x = fp.x.copy()
        state.y = fp.y.copy()
        pure_liquid, pure_vapour, new_two_phase = self._get_flag(state)
        if np.any(pure_liquid):
            state.y[pure_liquid, :] = state.x[pure_liquid, :]
        if np.any(pure_vapour):
            state.x[pure_vapour, :] = state.y[pure_vapour, :]
        state.sO = s_o
        state.sG = s_g
        state.L = np.asarray(1.0 - fr.V, dtype=float).reshape(-1)
        state.Z_L = np.asarray(fr.Z_L, dtype=float).reshape(-1)
        state.Z_V = np.asarray(fr.Z_V, dtype=float).reshape(-1)
        state.K = np.divide(
            state.y,
            state.x,
            out=np.ones_like(state.y),
            where=np.abs(state.x) > 0.0,
        )
        state.switched = old_two_phase != new_two_phase
        if state.switchCount is None:
            state.switchCount = np.zeros(state.pressure.size, dtype=float)
        state.switchCount = np.asarray(state.switchCount, dtype=float).reshape(-1)
        state.switchCount[state.switched] += 1.0
        state.z = self._overall_from_natural(state.pressure, state.sO, state.sG, state.x, state.y)
        return state

    def initial_state(self, p, z, sW=None) -> NaturalVariablesState:
        base = super().initial_state(p, z, sW=sW)
        fr = backend_flash(self.mix, self.eos_model, base.pressure, self.T, base.z)
        fp = backend_flow_props(self.mix, self.eos_model, base.pressure, self.T, base.z, flash_result=fr)
        if self.water:
            assert base.sW is not None
            s_o, s_g = self._scale_hydrocarbon_saturations(base.sW, fp.sL, fp.sV)
        else:
            s_o, s_g = self._normalize_saturations(fp.sL, fp.sV)
        flag = self._flag_from_vapour_fraction(fr.V)
        z_nat = self._overall_from_natural(base.pressure, s_o, s_g, fp.x, fp.y)
        return NaturalVariablesState(
            base.pressure,
            z_nat,
            base.qOs,
            base.qGs,
            base.bhp,
            base.sW,
            base.qWs,
            base.sT,
            fp.x.copy(),
            fp.y.copy(),
            s_o,
            s_g,
            flag,
            np.divide(fp.y, fp.x, out=np.ones_like(fp.y), where=np.abs(fp.x) > 0.0),
            np.asarray(1.0 - fr.V, dtype=float).reshape(-1),
            np.asarray(fr.Z_L, dtype=float).reshape(-1),
            np.asarray(fr.Z_V, dtype=float).reshape(-1),
            np.zeros(base.pressure.size, dtype=bool),
            np.zeros(base.pressure.size, dtype=float),
        )

    def _normalize_saturations(self, s_o, s_g) -> tuple[np.ndarray, np.ndarray]:
        s_o = np.asarray(s_o, dtype=float).copy()
        s_g = np.asarray(s_g, dtype=float).copy()
        eps = self.saturation_epsilon
        s_o = np.maximum(s_o, eps)
        s_g = np.maximum(s_g, eps)
        total = s_o + s_g
        return s_o / total, s_g / total

    def _scale_hydrocarbon_saturations(self, s_w, s_o, s_g) -> tuple[np.ndarray, np.ndarray]:
        s_w = np.clip(np.asarray(s_w, dtype=float), 0.0, 1.0 - 2.0 * self.saturation_epsilon)
        split_o, split_g = self._normalize_saturations(s_o, s_g)
        hydrocarbon = np.maximum(1.0 - s_w, 2.0 * self.saturation_epsilon)
        return hydrocarbon * split_o, hydrocarbon * split_g

    def _normalize_all_saturations(self, s_w, s_o, s_g):
        if not self.water:
            return None, *self._normalize_saturations(s_o, s_g)
        s_w = np.clip(np.asarray(s_w, dtype=float), self.saturation_epsilon, 1.0 - 2.0 * self.saturation_epsilon)
        s_o, s_g = self._scale_hydrocarbon_saturations(s_w, s_o, s_g)
        return s_w, s_o, s_g

    def _overall_from_natural(self, p, s_o, s_g, x, y) -> np.ndarray:
        """Reconstruct overall mole fractions from natural phase variables.

        MRST's natural-variable state stores phase saturations as volume
        fractions and `x`/`y` as mole fractions. Overall composition is
        therefore saturation weighted by phase molar density, not the simpler
        `sO*x + sG*y` volume average.
        """
        p = np.asarray(p, dtype=float)
        s_o = np.asarray(s_o, dtype=float)
        s_g = np.asarray(s_g, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        props = backend_natural_props(self.mix, self.eos_model, p, self.T, x, y)
        n_o = s_o * props["nL"]
        n_g = s_g * props["nV"]
        z = n_o[:, None] * x + n_g[:, None] * y
        z = np.maximum(z, 1.0e-12)
        return z / z.sum(axis=1, keepdims=True)

    def _component_list(self, primaries: List, ncomp: int) -> List:
        last = 1.0 - primaries[0]
        for i in range(1, ncomp - 1):
            last = last - primaries[i]
        return list(primaries) + [last]

    def _component_list_natural(self, x_prim: List, y_two_prim: List,
                                two_phase: np.ndarray, pure_liquid: np.ndarray,
                                pure_vapour: np.ndarray, ncomp: int) -> tuple[List, List]:
        """Build full-cell liquid/vapour composition AD variables.

        MRST keeps the liquid composition variables full-cell. Vapour
        composition variables are active only in two-phase cells; in single
        phase cells the absent phase composition is copied as a value, while
        the present phase carries the composition variable.
        """
        x = self._component_list(x_prim, ncomp)
        y = []
        for i in range(ncomp - 1):
            yi = x_prim[i].copy()
            if np.any(pure_liquid):
                yi[pure_liquid] = value(x_prim[i])[pure_liquid]
            if np.any(two_phase):
                yi[two_phase] = y_two_prim[i]
            y.append(yi)

        y_last = 1.0 - y[0]
        for i in range(1, ncomp - 1):
            y_last = y_last - y[i]
        y.append(y_last)

        if np.any(pure_vapour):
            for i in range(ncomp):
                x[i][pure_vapour] = value(x[i])[pure_vapour]
        return x, y

    @staticmethod
    def _phase_saturation_ad(p, s_two, two_phase: np.ndarray, fill: np.ndarray):
        out = p * 0.0 + np.asarray(fill, dtype=float)
        if np.any(two_phase):
            out[two_phase] = s_two
        return out

    def _natural_props_ad(self, p, x: List, y: List):
        return backend_natural_props_ad(self.mix, self.eos_model, p, self.T, x, y)

    def _natural_props0(self, st0: NaturalVariablesState):
        assert st0.x is not None and st0.y is not None and st0.sO is not None and st0.sG is not None
        x_m = mass_fraction(self.mix, [st0.x[:, i] for i in range(self.mix.n)])
        y_m = mass_fraction(self.mix, [st0.y[:, i] for i in range(self.mix.n)])
        props = backend_natural_props(self.mix, self.eos_model, st0.pressure, self.T, st0.x, st0.y)
        rho_o = props["rhoL"]
        rho_g = props["rhoV"]
        return x_m, y_m, st0.sO, st0.sG, rho_o, rho_g

    def assemble_natural(
        self,
        st0: NaturalVariablesState,
        p,
        x_prim: List,
        s_o,
        y_prim: List,
        s_g,
        qo,
        qg,
        bhp,
        dt: float,
        s_w=None,
        qw=None,
        bc: BoundaryCondition | None = None,
        src: Source | None = None,
        flag: np.ndarray | None = None,
    ):
        ops, pv = self.ops, self.pv
        ncomp = self.mix.n
        if flag is None:
            flag = np.zeros(self.g.cells.num, dtype=np.int8)
        flag = np.asarray(flag, dtype=np.int8)
        pure_liquid = flag == 1
        pure_vapour = flag == 2
        two_phase = flag == 0
        x, y = self._component_list_natural(x_prim, y_prim, two_phase, pure_liquid, pure_vapour, ncomp)
        if self.water:
            assert s_w is not None
            hydrocarbon = np.maximum(1.0 - value(s_w), 2.0 * self.saturation_epsilon)
        else:
            hydrocarbon = np.ones(self.g.cells.num)
        s_o_fill = np.where(pure_liquid, hydrocarbon, 0.0)
        s_g_fill = np.where(pure_vapour, hydrocarbon, 0.0)
        s_o = self._phase_saturation_ad(p, s_o, two_phase, s_o_fill)
        s_g = self._phase_saturation_ad(p, s_g, two_phase, s_g_fill)
        x_m, y_m, rho_o, rho_g, mu_o, mu_g, f_l, f_v = self._natural_props_ad(p, x, y)
        x_m0, y_m0, s_o0, s_g0, rho_o0, rho_g0 = self._natural_props0(st0)

        if self.water:
            assert s_w is not None and st0.sW is not None
        elif s_w is not None:
            raise ValueError("s_w was provided but water is not active")

        kr_w, kr_o, kr_g = self._relative_permeabilities(s_w if self.water else None, s_o, s_g)
        mob_o = kr_o / mu_o
        mob_g = kr_g / mu_g
        b_o = rho_o / self.rho_s[0]
        b_g = rho_g / self.rho_s[1]

        rho_of = ops.avg(s_o * rho_o) / np.maximum(value(ops.avg(s_o)), 1.0e-8)
        rho_gf = ops.avg(s_g * rho_g) / np.maximum(value(ops.avg(s_g)), 1.0e-8)
        dp_o = ops.grad(p) - rho_of * self.gdz
        dp_g = ops.grad(p) - rho_gf * self.gdz
        up_o = value(dp_o) <= 0
        up_g = value(dp_g) <= 0
        v_o = -ops.upw(up_o, mob_o) * ops.T * dp_o
        v_g = -ops.upw(up_g, mob_g) * ops.T * dp_g
        r_ovo = ops.upw(up_o, rho_o) * v_o
        r_gvg = ops.upw(up_g, rho_g) * v_g

        eqs = []
        if self.water:
            assert kr_w is not None
            rho_w = self.water_rho_s
            mob_w = kr_w / self.water_mu
            p_w = p - self.pc_ow(s_w) if self.pc_ow is not None else p
            dp_w = ops.grad(p_w) - rho_w * self.gdz
            up_w = value(dp_w) <= 0
            v_w = -ops.upw(up_w, mob_w) * ops.T * dp_w
            acc_w = (pv * s_w - pv * st0.sW) / dt
            eqs.append(acc_w + ops.div(v_w))

        for i in range(ncomp):
            acc = (1.0 / dt) * (
                pv * rho_o * s_o * x_m[i] - pv * rho_o0 * s_o0 * x_m0[i]
                + pv * rho_g * s_g * y_m[i] - pv * rho_g0 * s_g0 * y_m0[i]
            )
            flux = r_ovo * ops.upw(up_o, x_m[i]) + r_gvg * ops.upw(up_g, y_m[i])
            eqs.append(acc + ops.div(flux))

        bc_forces = self.bc if bc is None else bc
        if bc_forces is not None and bc_forces.face.size:
            bc_sources, bc_cells = self._boundary_component_sources(
                bc_forces, p, mob_o, mob_g, rho_o, rho_g, b_o, b_g, x_m, y_m,
                s_o=s_o, s_g=s_g,
                mob_w=mob_w if self.water else None,
                rho_w=self.water_rho_s if self.water else None,
                b_w=1.0 if self.water else None,
                s_w=s_w if self.water else None,
            )
            self._subtract_cell_sources(eqs, bc_sources, bc_cells)

        src_forces = self.src if src is None else src
        if src_forces is not None and src_forces.cell.size:
            src_sources, src_cells = self._source_component_sources(
                src_forces, mob_o, mob_g, rho_o, rho_g, b_o, b_g, x_m, y_m,
                mob_w=mob_w if self.water else None,
                rho_w=self.water_rho_s if self.water else None,
                b_w=1.0 if self.water else None,
            )
            self._subtract_cell_sources(eqs, src_sources, src_cells)

        well_eqs = []
        ctrls = []
        for k in range(len(self.wells)):
            if self.water:
                assert qw is not None
                cq_s, comp_src, weqs, wc = self._water_well_contributions(
                    k, p, mob_w, mob_o, mob_g, b_o, b_g, x_m, y_m,
                    qw[k:k + 1], qo[k:k + 1], qg[k:k + 1], bhp[k:k + 1],
                    rho_o=rho_o, rho_g=rho_g)
                eqs[0][wc] = eqs[0][wc] - cq_s[0]
                for i in range(ncomp):
                    eqs[i + 1][wc] = eqs[i + 1][wc] - comp_src[i]
                well_eqs.extend(weqs[:3])
                ctrls.append(weqs[3])
            else:
                _, comp_src, weqs, wc = self._well_contributions(
                    k, p, mob_o, mob_g, b_o, b_g, x_m, y_m,
                    qo[k:k + 1], qg[k:k + 1], bhp[k:k + 1],
                    rho_o=rho_o, rho_g=rho_g)
                for i in range(ncomp):
                    eqs[i][wc] = eqs[i][wc] - comp_src[i]
                well_eqs.extend(weqs[:2])
                ctrls.append(weqs[2])

        fugacity_eqs = [(f_l[i][two_phase] - f_v[i][two_phase]) / BARSA for i in range(ncomp)]
        closure = s_w + s_o + s_g - 1.0 if self.water else s_o + s_g - 1.0
        closure = closure[two_phase]
        ctrl = None if not ctrls else ctrls[0] if len(ctrls) == 1 else None
        if len(ctrls) > 1:
            from pymrst.utils.adi import vertcat_adi

            ctrl = vertcat_adi(*ctrls)
        return eqs, well_eqs, ctrl, fugacity_eqs, closure

    def _natural_converged(self, eqs, well_eqs, ctrl, fugacity_eqs, closure, dt: float, scale: float) -> bool:
        cnv = max(np.abs(value(eq) / self.pv).max() * dt / scale for eq in eqs)
        fug = max((np.abs(value(eq)).max() for eq in fugacity_eqs if value(eq).size), default=0.0)
        clo_v = value(closure)
        clo = np.abs(clo_v).max() if clo_v.size else 0.0
        if not self.wells:
            return cnv <= self.tol_cnv and fug <= self.tol_fugacity and clo <= self.tol_closure
        q_ref = max(max(abs(w.val) for w in self.wells if w.type == "rate"), 1.0e-12) if any(
            w.type == "rate" for w in self.wells
        ) else 1.0
        well_ok = all(np.abs(value(e)).max() <= self.tol_well * q_ref for e in well_eqs)
        ctrl_ok = np.all(
            np.abs(value(ctrl))
            <= self.tol_well * np.array([BARSA if w.type == "bhp" else q_ref for w in self.wells])
        )
        return cnv <= self.tol_cnv and fug <= self.tol_fugacity and clo <= self.tol_closure and well_ok and ctrl_ok

    def step(
        self,
        state: NaturalVariablesState,
        dt: float,
        bc: BoundaryCondition | None = None,
        src: Source | None = None,
    ) -> Tuple[NaturalVariablesState, bool, int]:
        st0 = state
        st = state.copy()
        assert st.x is not None and st.y is not None and st.sO is not None and st.sG is not None
        if st.flag is None:
            st.flag = self._flag_from_flash(st.pressure, st.z)
        nc = self.g.cells.num
        ncomp = self.mix.n
        nw = len(self.wells)
        scale = self._mass_scale(st0)
        pure_liquid, pure_vapour, two_phase = self._get_flag(st)
        ntwo = int(np.count_nonzero(two_phase))
        if ntwo == 0 or (bc is not None and len(self.wells) == 0):
            overall = CompositionalState(
                st.pressure.copy(),
                st.z.copy(),
                st.qOs.copy(),
                st.qGs.copy(),
                st.bhp.copy(),
                None if st.sW is None else st.sW.copy(),
                None if st.qWs is None else st.qWs.copy(),
                None if st.sT is None else st.sT.copy(),
            )
            overall_new, ok, it = super().step(overall, dt, bc=bc, src=src)
            st.pressure = overall_new.pressure
            st.z = overall_new.z
            st.qOs = overall_new.qOs
            st.qGs = overall_new.qGs
            st.bhp = overall_new.bhp
            st.sW = overall_new.sW
            st.qWs = overall_new.qWs
            st.sT = overall_new.sT
            st = self._sync_phase_state_from_flash(st)
            st.z = overall_new.z
            return st, ok, it

        for it in range(self.max_iterations):
            primary_values = [
                st.pressure,
                *[st.x[:, i] for i in range(ncomp - 1)],
            ]
            if self.water:
                assert st.sW is not None and st.qWs is not None
                primary_values.append(st.sW)
            primary_values.extend([
                st.sO[two_phase],
                *[st.y[two_phase, i] for i in range(ncomp - 1)],
                st.sG[two_phase],
            ])
            if self.water:
                primary_values.append(st.qWs)
            primary_values.extend([st.qOs, st.qGs, st.bhp])
            prim = self._init_primary_variables(*primary_values)
            p_ad = prim[0]
            x_prim = list(prim[1:1 + ncomp - 1])
            off = 1 + ncomp - 1
            s_w_ad = None
            if self.water:
                s_w_ad = prim[off]
                off += 1
            s_o_ad = prim[off]
            off += 1
            y_start = off
            y_prim = list(prim[y_start:y_start + ncomp - 1])
            off = y_start + ncomp - 1
            s_g_ad = prim[off]
            off += 1
            qw = None
            if self.water:
                qw = prim[off]
                off += 1
            qo, qg, bhp_ad = prim[off:off + 3]
            eqs, well_eqs, ctrl, fugacity_eqs, closure = self.assemble_natural(
                st0, p_ad, x_prim, s_o_ad, y_prim, s_g_ad, qo, qg, bhp_ad, dt,
                s_w=s_w_ad, qw=qw, bc=bc, src=src, flag=st.flag)
            if self._natural_converged(eqs, well_eqs, ctrl, fugacity_eqs, closure, dt, scale):
                st.z = self._overall_from_natural(st.pressure, st.sO, st.sG, st.x, st.y)
                st = self._sync_phase_state_from_flash(st)
                return st, True, it

            parts = list(eqs)
            if nw:
                parts.extend(well_eqs)
                parts.append(ctrl)
            parts.extend(fugacity_eqs)
            parts.append(closure)
            system = self._combine_equations_for_backend(*parts)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MatrixRankWarning)
                upd = spla.spsolve(system.jac[0].tocsc(), -system.val)
            if not np.all(np.isfinite(upd)):
                upd = np.linalg.lstsq(system.jac[0].toarray(), -system.val, rcond=None)[0]

            off = 0
            dp = upd[off:off + nc]
            off += nc
            cap = self.dp_max_rel * np.abs(st.pressure)
            st.pressure = st.pressure + np.clip(dp, -cap, cap)
            for i in range(ncomp - 1):
                dx = upd[off:off + nc]
                off += nc
                st.x[:, i] = st.x[:, i] + np.clip(dx, -self.dz_max, self.dz_max)
            if self.water:
                assert st.sW is not None
                ds_w = upd[off:off + nc]
                off += nc
                st.sW = st.sW + np.clip(ds_w, -self.ds_max, self.ds_max)
            ds_o = upd[off:off + ntwo]
            off += ntwo
            st.sO[two_phase] = st.sO[two_phase] + np.clip(ds_o, -self.ds_max, self.ds_max)
            for i in range(ncomp - 1):
                dy = upd[off:off + ntwo]
                off += ntwo
                st.y[two_phase, i] = st.y[two_phase, i] + np.clip(dy, -self.dz_max, self.dz_max)
            ds_g = upd[off:off + ntwo]
            off += ntwo
            st.sG[two_phase] = st.sG[two_phase] + np.clip(ds_g, -self.ds_max, self.ds_max)
            st.x[:, ncomp - 1] = 1.0 - st.x[:, :ncomp - 1].sum(axis=1)
            st.y[:, ncomp - 1] = 1.0 - st.y[:, :ncomp - 1].sum(axis=1)
            st.x = np.clip(st.x, self.min_z, 1.0)
            st.y = np.clip(st.y, self.min_z, 1.0)
            st.x = st.x / st.x.sum(axis=1, keepdims=True)
            st.y = st.y / st.y.sum(axis=1, keepdims=True)
            if np.any(pure_liquid):
                st.y[pure_liquid, :] = st.x[pure_liquid, :]
            if np.any(pure_vapour):
                st.x[pure_vapour, :] = st.y[pure_vapour, :]
            if self.water:
                assert st.sW is not None
                hydrocarbon = np.maximum(1.0 - st.sW, 2.0 * self.saturation_epsilon)
            else:
                hydrocarbon = np.ones(nc)
            if np.any(pure_liquid):
                st.sO[pure_liquid] = hydrocarbon[pure_liquid]
                st.sG[pure_liquid] = 0.0
            if np.any(pure_vapour):
                st.sO[pure_vapour] = 0.0
                st.sG[pure_vapour] = hydrocarbon[pure_vapour]
            st.sW, st.sO, st.sG = self._normalize_all_saturations(st.sW, st.sO, st.sG)
            if nw:
                if self.water:
                    assert st.qWs is not None
                    st.qWs = st.qWs + upd[off:off + nw]
                    off += nw
                st.qOs = st.qOs + upd[off:off + nw]
                off += nw
                st.qGs = st.qGs + upd[off:off + nw]
                off += nw
                st.bhp = st.bhp + upd[off:off + nw]
            st.z = self._overall_from_natural(st.pressure, st.sO, st.sG, st.x, st.y)
        st = self._sync_phase_state_from_flash(st)
        return st, False, self.max_iterations


GenericNaturalVariablesModel = NaturalVariablesCompositionalModel


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.compositional.eos import simple_mixture
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import DARCY, DAY, MILLI

    def _closed_model(nx: int = 4):
        g = compute_geometry(cart_grid([nx, 1, 1], [40.0, 10.0, 10.0]))
        rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
        return NaturalVariablesCompositionalModel(g, rock, simple_mixture(), temperature=350.0)

    def test_natural_initial_state_has_phase_variables():
        model = _closed_model()
        state = model.initial_state(50.0 * BARSA, [0.5, 0.3, 0.2])
        assert state.x.shape == (4, 3)
        assert state.y.shape == (4, 3)
        assert np.allclose(state.sO + state.sG, 1.0)
        assert np.all(np.isfinite(state.z))
        assert np.allclose(state.z.sum(axis=1), 1.0)
        print("ok test_natural_initial_state_has_phase_variables")

    def test_closed_natural_step_converges():
        model = _closed_model()
        state = model.initial_state(np.linspace(55.0 * BARSA, 45.0 * BARSA, 4), [0.5, 0.3, 0.2])
        new_state, ok, _ = model.step(state, 1.0 * DAY)
        assert ok
        assert np.allclose(new_state.sO + new_state.sG, 1.0)
        assert np.allclose(new_state.x.sum(axis=1), 1.0)
        assert np.allclose(new_state.y.sum(axis=1), 1.0)
        print("ok test_closed_natural_step_converges")

    def test_natural_well_step_converges():
        g = compute_geometry(cart_grid([4, 1, 1], [40.0, 10.0, 10.0]))
        rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
        wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                         val=1.0e-5, compi=np.array([0.0, 1.0]), sign=1)
        wells[0].components = np.array([0.9, 0.05, 0.05])
        wells = add_well(wells, g, rock, np.array([3]), well_type="bhp",
                         val=45.0 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        wells[1].components = np.array([0.5, 0.3, 0.2])
        model = NaturalVariablesCompositionalModel(g, rock, simple_mixture(), temperature=350.0, wells=wells)
        state = model.initial_state(50.0 * BARSA, [0.5, 0.3, 0.2])
        new_state, ok, _ = model.step(state, 0.5 * DAY)
        assert ok
        assert new_state.qGs[0] > 0.0
        assert new_state.qOs[1] + new_state.qGs[1] < 0.0
        print("ok test_natural_well_step_converges")

    def test_natural_water_step_converges():
        g = compute_geometry(cart_grid([4, 1, 1], [40.0, 10.0, 10.0]))
        rock = make_rock(g, 100.0 * MILLI * DARCY, 0.25)
        wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                         val=1.0e-5, compi=np.array([1.0, 0.0, 0.0]), sign=1)
        wells[0].components = np.array([0.5, 0.3, 0.2])
        wells = add_well(wells, g, rock, np.array([3]), well_type="bhp",
                         val=45.0 * BARSA, compi=np.array([0.0, 0.5, 0.5]), sign=-1)
        wells[1].components = np.array([0.5, 0.3, 0.2])
        model = NaturalVariablesCompositionalModel(
            g, rock, simple_mixture(), temperature=350.0, wells=wells, water=True
        )
        state = model.initial_state(50.0 * BARSA, [0.5, 0.3, 0.2], sW=0.2)
        new_state, ok, _ = model.step(state, 0.25 * DAY)
        assert ok
        assert new_state.sW is not None and new_state.qWs is not None
        assert np.allclose(new_state.sW + new_state.sO + new_state.sG, 1.0)
        assert new_state.qWs[0] > 0.0
        print("ok test_natural_water_step_converges")

    test_natural_initial_state_has_phase_variables()
    test_closed_natural_step_converges()
    test_natural_well_step_converges()
    test_natural_water_step_converges()
    print("\nAll tests passed!")


__all__ = [
    "GenericNaturalVariablesModel",
    "NaturalVariablesCompositionalModel",
    "NaturalVariablesState",
]
