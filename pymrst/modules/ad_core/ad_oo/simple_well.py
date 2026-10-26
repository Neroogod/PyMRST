"""MRST ``ad-core/models/facilities/SimpleWell.m``.

The standard well model: instantaneous flow, no wellbore storage, a
hydrostatic pressure drop between perforations, and Peaceman inflow through
each connection's well index. It assembles one well's rate and control
equations, keeps the connection pressure drops up to date, switches control
when a limit is breached, and applies Newton increments to its `wellSol`.

`wellSol` rows are records with attribute access (as `initWellSolAD` builds
them); `W` is a PyMRST `Well`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.ad_oo.compute_well_contributions_single_well import (
    compute_well_contributions_single_well,
)
from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel
from pymrst.modules.ad_core.ad_oo.setup_well_control_equations_single_well import (
    setup_well_control_equations_single_well,
)
from pymrst.modules.ad_core.utils.assign_value import assign_value
from pymrst.modules.ad_core.utils.phase_densities_to_bfactor import (
    phase_densities_to_bfactor,
)
from pymrst.params.gravity.gravity import gravity
from pymrst.utils.adi import value
from pymrst.utils.units import BARSA

ZERO_RATE: float = 1e-12
TOPOLOGY_ITERATIONS: int = 100
SHUT_WELL_TRANSMISSIBILITY: float = 1e-16
TINY_MIXTURE_DENSITY: float = 1e-8
SHUT_MASS_RATE: float = 1e-20


class SimpleWell(PhysicalModel):
    """One standard well and the equations that tie it to the reservoir."""

    def __init__(self, W: Any, **options: Any):
        super().__init__(None)
        self.W = W
        self.allowCrossflow = True
        self.allowSignChange = False
        self.allowControlSwitching = True
        self.doUpdatePressureDrop = True
        self.simplePressureDrop = False
        self.dpMaxRel = float("inf")
        self.dpMaxAbs = float("inf")
        self.dsMaxAbs = float("inf")
        self.VFPTable = None
        self.ControlDensity = None
        if options:
            from pymrst.modules.ad_core.ad_oo.physical_model import merge_options
            merge_options(self, options)

    def update_well(self, W: Any) -> "SimpleWell":
        self.W = W
        return self

    def validate_well_sol(self, resmodel: Any, wsol: Any, state: Any = None) -> Any:
        """Add any extra primary variable the well carries, at zero.

        MRST tests `isfield(wsol, 'fn')` - the literal string - so every
        extra variable is reset to zero on validation. Ported as written.
        """
        names, from_res_model = self.get_extra_primary_variable_names(resmodel)
        for name, from_res in zip(names, from_res_model):
            m = resmodel if from_res else self
            fn, _ = m.get_variable_field(name)
            if not hasattr(wsol, "fn"):
                wsol = m.set_prop(wsol, fn, 0)
        return wsol

    def get_variable_counts(self, fld: str) -> int:
        try:
            fn, _ = self.get_variable_field(fld)
        except KeyError:
            fn = None
        return 0 if fn is None else 1

    def get_extra_primary_variable_names(self, resmodel: Any):
        names = list(resmodel.get_extra_well_primary_variable_names())
        return names, [True]*len(names)

    def get_extra_equation_names(self, resmodel: Any):
        return resmodel.get_extra_well_equation_names()

    def get_extra_primary_variables(self, wellSol: Any, resmodel: Any):
        names, from_res = self.get_extra_primary_variable_names(resmodel)
        values = [resmodel.get_props(wellSol, n) if r else self.get_props(wellSol, n)
                  for n, r in zip(names, from_res)]
        return values, names

    def compute_well_equations(self, wellSol0: Any, wellSol: Any, resmodel: Any,
                               q_s: Sequence[Any], bh: Any, packed: Any,
                               dt: float, iteration: int):
        weqs, q_mass, mix_s, status, cstatus, q_vol = \
            compute_well_contributions_single_well(self, wellSol, resmodel,
                                                   q_s, bh, packed)
        ctrl_eq = setup_well_control_equations_single_well(
            self, wellSol0, wellSol, bh, q_s, status, mix_s, resmodel)
        cq_s_db = np.column_stack([np.asarray(value(q), dtype=float).ravel()
                                   for q in q_mass])
        wellSol.cqs = cq_s_db/resmodel.get_surface_densities()
        wellSol.cstatus = cstatus
        wellSol.status = status
        return weqs, ctrl_eq, [], [], q_mass, q_vol, wellSol

    def compute_component_contributions(self, wellSol0, wellSol, resmodel, q_s,
                                        bh, packed, q_mass, q_vol, dt, iteration):
        return resmodel.get_extra_well_contributions(
            self, wellSol0, wellSol, q_s, bh, packed, q_mass, q_vol, dt,
            iteration)

    def is_injector(self) -> bool:
        return self.W.sign > 0

    def get_well_equation_names(self, resmodel: Any):
        names = [f"{name}Wells" for name in resmodel.get_long_phase_names()]
        return names, ["perf"]*len(names)

    def update_connection_pressure_drop(self, wellSol0, wellSol, model, q_s,
                                        bhp, packed, dt, iteration):
        """Hydrostatic drop along the wellbore, once per timestep (iteration 1)."""
        if iteration != 1 or not self.doUpdatePressureDrop:
            return wellSol
        p = packed.pressure
        rho = np.column_stack([np.asarray(value(r), dtype=float).ravel()
                               for r in packed.rho])
        dissolved = packed.dissolved
        if dissolved:
            dissolved = [[None if d is None else np.asarray(value(d), dtype=float)
                          for d in row] for row in dissolved]
        num_ph = int(np.count_nonzero(model.get_active_phases()))
        rho_s = model.get_surface_densities()
        b = np.column_stack(phase_densities_to_bfactor(
            [rho[:, i] for i in range(rho.shape[1])],
            [np.asarray(rho_s).ravel(order="F")[i] for i in range(rho.shape[1])],
            dissolved))
        w = self.W
        topo = _topology(w)
        qs = np.array(wellSol.cqs, dtype=float, copy=True)
        qs[qs == 0] = w.sign*ZERO_RATE
        C = self.wb2in(w, topo)
        wbqs = np.abs(spla.spsolve(C.tocsc(), qs).reshape(qs.shape))
        wbqst = np.sum(wbqs, axis=1)
        zi = wbqst == 0
        if np.any(zi):
            wbqs[zi, :] = np.asarray(w.compi, dtype=float)
            wbqst[zi] = np.sum(wbqs[zi, :], axis=1)
        mixs = wbqs/wbqst[:, None]
        vol_rat = self.comp_vol_rat(mixs, p, b, model)
        rho_mix = (mixs @ np.asarray(rho_s)[0])/vol_rat
        dpt = np.concatenate([[0.0], np.asarray(w.dZ, dtype=float).ravel()])
        dz = np.diff(dpt)
        g = np.linalg.norm(gravity())
        ddp = g*rho_mix*dz
        wellSol.cdp = _accumulate_along_topology(ddp, topo, wellSol.name)
        return wellSol

    def update_connection_pressure_drop_state(self, model, wellSol, rho_res,
                                              rho_well, mob_res=None):
        if not self.doUpdatePressureDrop:
            return wellSol
        w = self.W
        topo = _topology(w)
        WI = np.asarray(w.WI, dtype=float).reshape(-1, 1)
        if w.sign < 1 and getattr(wellSol, "ComponentTotalFlux", None) is not None \
                and np.any(wellSol.ComponentTotalFlux):
            q_vol = np.sum(wellSol.flux, axis=1)
            q_mass = np.sum(wellSol.ComponentTotalFlux, axis=1)
        else:
            sgn = w.sign if w.sign != 0 else 1
            if sgn == -1 and mob_res is not None and np.size(mob_res):
                q_vol = WI*mob_res
            else:
                q_vol = WI*np.asarray(w.compi, dtype=float)
            q_mass = np.sum(q_vol*rho_res, axis=1)
            q_vol = np.sum(q_vol, axis=1)
        if self.allowCrossflow and wellSol.sign != 0:
            x_flow = np.sign(q_mass) != wellSol.sign
            x_flow = x_flow & ~np.all(x_flow)
            q_mass = np.where(x_flow, 0.0, q_mass)
            q_vol = np.where(x_flow, 0.0, q_vol)
        is_shut = abs(np.sum(q_mass)) < SHUT_MASS_RATE or not wellSol.status
        g = np.linalg.norm(model.gravity)
        if self.simplePressureDrop:
            if is_shut:
                T = np.maximum(WI, SHUT_WELL_TRANSMISSIBILITY)
                q_approx = T*mob_res
                rho_mix = np.sum(q_approx*rho_res, axis=1)/np.sum(q_approx, axis=1)
            else:
                rho_mix = np.sum(q_mass)/np.sum(q_vol)
            cdp = g*np.asarray(w.dZ, dtype=float).ravel()*rho_mix
        else:
            C = self.wb2in(w, topo).tocsc()
            wb_mass = np.abs(spla.spsolve(C, q_mass))
            wb_vol = np.abs(spla.spsolve(C, q_vol))
            if is_shut:
                T = np.maximum(WI, SHUT_WELL_TRANSMISSIBILITY)
                q_approx = T*mob_res
                rho_mix = np.sum(q_approx*rho_res, axis=1)/np.sum(q_approx, axis=1)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rho_mix = wb_mass/wb_vol
                rho_mix[np.isnan(rho_mix)] = 0.0
                for i in range(rho_mix.size - 2, -1, -1):
                    if rho_mix[i] < TINY_MIXTURE_DENSITY:
                        rho_mix[i] = rho_mix[i + 1]
            bad = (rho_mix == 0) | ~np.isfinite(rho_mix)
            if np.all(bad):
                warnings.warn(f"Unable to compute wellbore mixture density for "
                              f"well {wellSol.name}. Pressure drop may be incorrect.")
            else:
                rho_mix[bad] = np.mean(rho_mix[~bad])
            dpt = np.concatenate([[0.0], np.asarray(w.dZ, dtype=float).ravel()])
            ddp = g*rho_mix*np.diff(dpt)
            cdp = _accumulate_along_topology(ddp, topo, wellSol.name)
        wellSol.cdp = cdp
        return wellSol

    def update_limits(self, wellSol0, wellSol, model, q_s, bhp, *args):
        """Switch control when the well breaches a limit; `(q_s, bhp, ws, ok)`."""
        within_limits = True
        if not self.allowControlSwitching or not getattr(self.W, "status", True):
            return q_s, bhp, wellSol, within_limits
        lims = self.W.lims
        qs_double = np.array([float(np.ravel(value(q))[0]) for q in q_s])
        qs_t = float(np.sum(qs_double))
        act_ph = model.get_active_phases()
        gas_ix = int(np.sum(act_ph[:3])) - 1
        oil_ix = int(np.sum(act_ph[:2])) - 1
        wat_ix = 0
        bhp_value = float(np.ravel(value(bhp))[0])
        if isinstance(lims, dict):
            if self.is_injector():
                modes = ["bhp", "rate", "vrat"]
                lims = self.set_missing_limits(lims, modes, np.inf)
                if not np.isfinite(lims["vrat"]):
                    lims["vrat"] = -np.inf
                flags = [bhp_value > lims["bhp"], qs_t > lims["rate"],
                         qs_t < lims["vrat"]]
            else:
                modes = ["bhp", "orat", "lrat", "grat", "wrat", "vrat"]
                lims = self.set_missing_limits(lims, modes, -np.inf)
                if not np.isfinite(lims["vrat"]):
                    lims["vrat"] = np.inf
                q_w = qs_double[wat_ix] if model.water else 0.0
                q_o = qs_double[oil_ix] if model.oil else 0.0
                q_g = qs_double[gas_ix] if model.gas else 0.0
                q_sl = 0.0
                flags = [bhp_value < lims["bhp"], q_o < lims["orat"],
                         q_w + q_o < lims["lrat"], q_g + q_sl < lims["grat"],
                         q_w < lims["wrat"], qs_t > lims["vrat"]]
        else:
            modes, flags = [], []
            assert lims is None or np.isinf(lims)
        checked = [(m, f) for m, f in zip(modes, flags) if m != wellSol.type]
        violated = next((m for m, f in checked if f), None)
        if violated is not None:
            within_limits = False
            print(f"Well {wellSol.name}: Control mode changed from "
                  f"{wellSol.type} to {violated}.")
            wellSol.type = violated
            wellSol.val = lims[violated]
            v = wellSol.val
            q_s = list(q_s)
            if wellSol.type == "bhp":
                bhp = assign_value(bhp, v, 0)
            elif wellSol.type == "rate":
                compi = np.asarray(self.W.compi, dtype=float).ravel()
                q_s = [assign_value(q, v*compi[ix], 0) for ix, q in enumerate(q_s)]
            elif wellSol.type == "orat":
                q_s[oil_ix] = assign_value(q_s[oil_ix], v, 0)
            elif wellSol.type == "wrat":
                q_s[wat_ix] = assign_value(q_s[wat_ix], v, 0)
            elif wellSol.type == "grat":
                q_s[gas_ix] = assign_value(q_s[gas_ix], v, 0)
        return q_s, bhp, wellSol, within_limits

    def update_well_sol(self, wellSol: Any, variables: Sequence[str],
                        dx: Sequence[Any], resmodel: Any) -> Any:
        """Apply this well's increments, keeping bhp on its side of a limit."""
        variables, dx = list(variables), list(dx)
        is_bhp = [v.lower() == "bhp" for v in variables]
        if any(is_bhp):
            k = is_bhp.index(True)
            dv = np.asarray(value(dx[k]), dtype=float)
            dv, _ = self.limit_update_relative(dv, np.atleast_1d(wellSol.bhp),
                                               self.dpMaxRel)
            dv, _ = self.limit_update_absolute(dv, self.dpMaxAbs)
            dv = float(np.ravel(dv)[0])
            lims = self.W.lims
            has_lim = isinstance(lims, dict) and "bhp" in lims
            if has_lim:
                lim = lims["bhp"]
                dist = wellSol.bhp - lim
            wellSol.bhp = wellSol.bhp + dv
            tol = 0.1*BARSA
            if has_lim and np.sign(wellSol.bhp - lim) != np.sign(dist) \
                    and abs(dist) > 2*tol:
                wellSol.bhp = lim + tol*np.sign(dist)
            variables = [v for v, b in zip(variables, is_bhp) if not b]
            dx = [d for d, b in zip(dx, is_bhp) if not b]
        is_mf = [v.lower() == "well_massfractions" for v in variables]
        if any(is_mf):
            k = is_mf.index(True)
            wellSol, _, _ = self.update_state_from_increment(
                wellSol, dx[k], None, "well_massfractions")
            wellSol = self.cap_property(wellSol, "well_massfractions", 0, 1)
            wellSol.massfractions = wellSol.massfractions/np.sum(wellSol.massfractions)
            variables = [v for v, m in zip(variables, is_mf) if not m]
            dx = [d for d, m in zip(dx, is_mf) if not m]
        names, from_res = self.get_extra_primary_variable_names(resmodel)
        lowered = [n.lower() for n in names]
        for i, name in enumerate(variables):
            # One well's increment for a scalar field (`qOs`) arrives as a
            # one-element array; MRST's `wellSol.qOs` stays a scalar.
            dxi = np.asarray(value(dx[i]), dtype=float)
            if dxi.size == 1 and np.ndim(getattr(wellSol, name, 0.0)) == 0:
                dx[i] = float(dxi.ravel()[0])
            if name.lower() in lowered and from_res[lowered.index(name.lower())]:
                wellSol, _, _ = resmodel.update_state_from_increment(
                    wellSol, dx[i], None, name)
            else:
                wellSol, _, _ = self.update_state_from_increment(
                    wellSol, dx[i], None, name)
        return wellSol

    def update_well_sol_after_step(self, resmodel: Any, wellSol: Any,
                                   wellSol0: Any) -> Tuple[Any, bool]:
        """Shut a rate-controlled well whose flow reversed."""
        well_shut = False
        w = self.W
        if not w.status:
            return wellSol, well_shut
        ssg = np.sign(self.get_total_rate([wellSol])[0])
        closed = w.sign != ssg and w.sign != 0
        closed = closed and w.type.lower() != "bhp" and w.val != 0
        if closed and not self.allowSignChange and self.allowControlSwitching:
            if self.verbose:
                print(f"Well {w.name} shut down.")
            wellSol.status = False
            well_shut = True
        switched = wellSol.type.lower() != wellSol0.type.lower()
        if switched and self.verbose and not (
                wellSol0.type.lower() == "resv_history"
                and wellSol.type.lower() == "resv"):
            print(f"Step complete: Well {w.name} was switched from "
                  f"{wellSol0.type} to {wellSol.type} controls.")
        return wellSol, well_shut

    def get_variable_field(self, name: str, throw_error: bool = True):
        fields = {"bhp": "bhp", "qos": "qOs", "qgs": "qGs", "qws": "qWs",
                  "qss": "qSs", "well_temperature": "T", "t": "T"}
        key = name.lower()
        if key in fields:
            return fields[key], ":"
        if key == "well_massfractions":
            return "massfractions", ":"
        return super().get_variable_field(name, throw_error)

    def ensure_well_sol_consistency(self, ws: Any) -> Any:
        return ws

    def set_missing_limits(self, lims: Dict[str, float], modes: Sequence[str],
                           val: float) -> Dict[str, float]:
        lims = dict(lims)
        for mode in modes:
            lims.setdefault(mode, val)
        return lims

    @staticmethod
    def wb2in(w: Any, topo: np.ndarray = None) -> sp.csr_matrix:
        """Map wellbore fluxes to inflows: `C`, with `C wb = inflow`."""
        topo = _topology(w) if topo is None else topo
        conn = topo[1:, :]
        nconn = conn.shape[0]
        nperf = np.asarray(w.cells).size
        if nconn + 1 != nperf:
            warnings.warn(
                f"Mismatch between connection count ({nconn + 1}) and "
                f"perforation count ({nperf}). Well modelDoes not appear to "
                f"be a tree.")
        idx = np.arange(nperf)
        ii = np.concatenate([idx, conn[:, 0] - 1])
        jj = np.concatenate([idx, conn[:, 1] - 1])
        vv = np.concatenate([np.ones(nperf), -np.ones(nconn)])
        keep = (ii >= 0) & (jj >= 0)
        return sp.csr_matrix((vv[keep], (ii[keep], jj[keep])), shape=(nperf, nperf))

    @staticmethod
    def comp_vol_rat(mixs: np.ndarray, p: Any, b: np.ndarray, model: Any) -> np.ndarray:
        """Reservoir volume per surface volume of the wellbore mixture."""
        x = np.array(mixs, dtype=float, copy=True)
        dg = bool(getattr(model, "disgas", False))
        vo = bool(getattr(model, "vapoil", False))
        if dg or vo:
            isgas = model.get_variable_field("sg")[1]
            isoil = model.get_variable_field("so")[1]
            both = [isgas, isoil]
            g = mixs[:, isgas]
            o = mixs[:, isoil]
            pv = np.asarray(value(p), dtype=float)
            if dg:
                fn = model.fluid.rsSat
                rs_max = (fn[0] if isinstance(fn, (list, tuple)) else fn)(pv)
            else:
                rs_max = 0.0
            from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
                ThreePhaseBlackOilModel,
            )
            if isinstance(model, ThreePhaseBlackOilModel):
                if vo:
                    fn = model.fluid.rvSat
                    rv_max = (fn[0] if isinstance(fn, (list, tuple)) else fn)(pv)
                else:
                    rv_max = 0.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    gor = np.abs(g/o)
                    ogr = np.abs(o/g)
                gor[np.isnan(gor)] = np.inf
                ogr[np.isnan(ogr)] = np.inf
                rs = np.minimum(rs_max, gor)
                rv = np.minimum(rv_max, ogr)
                d = 1 - rs*rv
                x[:, isgas] = (x[:, isgas] - rs*o)/d
                x[:, isoil] = (x[:, isoil] - rv*g)/d
                x[:, both] = x[:, both]*(x[:, both] > 0)
            else:
                x[:, isgas] = x[:, isgas] - rs_max*o
                x[:, isgas] = x[:, isgas]*(x[:, isgas] > 0)
        return np.sum(x/b, axis=1)

    @staticmethod
    def get_total_rate(sol: Sequence[Any]) -> np.ndarray:
        qt = np.zeros(len(sol))
        if not sol:
            return qt
        types = [t for t in ("qWs", "qOs", "qGs", "qSs") if hasattr(sol[0], t)]
        for w, row in enumerate(sol):
            for t in types:
                x = getattr(row, t)
                if x is not None and np.size(x):
                    qt[w] += float(np.ravel(value(x))[0])
        return qt

    updateWell = update_well
    validateWellSol = validate_well_sol
    getVariableCounts = get_variable_counts
    getExtraPrimaryVariableNames = get_extra_primary_variable_names
    getExtraEquationNames = get_extra_equation_names
    getExtraPrimaryVariables = get_extra_primary_variables
    computeWellEquations = compute_well_equations
    computeComponentContributions = compute_component_contributions
    isInjector = is_injector
    getWellEquationNames = get_well_equation_names
    updateConnectionPressureDrop = update_connection_pressure_drop
    updateConnectionPressureDropState = update_connection_pressure_drop_state
    updateLimits = update_limits
    updateWellSol = update_well_sol
    updateWellSolAfterStep = update_well_sol_after_step
    getVariableField = get_variable_field
    ensureWellSolConsistency = ensure_well_sol_consistency
    setMissingLimits = set_missing_limits
    compVolRat = comp_vol_rat
    getTotalRate = get_total_rate


def _topology(w: Any) -> np.ndarray:
    """`W.topo`, or MRST's default chain `[(0:n-1)', (1:n)']` (1-based)."""
    topo = getattr(w, "topo", None)
    if topo is not None:
        return np.asarray(topo, dtype=np.int64)
    nperf = np.asarray(w.cells).size
    return np.column_stack([np.arange(nperf), np.arange(1, nperf + 1)])


def _accumulate_along_topology(ddp: np.ndarray, topo: np.ndarray,
                               name: str) -> np.ndarray:
    """`cdp(child) = cdp(parent) + ddp`, walked until every node is reached."""
    cdp = np.full(ddp.shape, np.nan)
    cdp[0] = ddp[0]
    its = 0
    while np.any(np.isnan(cdp)) and its < TOPOLOGY_ITERATIONS:
        its += 1
        for cnr in range(1, cdp.size):
            cdp[topo[cnr, 1] - 1] = cdp[topo[cnr, 0] - 1] + ddp[cnr]
    if its == TOPOLOGY_ITERATIONS:
        raise ValueError(f"Problem with topology for well: {name}. Segments "
                         f"appear not to be connected")
    return cdp


__all__ = ["SimpleWell"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _well(nperf: int = 3, sign: int = -1, ctrl: str = "bhp", lims=None):
        return SimpleNamespace(cells=np.arange(nperf), WI=np.full(nperf, 1e-12),
                               dZ=np.arange(nperf, dtype=float), sign=sign,
                               compi=np.array([1.0, 0.0]), status=True,
                               type=ctrl, val=100e5, name="W1", lims=lims,
                               cstatus=None)

    def test_default_topology_is_a_chain_from_the_top():
        C = SimpleWell.wb2in(_well(3)).toarray()
        np.testing.assert_allclose(C, [[1, -1, 0], [0, 1, -1], [0, 0, 1]])
        print("ok test_default_topology_is_a_chain_from_the_top")

    def test_pressure_drop_accumulates_down_the_chain():
        topo = _topology(_well(3))
        cdp = _accumulate_along_topology(np.array([1.0, 2.0, 3.0]), topo, "W1")
        np.testing.assert_allclose(cdp, [1.0, 3.0, 6.0])
        print("ok test_pressure_drop_accumulates_down_the_chain")

    def test_bhp_increment_is_kept_on_its_side_of_the_limit():
        well = SimpleWell(_well(lims={"bhp": 50e5}))
        ws = SimpleNamespace(bhp=60e5)
        resmodel = SimpleNamespace(get_extra_well_primary_variable_names=lambda: [])
        well.update_well_sol(ws, ["bhp"], [np.array([-20e5])], resmodel)
        np.testing.assert_allclose(ws.bhp, 50e5 + 0.1*BARSA)
        print("ok test_bhp_increment_is_kept_on_its_side_of_the_limit")

    def test_a_producer_below_its_bhp_limit_switches_to_bhp():
        well = SimpleWell(_well(ctrl="orat", lims={"bhp": 90e5}))
        ws = SimpleNamespace(type="orat", val=-1.0, name="W1")
        model = SimpleNamespace(water=True, oil=True, gas=False,
                                get_active_phases=lambda: np.array([True, True, False]))
        q_s, bhp, ws, ok = well.update_limits(
            None, ws, model, [np.array([-0.5]), np.array([-1.0])], np.array([80e5]))
        assert not ok and ws.type == "bhp" and ws.val == 90e5
        np.testing.assert_allclose(bhp, 90e5)
        print("ok test_a_producer_below_its_bhp_limit_switches_to_bhp")

    def test_total_rate_sums_the_phase_rates():
        ws = SimpleNamespace(qWs=-1.0, qOs=-2.0)
        np.testing.assert_allclose(SimpleWell.get_total_rate([ws]), [-3.0])
        print("ok test_total_rate_sums_the_phase_rates")

    test_default_topology_is_a_chain_from_the_top()
    test_pressure_drop_accumulates_down_the_chain()
    test_bhp_increment_is_kept_on_its_side_of_the_limit()
    test_a_producer_below_its_bhp_limit_switches_to_bhp()
    test_total_rate_sums_the_phase_rates()
    print("\nAll tests passed!")
