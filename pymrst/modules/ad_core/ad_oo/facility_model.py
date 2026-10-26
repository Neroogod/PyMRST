"""MRST ``ad-core/models/facilities/FacilityModel.m``.

The facility: the set of well models attached to a reservoir model. It owns
the wells' primary variables (surface rates and bottom-hole pressure, plus
anything a well or the reservoir adds), assembles every active well's
equations and source terms through its well models, applies Newton
increments to `wellSol`, and reports well convergence.

`wellSol` is a list of records, one per well. Well and variable maps are
0-based; `wellMap.extraMap` holds 1-based variable indices with 0 as padding,
which is MRST's own layout for that table.
"""

from __future__ import annotations

from copy import copy

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.ad_oo.physical_model import (
    PhysicalModel,
    StepReport,
    merge_options,
)
from pymrst.modules.ad_core.models.facilities.perforation_properties import (
    pack_perforation_properties,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_core.utils.assign_well_values_from_control import (
    assign_well_values_from_control,
)
from pymrst.modules.ad_core.utils.check_well_convergence import (
    get_convergence_values_wells,
)
from pymrst.modules.ad_core.utils.get_perforation_to_well_mapping import (
    get_perforation_to_well_mapping,
)
from pymrst.utils.adi import ad_matmul, value, vertcat_adi
from pymrst.utils.unique_stable import unique_stable
from pymrst.utils.units import BARSA, DAY

TOLERANCE_WELL_BHP: float = 1*BARSA
TOLERANCE_WELL_RATE: float = 1/DAY
TOLERANCE_WELL_MS: float = 1e-6


class FacilityModel(PhysicalModel):
    """MRST `FacilityModel`: wells as a submodel of a reservoir model."""

    def __init__(self, reservoir_model: Any = None, **options: Any):
        super().__init__(None)
        self.WellModels: List[Any] = []
        self.FacilityFlowDiscretization = None
        self.toleranceWellBHP = TOLERANCE_WELL_BHP
        self.toleranceWellRate = TOLERANCE_WELL_RATE
        self.toleranceWellMS = TOLERANCE_WELL_MS
        self.ReservoirModel = reservoir_model
        self.VFPTablesInjector: List[Any] = []
        self.VFPTablesProducer: List[Any] = []
        self.primaryVariableSet = "standard"
        self.setWellTargets = True
        self.addedPrimaryVarNames: List[str] = []
        self.addedPrimaryVarNamesIsFromResModel: List[bool] = []
        self.addedEquationNames: List[str] = []
        self.addedEquationTypes: List[str] = []
        merge_options(self, options)
        self.WellModels = []

    # -- groupings ----------------------------------------------------------

    def setup_state_function_groupings(self, use_defaults: Optional[bool] = None):
        if use_defaults is None:
            use_defaults = self.FacilityFlowDiscretization is None
        if use_defaults:
            from pymrst.modules.ad_core.ad_oo.statefunctions.facility_flow_discretization import (
                FacilityFlowDiscretization,
            )
            self.FacilityFlowDiscretization = FacilityFlowDiscretization(self)
        return self

    def remove_state_function_groupings(self):
        self.FacilityFlowDiscretization = None
        return self

    # -- property access ----------------------------------------------------

    def get_prop(self, state: State, name: str) -> Any:
        if name.lower() in ("surfacerates", "q_s"):
            names = self.ReservoirModel.get_phase_names()
            return [self.get_prop(state, f"q{x}s") for x in names]
        fs = state.get("FacilityState") if isinstance(state, dict) else None
        if fs is not None:
            found = _index_ci(fs["names"], name)
            if found is not None:
                return fs["primaryVariables"][found]
        field, index = self.get_variable_field(name, False)
        if fs is not None and field is not None:
            found = _index_ci(fs["names"], field)
            if found is not None:
                return fs["primaryVariables"][found]
        well_sol = state["wellSol"] if isinstance(state, dict) else None
        if field is not None and well_sol and hasattr(well_sol[0], field):
            rows = [ws for ws in well_sol if ws.status]
            values = [np.atleast_1d(getattr(ws, field)) for ws in rows]
            if index != ":":
                values = [np.atleast_2d(v)[:, index] for v in values]
            return np.concatenate(values) if values else np.zeros(0)
        return super().get_prop(state, name)

    def get_variable_field(self, name: str, throw_error: bool = True):
        if name == "bhp":
            return name, ":"
        if name.startswith("q") and self.ReservoirModel is not None:
            qnames = [f"q{x}s" for x in self.ReservoirModel.get_phase_names()]
            if name in qnames:
                return name, ":"
        return super().get_variable_field(name, throw_error)

    # -- wells --------------------------------------------------------------

    def setup_wells(self, W: Sequence[Any], wellmodels: Optional[Sequence[Any]] = None):
        """Instantiate one well model per well (or refresh their controls)."""
        from pymrst.modules.ad_core.ad_oo.multisegment_well import MultisegmentWell
        from pymrst.modules.ad_core.ad_oo.simple_well import SimpleWell
        W = list(W or [])
        nw = len(W)
        if self.get_number_of_wells() == 0:
            pvars, from_res, eqnames, eqtypes = [], [], [], []
            self.WellModels = []
            for i, w in enumerate(W):
                if w.status:
                    assert np.size(w.val) == 1, (
                        f"Bad controls for well {i + 1}: One value must be "
                        f"specified for .val")
                if wellmodels is None:
                    if getattr(w, "multisegment", None):
                        raise NotImplementedError(
                            "PyMRST's own multisegment wells (`Well.multisegment`) "
                            "are not AD-OO wells; build MRST's with convert2MSWell")
                    if getattr(w, "isMS", False):
                        wm = MultisegmentWell(w)
                    else:
                        wm = SimpleWell(w)
                else:
                    wm = wellmodels[i]
                wm.dsMaxAbs = self.ReservoirModel.dsMaxAbs
                wm.dpMaxAbs = self.ReservoirModel.dpMaxAbs
                vfp_ix = getattr(w, "vfp_index", None)
                if vfp_ix is not None and vfp_ix > 0:
                    tables = (self.VFPTablesInjector if wm.is_injector()
                              else self.VFPTablesProducer)
                    wm.VFPTable = tables[vfp_ix - 1]
                names, from_resmodel = wm.get_extra_primary_variable_names(
                    self.ReservoirModel)
                pvars += names
                from_res += from_resmodel
                enames, etypes = wm.get_extra_equation_names(self.ReservoirModel)
                eqnames += list(enames)
                eqtypes += list(etypes)
                self.WellModels.append(wm)
            _, keep = _unique_stable_index(pvars)
            self.addedPrimaryVarNames = [pvars[k] for k in keep]
            self.addedPrimaryVarNamesIsFromResModel = [from_res[k] for k in keep]
            _, keep = _unique_stable_index(eqnames)
            self.addedEquationNames = [eqnames[k] for k in keep]
            self.addedEquationTypes = [eqtypes[k] for k in keep]
        else:
            assert self.get_number_of_wells() == nw, (
                "Number of wells in facility model has changed during simulation")
            self.WellModels = [copy(wm).update_well(w) for wm, w in zip(self.WellModels, W)]
        return self

    def set_reservoir_model(self, resmodel: Any):
        self.ReservoirModel = resmodel
        self.AutoDiffBackend = resmodel.AutoDiffBackend
        return self

    def get_well_struct(self, subs: Any = None) -> List[Any]:
        models = self.WellModels if subs is None else [self.WellModels[i] for i in subs]
        return [wm.W for wm in models]

    def get_number_of_active_wells(self, well_sol: Sequence[Any]) -> int:
        return int(np.count_nonzero(self.get_well_status_mask(well_sol)))

    def get_number_of_wells(self) -> int:
        return len(self.WellModels)

    def get_indices_of_active_wells(self, well_sol: Sequence[Any]) -> np.ndarray:
        return np.flatnonzero(self.get_well_status_mask(well_sol))

    def get_well_status_mask(self, well_sol: Sequence[Any]) -> np.ndarray:
        act_model = np.array([bool(wm.W.status) for wm in self.WellModels], dtype=bool)
        act_ws = np.array([bool(ws.status) for ws in (well_sol or [])], dtype=bool)
        return act_model & act_ws

    # -- primary variables --------------------------------------------------

    def init_state_ad(self, state: State, variables, names, origin) -> State:
        variables, names = list(variables), list(names)
        mass_fractions = None
        if "well_massfractions" in names:
            k = names.index("well_massfractions")
            mass_fractions = variables.pop(k)
            names.pop(k)
        phase_names = self.ReservoirModel.get_phase_names()
        active = [ws for ws in state["wellSol"] if ws.status]
        rates = []
        for x in phase_names:
            fn = f"q{x}s"
            if fn in names:
                rates.append(variables[names.index(fn)])
            else:
                rates.append(np.array([getattr(ws, fn) for ws in active], dtype=float))
        state["FacilityState"] = {"primaryVariables": variables,
                                  "surfacePhaseRates": rates, "names": names}
        if mass_fractions is not None:
            ncomp = self.ReservoirModel.get_number_of_components()
            length = np.size(value(mass_fractions))//(ncomp - 1)
            fill, fractions = 1.0, []
            for i in range(ncomp - 1):
                v = mass_fractions[length*i:length*(i + 1)]
                fill = fill - v
                fractions.append(v)
            fractions.append(fill)
            state["FacilityState"]["massfractions"] = fractions
        return super().init_state_ad(state, [], [], [])

    def get_primary_variables(self, state: State):
        if "wellSol" in state:
            variables, names, _ = self.get_basic_primary_variables(state["wellSol"])
            return variables, names, [type(self).__name__]*len(variables)
        return [], [], []

    def validate_model(self, *args: Any) -> "FacilityModel":
        if args:
            wells = _field(args[0], "W")
            self.setup_wells([] if wells is None else
                             (wells if isinstance(wells, (list, tuple)) else [wells]))
        return self

    def get_primary_variable_names(self) -> List[str]:
        return self.get_basic_primary_variable_names() + list(self.addedPrimaryVarNames)

    def get_basic_primary_variable_names(self) -> List[str]:
        pvs = self.primaryVariableSet.lower()
        if pvs in ("standard", "explicit"):
            return [f"q{x}s" for x in self.ReservoirModel.get_phase_names()] + ["bhp"]
        if pvs == "bhp":
            return ["bhp"]
        if pvs == "bhp_massfractions":
            return ["bhp", "well_massfractions"]
        if pvs == "none":
            return []
        raise ValueError(f"Invalid primary variable set {self.primaryVariableSet} "
                         f"for FacilityModel")

    def get_basic_primary_variables(self, well_sol: Sequence[Any]):
        if self.get_number_of_active_wells(well_sol) == 0:
            return [], [], {"isBHP": np.zeros(0, dtype=bool),
                            "isRate": np.zeros(0, dtype=bool)}
        active = self.get_well_status_mask(well_sol)
        rows = [ws for ws, a in zip(well_sol, active) if a]
        names = self.get_basic_primary_variable_names()
        variables = []
        for name in names:
            if name == "well_massfractions":
                comp = np.vstack([np.atleast_1d(ws.massfractions) for ws in rows])
                variables.append(comp[:, :-1].reshape(-1, order="F"))
            elif name == "well_temperature":
                variables.append(np.array([ws.T for ws in rows], dtype=float))
            else:
                variables.append(np.array([getattr(ws, name) for ws in rows],
                                          dtype=float).ravel())
        is_bhp = np.zeros(len(names), dtype=bool)
        if names:
            is_bhp[-1] = True
        return variables, names, {"isBHP": is_bhp, "isRate": ~is_bhp}

    def get_extra_primary_variables(self, well_sol: Sequence[Any]):
        act_ix = self.get_indices_of_active_wells(well_sol)
        nw = act_ix.size
        if nw == 0:
            return [], [], np.zeros((0, 0), dtype=np.int64)
        names = list(self.addedPrimaryVarNames)
        nv = len(names)
        rows: List[List[Any]] = [[None]*nv for _ in range(nw)]
        wellmap = np.zeros((nw, nv), dtype=np.int64)
        lowered = [n.lower() for n in names]
        for i, wno in enumerate(act_ix):
            vals, vnames = self.WellModels[wno].get_extra_primary_variables(
                well_sol[wno], self.ReservoirModel)
            for j, (v, n) in enumerate(zip(vals, vnames)):
                ix = lowered.index(n.lower())
                wellmap[i, j] = ix + 1
                rows[i][ix] = v
        variables = [np.concatenate([np.atleast_1d(rows[i][j]) for i in range(nw)
                                     if rows[i][j] is not None])
                     for j in range(nv)]
        return variables, names, wellmap

    def get_all_primary_variables(self, well_sol: Sequence[Any]):
        basic, bnames, well_map = self.get_basic_primary_variables(well_sol)
        extra, enames, extra_map = self.get_extra_primary_variables(well_sol)
        well_map = {
            "isBHP": np.concatenate([well_map["isBHP"], np.zeros(len(extra), dtype=bool)]),
            "isRate": np.concatenate([well_map["isRate"], np.zeros(len(extra), dtype=bool)]),
            "extraMap": extra_map,
        }
        return basic + extra, bnames + enames, well_map

    # -- assembly -----------------------------------------------------------

    def get_well_contributions(self, well_sol0, well_sol, wellvars, well_map,
                               p, mob, rho, dissolved, comp, dt, iteration):
        """Every active well's equations and its source terms in the cells."""
        if np.isnan(iteration) or iteration < 0:
            import warnings
            warnings.warn("Iteration number is not passed on to well model,"
                          "this may indicate wellbore pressure-drop will never "
                          "be updated")
        act_ix = self.get_indices_of_active_wells(well_sol)
        nw = act_ix.size
        all_base, all_ctrl, all_vol, all_mass, all_comp = [], [], [], [], []
        enames = list(self.addedEquationNames)
        etypes = list(self.addedEquationTypes)
        res_model = self.ReservoirModel
        cnames = res_model.get_component_names()
        ncomp = len(cnames)
        assert ncomp == len(comp), (
            "Number of input components must match length of getComponentNames!")
        n_extra = len(enames)
        assert len(etypes) == n_extra
        all_extra: List[List[Any]] = [[None]*n_extra for _ in range(nw)]
        added = list(self.addedPrimaryVarNames)
        varmaps = [self.get_well_variable_map(v, well_sol) for v in added]
        is_bh = np.asarray(well_map["isBHP"], dtype=bool)
        is_q = np.asarray(well_map["isRate"], dtype=bool)
        emap = well_map["extraMap"]
        bhp = wellvars[int(np.flatnonzero(is_bh)[0])]
        q_well = [v for v, q in zip(wellvars, is_q) if q]
        extra_vars = [v for v, b, q in zip(wellvars, is_bh, is_q) if not (b or q)]
        well_sol = list(well_sol)
        basenames, basetypes = self.WellModels[0].get_well_equation_names(res_model)
        for i, wno in enumerate(act_ix):
            wm = self.WellModels[wno]
            ws, ws0 = well_sol[wno], well_sol0[wno]
            packed = pack_perforation_properties(
                wm.W, p, mob, rho, dissolved, comp, extra_vars, added, varmaps,
                emap, i)
            qw = [q[i] for q in q_well]
            bh = bhp[i]
            ws = wm.update_connection_pressure_drop(ws0, ws, res_model, qw, bh,
                                                    packed, dt, iteration)
            qw, bh, ws, ok = wm.update_limits(ws0, ws, res_model, qw, bh, packed,
                                              dt, iteration)
            if not ok:
                bhp = _set_row(bhp, i, bh)
                q_well = [_set_row(q, i, qw[k]) for k, q in enumerate(q_well)]
            base, ctrl, extra_eqs, extra_names, mass, vol, ws = \
                wm.compute_well_equations(ws0, ws, res_model, qw, bh, packed, dt,
                                          iteration)
            comp_eqs, comp_src, comp_names, ws = wm.compute_component_contributions(
                ws0, ws, res_model, qw, bh, packed, mass, vol, dt, iteration)
            all_base.append(base)
            all_ctrl.append(ctrl)
            all_mass.append(mass)
            all_vol.append(vol)
            all_comp.append(comp_src)
            lowered = [n.lower() for n in enames]
            for eq, name in zip(list(extra_eqs) + list(comp_eqs),
                                list(extra_names) + list(comp_names)):
                all_extra[i][lowered.index(name.lower())] = eq
            well_sol[wno] = ws
        n_ph = int(np.count_nonzero(res_model.get_active_phases()))
        src_mass = [self.combine_cell_data(all_mass, ph) for ph in range(n_ph)]
        src_vol = [self.combine_cell_data(all_vol, ph) for ph in range(n_ph)]
        eqs = [self.combine_cell_data(all_base, ph) for ph in range(n_ph)]
        src_comp = [self.combine_cell_data(all_comp, c) for c in range(ncomp)]
        if n_extra > 0:
            for k in range(n_extra):
                parts = [all_extra[i][k] for i in range(nw) if all_extra[i][k] is not None]
                eqs.append(vertcat_adi(*parts) if parts else None)
            names = list(basenames) + enames
            types = list(basetypes) + etypes
        else:
            names, types = list(basenames), list(basetypes)
        ctrleq = vertcat_adi(*[_as_ad_column(c) for c in all_ctrl])
        wc = self.get_well_cells(act_ix)
        wc, (src_mass, src_vol) = self.handle_repeated_perforated_cells(
            wc, src_mass, src_vol)
        well_system = {"wellEquations": eqs, "names": names, "types": types,
                       "controlEquation": ctrleq}
        sources = {"phaseMass": src_mass, "phaseVolume": src_vol,
                   "components": src_comp, "sourceCells": wc}
        if res_model.extraWellSolOutput:
            well_sol = self.set_well_sol_statistics(well_sol, sources)
        return sources, well_system, well_sol

    def update_well_sol_after_step(self, well_sol, well_sol0):
        out = list(well_sol)
        for wno, wm in enumerate(self.WellModels):
            out[wno], _ = wm.update_well_sol_after_step(self.ReservoirModel,
                                                        out[wno], well_sol0[wno])
        return out

    def update_after_convergence(self, state0, state, dt, driving_forces):
        state["wellSol"] = self.update_well_sol_after_step(state["wellSol"],
                                                           state0["wellSol"])
        return state, None

    def get_well_cells(self, subs: Any = None) -> np.ndarray:
        models = self.WellModels if subs is None else [self.WellModels[i] for i in subs]
        cells = [np.asarray(wm.W.cells, dtype=np.int64).ravel() for wm in models]
        return np.concatenate(cells) if cells else np.zeros(0, dtype=np.int64)

    def get_active_well_cells(self, well_sol):
        active = self.get_well_status_mask(well_sol)
        cells = [np.asarray(wm.W.cells, dtype=np.int64).ravel()
                 for wm, a in zip(self.WellModels, active) if a]
        wc = np.concatenate(cells) if cells else np.zeros(0, dtype=np.int64)
        p2w, _ = get_perforation_to_well_mapping(self.get_well_struct())
        p2w = np.asarray(p2w)[active[np.asarray(p2w, dtype=np.int64)]] if np.size(p2w) else p2w
        return wc, p2w

    def set_well_sol_statistics(self, ws, sources):
        """Reservoir rates, totals, cuts and ratios on each active well."""
        p2w, _ = get_perforation_to_well_mapping(self.get_well_struct())
        p2w = np.asarray(p2w, dtype=np.int64)
        active = self.get_well_status_mask(ws)
        p2w = p2w[active[p2w]] if p2w.size else p2w
        rm = self.ReservoirModel
        gind, oind, wind = rm.get_phase_index("G", "O", "W")
        src_res = [np.asarray(value(x), dtype=float).ravel() for x in sources["phaseVolume"]]
        qR = np.column_stack(src_res) if src_res else np.zeros((0, 0))
        if qR.shape[0] != p2w.size:
            return ws
        for i, row in enumerate(ws):
            if not active[i]:
                continue
            row.qTs = 0.0
            row.qTr = 0.0
            mine = p2w == i
            if rm.gas:
                row.qGr = float(src_res[gind][mine].sum())
                row.qTr += row.qGr
                row.qTs += row.qGs
            if rm.oil:
                row.qOr = float(src_res[oind][mine].sum())
                row.qTr += row.qOr
                row.qTs += row.qOs
            if rm.water:
                row.qWr = float(src_res[wind][mine].sum())
                row.qTr += row.qWr
                row.qTs += row.qWs
            # MATLAB's 0/0 is NaN and x/0 is Inf; Python floats raise.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = lambda a, b: np.float64(a)/np.float64(b)
                if rm.water and rm.oil:
                    row.wcut = ratio(row.qWs, row.qWs + row.qOs)
                if rm.gas:
                    row.gcut = ratio(row.qGs, row.qTs)
                if rm.oil:
                    row.ocut = ratio(row.qOs, row.qTs)
                if rm.gas and rm.oil:
                    row.gor = ratio(row.qGs, row.qOs)
            row.flux = qR[mine, :]
        return ws

    def update_well_sol(self, well_sol, problem, dx, driving_forces, rest_vars=None):
        """Route each well variable's increment to the well that owns it."""
        rest_vars = list(problem.primary_variables if rest_vars is None else rest_vars)
        if len(well_sol) == 0:
            return well_sol, rest_vars
        well_vars = self.get_primary_variable_names()
        n_vars = len(well_vars)
        act_ix = self.get_indices_of_active_wells(well_sol)
        nW = act_ix.size
        active_vars = np.zeros((nW, n_vars), dtype=bool)
        dx_well: List[List[Any]] = [[None]*n_vars for _ in range(nW)]
        for var_no, wf in enumerate(well_vars):
            if wf == "well_massfractions":
                c = np.asarray(value(dx[var_no]), dtype=float).reshape(nW, -1, order="F")
                c = np.column_stack([c, -c.sum(axis=1)])
                active_vars[:, var_no] = True
                rest_vars, _ = self.strip_vars(rest_vars, wf)
                for i in range(nW):
                    dx_well[i][var_no] = c[i, :]
            elif wf in rest_vars:
                dv = self.get_increment(dx, problem, wf)
                is_var_well = self.get_well_variable_map(wf, well_sol)
                for i in range(nW):
                    subs = is_var_well == i
                    if np.any(subs):
                        active_vars[i, var_no] = True
                        dx_well[i][var_no] = np.asarray(value(dv))[subs]
                rest_vars, _ = self.strip_vars(rest_vars, wf)
        well_sol = list(well_sol)
        for i, wno in enumerate(act_ix):
            act = active_vars[i]
            names = [v for v, a in zip(well_vars, act) if a]
            dxw = [dx_well[i][k] for k in np.flatnonzero(act)]
            well_sol[wno] = self.WellModels[wno].update_well_sol(
                well_sol[wno], names, dxw, self.ReservoirModel)
        return well_sol, rest_vars

    def get_well_variable_map(self, wf: str, ws: Sequence[Any]) -> np.ndarray:
        """Which active well (0-based) owns each row of variable `wf`."""
        act = self.get_well_status_mask(ws)
        is_res = self.addedPrimaryVarNamesIsFromResModel
        lowered = [n.lower() for n in self.addedPrimaryVarNames]
        if wf.lower() in lowered and is_res[lowered.index(wf.lower())]:
            counts = np.ones(int(act.sum()), dtype=np.int64)
        else:
            counts = np.array([wm.get_variable_counts(wf) for wm, a
                               in zip(self.WellModels, act) if a], dtype=np.int64)
        return np.repeat(np.arange(int(act.sum())), counts)

    def get_equations(self, state0, state, dt, driving_forces, *,
                      iteration=float("nan"), resOnly=False, **_):
        """The wells' own equations with the reservoir held fixed."""
        from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearizedProblem
        from pymrst.utils.adi import init_variables_adi
        well_sol, well_sol0 = state["wellSol"], state0["wellSol"]
        resmodel = self.ReservoirModel
        well_vars, primary_vars, well_map = self.get_all_primary_variables(well_sol)
        if not resOnly:
            well_vars = list(init_variables_adi(*well_vars))
        from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
            ThreePhaseBlackOilModel,
        )
        if isinstance(resmodel, ThreePhaseBlackOilModel):
            rs, rv = resmodel.get_props(state, "rs", "rv")
        else:
            rs = rv = None
        if "rho" not in state or "mob" not in state:
            resmodel.FacilityModel = self
            resmodel.extraStateOutput = True
            _, state = resmodel.get_equations(state0, state, dt, driving_forces,
                                              iteration=float("inf"), resOnly=True)
            assert "rho" in state, "Density missing from state!"
            assert "mob" in state, "Mobility missing from state!"
        p = resmodel.get_prop(state, "pressure")
        n_ph = state["rho"].shape[1]
        mob = [state["mob"][:, i] for i in range(n_ph)]
        rho = [state["rho"][:, i] for i in range(n_ph)]
        dissolution = resmodel.get_dissolution_matrix(rs, rv)
        src, wellsys, state["wellSol"] = self.get_well_contributions(
            well_sol0, well_sol, well_vars, well_map, p, mob, rho, dissolution,
            [], dt, iteration)
        eqs = list(wellsys["wellEquations"]) + [wellsys["controlEquation"]]
        names = list(wellsys["names"]) + ["closureWells"]
        types = list(wellsys["types"]) + ["well"]
        problem = LinearizedProblem.from_equations(eqs, types, names, primary_vars,
                                                   state, dt)
        return problem, state

    def update_state(self, state, problem, dx, driving_forces):
        if "wellSol" in state:
            state["wellSol"], _ = self.update_well_sol(state["wellSol"], problem, dx,
                                                       driving_forces)
            if self.setWellTargets:
                W = _field(driving_forces, "W")
                W = [] if W is None else (W if isinstance(W, (list, tuple)) else [W])
                ph = self.ReservoirModel.get_phase_indices()
                state["wellSol"] = assign_well_values_from_control(
                    self.ReservoirModel, state["wellSol"], W, ph[0], ph[1], ph[2])
        return state, None

    def validate_state(self, state):
        well_sol = state.get("wellSol") if isinstance(state, dict) else None
        if not well_sol or not hasattr(well_sol[0], "bhp"):
            state.pop("wellSol", None)
            from pymrst.modules.ad_core.utils.init_well_sol_ad import init_well_sol_ad
            state["wellSol"] = init_well_sol_ad(self.get_well_struct(),
                                                self.ReservoirModel, state)
        if self.primaryVariableSet.lower() == "bhp_massfractions":
            ncomp = self.ReservoirModel.get_number_of_components()
            for row in state["wellSol"]:
                row.massfractions = np.ones(ncomp)/ncomp
        for wno, wm in enumerate(self.WellModels):
            new_ws = wm.validate_well_sol(self.ReservoirModel, state["wellSol"][wno], state)
            for key, item in vars(new_ws).items():
                setattr(state["wellSol"][wno], key, item)
        return state

    def get_valid_driving_forces(self) -> Dict[str, Any]:
        forces = super().get_valid_driving_forces()
        forces.update(W=None, bc=None, src=None)
        return forces

    def check_facility_convergence(self, problem):
        values, names, tolerances, evaluated = get_convergence_values_wells(self, problem)
        return values < tolerances, values, names, evaluated

    def get_facility_convergence_values(self, problem, *args):
        values, names, tolerances, evaluated = get_convergence_values_wells(self, problem)
        return values, tolerances, names, evaluated

    def get_convergence_values(self, problem, *args):
        values, names, tolerances, _ = get_convergence_values_wells(self, problem)
        return values, tolerances, names

    @staticmethod
    def handle_repeated_perforated_cells(wc: np.ndarray, *sources: Sequence[Any]):
        """Merge repeated perforated cells, summing their source terms."""
        out = list(sources)
        c, _, ic = unique_stable(np.asarray(wc))
        if np.size(c) != np.size(wc):
            n = np.size(wc)
            A = sp.csr_matrix((np.ones(n), (np.asarray(ic), np.arange(n))),
                              shape=(np.size(c), n))
            wc = np.asarray(c)
            out = [[ad_matmul(A, q) for q in group] for group in sources]
        return wc, tuple(out)

    @staticmethod
    def combine_cell_data(data: Sequence[Sequence[Any]], ix: int) -> Any:
        parts = [_as_ad_column(d[ix]) for d in data]
        if not parts:
            return np.zeros(0)
        if any(hasattr(p, "jac") for p in parts):
            return vertcat_adi(*parts)
        return np.concatenate([np.ravel(p) for p in parts])

    setupStateFunctionGroupings = setup_state_function_groupings
    removeStateFunctionGroupings = remove_state_function_groupings
    getProp = get_prop
    getVariableField = get_variable_field
    setupWells = setup_wells
    setReservoirModel = set_reservoir_model
    getWellStruct = get_well_struct
    getNumberOfActiveWells = get_number_of_active_wells
    getNumberOfWells = get_number_of_wells
    getIndicesOfActiveWells = get_indices_of_active_wells
    getWellStatusMask = get_well_status_mask
    initStateAD = init_state_ad
    getPrimaryVariables = get_primary_variables
    validateModel = validate_model
    getPrimaryVariableNames = get_primary_variable_names
    getBasicPrimaryVariableNames = get_basic_primary_variable_names
    getBasicPrimaryVariables = get_basic_primary_variables
    getExtraPrimaryVariables = get_extra_primary_variables
    getAllPrimaryVariables = get_all_primary_variables
    getWellContributions = get_well_contributions
    updateWellSolAfterStep = update_well_sol_after_step
    updateAfterConvergence = update_after_convergence
    getWellCells = get_well_cells
    getActiveWellCells = get_active_well_cells
    setWellSolStatistics = set_well_sol_statistics
    updateWellSol = update_well_sol
    getWellVariableMap = get_well_variable_map
    getEquations = get_equations
    updateState = update_state
    validateState = validate_state
    getValidDrivingForces = get_valid_driving_forces
    checkFacilityConvergence = check_facility_convergence
    getFacilityConvergenceValues = get_facility_convergence_values
    getConvergenceValues = get_convergence_values
    handleRepeatedPerforatedcells = handle_repeated_perforated_cells
    combineCellData = combine_cell_data


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _index_ci(names: Sequence[str], name: str) -> Optional[int]:
    lowered = [n.lower() for n in names]
    return lowered.index(name.lower()) if name.lower() in lowered else None


def _unique_stable_index(items: Sequence[str]) -> Tuple[List[str], List[int]]:
    seen: Dict[str, int] = {}
    keep: List[int] = []
    for k, item in enumerate(items):
        if item not in seen:
            seen[item] = k
            keep.append(k)
    return [items[k] for k in keep], keep


def _set_row(x: Any, i: int, new: Any) -> Any:
    """`x(i) = new`, where `new` is a one-element value (AD or not)."""
    if hasattr(x, "jac"):
        out = x + 0.0
        out[[i]] = new
        return out
    out = np.array(x, dtype=float, copy=True)
    out[i] = float(np.ravel(value(new))[0])
    return out


def _as_ad_column(x: Any) -> Any:
    if hasattr(x, "jac"):
        return x
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


__all__ = ["FacilityModel"]


if __name__ == "__main__":
    def _rows(*status):
        return [SimpleNamespace(status=s, bhp=100e5 + i, qWs=-1.0*i, qOs=-2.0*i)
                for i, s in enumerate(status)]

    def _facility(nw=2):
        fm = FacilityModel()
        fm.ReservoirModel = SimpleNamespace(get_phase_names=lambda: "WO",
                                            get_component_names=lambda: [])
        fm.WellModels = [SimpleNamespace(W=SimpleNamespace(status=True,
                                                           cells=np.array([i, i + 1])),
                                         get_variable_counts=lambda f: 1)
                         for i in range(nw)]
        return fm

    def test_status_mask_needs_both_the_model_and_the_well_solution_open():
        fm = _facility()
        np.testing.assert_array_equal(fm.get_well_status_mask(_rows(True, False)),
                                      [True, False])
        assert fm.get_number_of_active_wells(_rows(True, True)) == 2
        print("ok test_status_mask_needs_both_the_model_and_the_well_solution_open")

    def test_basic_variables_are_rates_then_bhp_for_active_wells():
        fm = _facility()
        variables, names, wmap = fm.get_basic_primary_variables(_rows(True, True))
        assert names == ["qWs", "qOs", "bhp"]
        np.testing.assert_allclose(variables[2], [100e5, 100e5 + 1])
        np.testing.assert_array_equal(wmap["isBHP"], [False, False, True])
        print("ok test_basic_variables_are_rates_then_bhp_for_active_wells")

    def test_repeated_perforations_are_merged_and_summed():
        wc, (mass,) = FacilityModel.handle_repeated_perforated_cells(
            np.array([3, 5, 3]), [np.array([1.0, 2.0, 4.0])])
        np.testing.assert_array_equal(wc, [3, 5])
        np.testing.assert_allclose(mass[0], [5.0, 2.0])
        print("ok test_repeated_perforations_are_merged_and_summed")

    def test_variable_map_names_the_owning_active_well():
        fm = _facility(3)
        rows = _rows(True, False, True)
        fm.WellModels[1].W.status = True
        np.testing.assert_array_equal(fm.get_well_variable_map("bhp", rows), [0, 1])
        print("ok test_variable_map_names_the_owning_active_well")

    test_status_mask_needs_both_the_model_and_the_well_solution_open()
    test_basic_variables_are_rates_then_bhp_for_active_wells()
    test_repeated_perforations_are_merged_and_summed()
    test_variable_map_names_the_owning_active_well()
    print("\nAll tests passed!")
