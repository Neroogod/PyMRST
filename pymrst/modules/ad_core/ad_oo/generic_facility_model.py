"""MRST ``ad-core/models/facilities/GenericFacilityModel.m``.

The facility for the generic (component mass) models. Where `FacilityModel`
asks each well model for its equations, this one evaluates the
`FacilityFlowDiscretization` state functions - perforation pressure
differences, phase and component fluxes - and assembles from them:

- the component sources the reservoir equations receive;
- the surface-rate equations, `q_s - surface rates`, scaled to mass;
- one control equation per well (bhp, a surface rate, reservoir volume,
  or total volume), closing on the total rate when a target is zero or the
  flow runs the wrong way.

Between Newton iterations it enforces well limits, and once per report step
it converts history-matched reservoir-volume controls (`resv_history`) to
`resv` targets from the reservoir's average state.
"""

from __future__ import annotations

from copy import copy, deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import numpy as np

from pymrst.modules.ad_core.ad_oo.facility_model import FacilityModel
from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel
from pymrst.utils.adi import value
from pymrst.utils.units import ATM, BARSA, DAY

STANDARD_TEMPERATURE: float = 288.15
ZERO_FLUX: float = 1e-20


class GenericFacilityModel(FacilityModel):
    """MRST `GenericFacilityModel`."""

    def __init__(self, reservoir_model: Any = None, **options: Any):
        own = {k: options.pop(k) for k in
               ("T", "pressure", "SeparatorGroup", "outputFluxes", "doPostUpdate")
               if k in options}
        super().__init__(reservoir_model, **options)
        self.T = STANDARD_TEMPERATURE
        self.pressure = 1*ATM
        self.SeparatorGroup = None
        self.outputFluxes = True
        self.doPostUpdate = None
        for key, item in own.items():
            setattr(self, key, item)
        self.setWellTargets = False

    def get_number_of_components(self) -> int:
        return self.ReservoirModel.get_number_of_components()

    def get_number_of_phases(self) -> int:
        return self.ReservoirModel.get_number_of_phases()

    def get_component_sources(self, state: Any) -> SimpleNamespace:
        fp = self.FacilityFlowDiscretization
        mapping = fp.get(self, state, "FacilityWellMapping")
        val = [] if not mapping.W else fp.get(self, state, "ComponentTotalFlux")
        return SimpleNamespace(value=val, cells=mapping.cells)

    def get_surface_rates(self, state: Any):
        """Per-well surface rates of each phase, and the surface densities."""
        fp = self.FacilityFlowDiscretization
        cflux = fp.get(self, state, "ComponentTotalFlux")
        mapping = fp.get(self, state, "FacilityWellMapping")
        model = self.ReservoirModel
        cflux = [_matmul(mapping.perforationSum, c) for c in cflux]
        p, temp = self.get_surface_conditions()
        surface_density = list(fp.get(self, state, "InjectionSurfaceDensity"))
        nph = model.get_number_of_phases()
        nw = np.size(value(cflux[0]))
        surface_rates: List[Any] = [np.zeros(nw) for _ in range(nph)]
        from pymrst.modules.ad_core.statefunctions.components.components import (
            ConcentrationComponent,
        )
        for c, flux in enumerate(cflux):
            component = model.Components[c]
            if isinstance(component, ConcentrationComponent):
                continue
            composition = component.get_phase_composition_surface(model, state, p, temp)
            for ph in range(nph):
                if composition[ph] is not None and np.any(composition[ph]):
                    ci = composition[ph]/surface_density[ph]
                    surface_rates[ph] = surface_rates[ph] + ci*flux
        is_prod = ~np.asarray(mapping.isInjector, dtype=bool)
        if self.SeparatorGroup is not None and np.any(is_prod):
            cflux_prod = [c[np.flatnonzero(is_prod)] for c in cflux]
            rates_prod, density_prod = self.SeparatorGroup.get_surface_rates(model, cflux_prod)
            rows = np.flatnonzero(is_prod)
            for ph in range(nph):
                if not hasattr(surface_density[ph], "jac") and hasattr(density_prod[ph], "jac"):
                    surface_density[ph] = model.AutoDiffBackend.convert_to_ad(
                        surface_density[ph], density_prod[ph])
                surface_rates[ph] = _assign_rows(surface_rates[ph], rows, rates_prod[ph])
                surface_density[ph] = _assign_rows(surface_density[ph], rows, density_prod[ph])
        return surface_rates, surface_density

    def get_equations(self, state0, state, dt, driving_forces, **options):
        return PhysicalModel.get_equations(self, state0, state, dt, driving_forces,
                                           **options)

    def get_model_equations(self, state0: Any, state: Any, dt: float,
                            driving_forces: Any):
        """The wells' rate and control equations from the facility grouping."""
        model = self.ReservoirModel
        fp = self.FacilityFlowDiscretization
        mapping = fp.get(self, state, "FacilityWellMapping")
        choice = self.primaryVariableSet.lower()
        if not mapping.W or choice == "none":
            return [], [], [], state
        nph = model.get_number_of_phases()
        if choice == "explicit":
            ph = model.get_phase_names()
            q_s, bh = self.get_props(state, "q_s", "bhp")
            bhp = self.get_props(state, "BottomHolePressure")
            surface_rates, _ = self.get_surface_rates(state)
            eqs = [surface_rates[i] - q_s[i] for i in range(nph)] + [bhp - bh]
            names = [f"q{ph[i]}" for i in range(nph)] + ["bh closure"]
            return eqs, names, ["well"]*(nph + 1), state
        wsum = mapping.perforationSum
        surface_rates, rho_s = self.get_surface_rates(state)
        rho_r = np.column_stack([np.asarray(value(r), dtype=float)
                                 for r in model.get_props(state0, "Density")])
        rho_scale = (np.column_stack([np.asarray(value(r), dtype=float)*np.ones(len(mapping.active))
                                      for r in rho_s])
                     / rho_r.mean(axis=0))
        sn = model.get_phase_names()
        phnames = model.get_long_phase_names()
        if choice == "standard":
            q_s = self.get_prop(state, "q_s")
            eqs = [(q_s[ph] - surface_rates[ph])*rho_scale[:, ph] for ph in range(nph)]
            names = [f"{phnames[ph]}Wells" for ph in range(nph)]
            types = ["perf"]*nph
            eqs.append(None)
            names.append(None)
            types.append(None)
            target_rates = q_s
            ctrl_index = nph
        elif choice in ("bhp_massfractions", "bhp"):
            if choice == "bhp_massfractions":
                mass_fractions = state["FacilityState"]["massfractions"]
                cnames = model.get_component_names()
                sources = self.get_props(state, "ComponentTotalFlux")
                ncomp = model.get_number_of_components()
                eqs, names, types = [None]*ncomp, [None]*ncomp, [None]*ncomp
                total = 0.0
                for i in range(ncomp):
                    sources[i] = _matmul(wsum, sources[i])
                    total = total + sources[i]
                for i in range(ncomp - 1):
                    eqs[i + 1] = mass_fractions[i] - sources[i]/total
                    names[i + 1] = f"well_{cnames[i]}"
                    types[i + 1] = "wellcomposition"
            else:
                eqs, names, types = [None], [None], [None]
            ctrl_index = 0
            target_rates = surface_rates
            q_surf = np.column_stack([np.asarray(value(r), dtype=float).ravel()
                                      for r in surface_rates])
            for ph, letter in enumerate(sn):
                for i, ix in enumerate(mapping.active):
                    setattr(state["wellSol"][ix], f"q{letter}s", float(q_surf[i, ph]))
        else:
            raise ValueError(f"Unknown variable set for facility {self.primaryVariableSet}")
        eqs[ctrl_index] = self._control_equations(state, fp, mapping, target_rates,
                                                  surface_rates)
        names[ctrl_index] = "closureWells"
        types[ctrl_index] = "well"
        return eqs, names, types, state

    def _control_equations(self, state, fp, mapping, target_rates, surface_rates):
        model = self.ReservoirModel
        nph = model.get_number_of_phases()
        nact = len(mapping.active)
        bhp = self.get_prop(state, "bhp")
        backend = model.AutoDiffBackend
        ctrl_eq = backend.convert_to_ad(np.zeros(nact), bhp)
        rows = [state["wellSol"][i] for i in mapping.active]
        controls = np.array([r.type for r in rows])
        targets = np.array([float(r.val) for r in rows])
        is_bhp = controls == "bhp"
        if np.isfinite(self.toleranceWellRate) and np.isfinite(self.toleranceWellBHP):
            eq_scale = self.toleranceWellRate/self.toleranceWellBHP
        else:
            eq_scale = 1/(DAY*BARSA)
        if np.any(is_bhp):
            k = np.flatnonzero(is_bhp)
            ctrl_eq[k] = eq_scale*(bhp[k] - targets[k])
        lowered = np.char.lower(controls.astype(str))
        is_rate = (controls == "rate") | (lowered == "vrat")
        is_orat, is_wrat = controls == "orat", controls == "wrat"
        is_grat, is_lrat = controls == "grat", controls == "lrat"
        is_resv, is_volume = controls == "resv", controls == "volume"
        phases = model.get_phase_names()
        is_surface = np.zeros(nact, dtype=bool)
        wrates = backend.convert_to_ad(np.zeros(nact), bhp)
        wrates_actual = np.zeros(nact)
        surface_value = np.column_stack([np.asarray(value(r), dtype=float).ravel()
                                         for r in surface_rates])
        for i in range(nph):
            act = {"W": is_rate | is_wrat | is_lrat,
                   "O": is_rate | is_orat | is_lrat,
                   "G": is_rate | is_grat}[phases[i]]
            is_surface |= act
            if np.any(act):
                k = np.flatnonzero(act)
                wrates[k] = wrates[k] + target_rates[i][k]
                wrates_actual[act] += surface_value[act, i]
        if np.any(is_surface):
            k = np.flatnonzero(is_surface)
            ctrl_eq[k] = wrates[k] - targets[k]
        if np.any(is_resv):
            q_s = self.get_prop(state, "q_s")
            dens = np.vstack([self.WellModels[i].ControlDensity
                              for i, r in zip(mapping.active, is_resv) if r])
            k = np.flatnonzero(is_resv)
            resv = 0.0
            for ph in range(nph):
                resv = resv + q_s[ph][k]*dens[:, ph]
            ctrl_eq[k] = resv - targets[k]
        wsign = np.array([w.sign for w in mapping.W])
        surf = np.asarray(value(wrates), dtype=float)
        zero_target = (targets == 0) & (is_surface | is_resv)
        zero_bhp = is_bhp & (np.sign(surf) != wsign) & (wsign != 0) & (surf != 0)
        zero_rates = zero_target | zero_bhp | ((wrates_actual == 0) & is_surface)
        if np.any(zero_rates):
            k = np.flatnonzero(zero_rates)
            q_t = 0.0
            for i in range(nph):
                q_t = q_t + target_rates[i][k]
            ctrl_eq[k] = q_t
        if np.any(is_volume):
            phase_flux = fp.get(self, state, "PhaseFlux")
            total_flux = 0.0
            for flux in phase_flux:
                total_flux = total_flux + flux
            well_total = _matmul(mapping.perforationSum, total_flux)
            k = np.flatnonzero(is_volume)
            ctrl_eq[k] = well_total[k] - targets[k]
        assert np.all(is_surface | is_bhp | is_volume | is_resv)
        return ctrl_eq

    def apply_well_limits(self, state: Any) -> Any:
        for w in self.get_indices_of_active_wells(state["wellSol"]):
            state["wellSol"][w] = self.apply_well_limits_well_sol(
                self.WellModels[w], state["wellSol"][w])
        return state

    def validate_model(self, *args: Any) -> "GenericFacilityModel":
        if self.doPostUpdate is None:
            self.doPostUpdate = self.primaryVariableSet.lower() == "none"
        return super().validate_model(*args)

    def prepare_report_step(self, state, state0, dt, driving_forces):
        model, state = super().prepare_report_step(state, state0, dt, driving_forces)
        return model.update_resv_controls(state, state0)

    def update_state(self, state, problem, dx, driving_forces):
        state, report = super().update_state(state, problem, dx, driving_forces)
        if self.doPostUpdate:
            surface_rates, _ = self.get_surface_rates(state)
            names = self.ReservoirModel.get_phase_names()
            mapping = self.get_prop(state, "FacilityWellMapping")
            for j, ix in enumerate(mapping.active):
                row = state["wellSol"][ix]
                for i, letter in enumerate(names):
                    setattr(row, f"q{letter}s",
                            float(np.ravel(value(surface_rates[i]))[j]))
        return state, report

    def prepare_timestep(self, state, state0, dt, driving_forces):
        """Connection pressure drops from the current densities; shut wells
        whose rate target is zero."""
        well_sol = state["wellSol"]
        act = self.get_indices_of_active_wells(well_sol)
        if act.size > 0:
            rho = np.column_stack([np.asarray(value(r), dtype=float)
                                   for r in self.ReservoirModel.get_props(state, "Density")])
            wc, perf2well = self.get_active_well_cells(well_sol)
            rho = rho[wc, :]
            flux_missing = any(
                getattr(ws, "ComponentTotalFlux", None) is None
                or np.sum(np.abs(ws.ComponentTotalFlux)) < ZERO_FLUX for ws in well_sol)
            mob = None
            if flux_missing:
                mob = np.column_stack([np.asarray(value(m), dtype=float) for m in
                                       self.ReservoirModel.get_props(state, "Mobility")])[wc, :]
            for well_no in act:
                wm = self.WellModels[well_no]
                mine = np.asarray(perf2well) == well_no
                rho_i = rho[mine, :]
                mob_i = None if mob is None else mob[mine, :]
                well_sol[well_no] = wm.update_connection_pressure_drop_state(
                    self.ReservoirModel, well_sol[well_no], rho_i, rho_i, mob_i)
                ctrl = wm.W.type.lower()
                if ctrl not in ("bhp", "thp"):
                    well_sol[well_no].status = wm.W.val != 0
        state["wellSol"] = well_sol
        state = self.apply_well_limits(state)
        return self, state

    def update_after_convergence(self, state0, state, dt, driving_forces):
        state, report = super().update_after_convergence(state0, state, dt,
                                                         driving_forces)
        if self.get_number_of_active_wells(state["wellSol"]) > 0:
            mapping = self.FacilityFlowDiscretization.get(self, state,
                                                          "FacilityWellMapping")
            phaseq = np.column_stack([np.asarray(value(q), dtype=float)
                                      for q in self.get_prop(state, "PhaseFlux")])
            cf = np.column_stack([np.asarray(value(c), dtype=float)
                                  for c in self.get_prop(state, "ComponentTotalFlux")])
            nwt = len(state["wellSol"])
            act_index = {int(w): i for i, w in enumerate(mapping.active)}
            nph, ncomp = self.get_number_of_phases(), self.get_number_of_components()
            wells = driving_forces["W"] if isinstance(driving_forces, dict) else driving_forces.W
            for wi in range(nwt):
                if wi in act_index:
                    mine = np.asarray(mapping.perf2well) == act_index[wi]
                    flux, ctf = phaseq[mine, :], cf[mine, :]
                else:
                    nc = np.size(wells[wi].cells)
                    flux, ctf = np.zeros((nc, nph)), np.zeros((nc, ncomp))
                state["wellSol"][wi].flux = flux
                state["wellSol"][wi].ComponentTotalFlux = ctf
        return state, report

    def get_state_function_groupings(self):
        groups, names, models = PhysicalModel.get_state_function_groupings(self)
        if self.FacilityFlowDiscretization is not None:
            groups.append(self.FacilityFlowDiscretization)
            names.append("FacilityFlowDiscretization")
            models.append(self)
        return groups, names, models

    def get_surface_conditions(self):
        return self.pressure, self.T

    def apply_well_limits_well_sol(self, well: Any, well_sol: Any) -> Any:
        """Switch control when the wellSol breaches a limit."""
        if not well.allowControlSwitching or not getattr(well.W, "status", True):
            return well_sol
        lims = well.W.lims
        model = self.ReservoirModel
        phases = model.get_phase_names()
        if isinstance(lims, dict):
            qs_t = sum(getattr(well_sol, f"q{p}s") for p in phases)
            bhp = well_sol.bhp
            if well.is_injector():
                modes = ["bhp", "rate", "vrat"]
                lims = well.set_missing_limits(lims, modes, np.inf)
                if not np.isfinite(lims["vrat"]):
                    lims["vrat"] = -np.inf
                flags = [bhp > lims["bhp"], qs_t > lims["rate"], qs_t < lims["vrat"]]
            else:
                modes = ["bhp", "orat", "lrat", "grat", "wrat", "vrat"]
                lims = self.set_missing_limits(lims, modes, -np.inf)
                if not np.isfinite(lims["vrat"]):
                    lims["vrat"] = np.inf
                q_w = well_sol.qWs if model.water else 0.0
                q_o = well_sol.qOs if model.oil else 0.0
                q_g = well_sol.qGs if model.gas else 0.0
                q_sl = getattr(well_sol, "qSs", None) or 0.0
                flags = [bhp < lims["bhp"], q_o < lims["orat"], q_w + q_o < lims["lrat"],
                         q_g + q_sl < lims["grat"], q_w < lims["wrat"], qs_t > lims["vrat"]]
        else:
            modes, flags = [], []
            assert lims is None or np.isinf(lims)
        checked = [(m, f) for m, f in zip(modes, flags) if m != well_sol.type]
        violated = next((m for m, f in checked if f), None)
        if violated is None:
            return well_sol
        print(f"Well {well_sol.name}: Control mode changed from {well_sol.type} "
              f"to {violated}.")
        well_sol.type = violated
        well_sol.val = lims[violated]
        v = well_sol.val
        if well_sol.type == "rate":
            compi = np.asarray(well.W.compi, dtype=float).ravel()
            for ix, letter in enumerate(phases):
                setattr(well_sol, f"q{letter}s", v*compi[ix])
        elif well_sol.type in ("orat", "wrat", "grat"):
            setattr(well_sol, f"q{well_sol.type[0].upper()}s", v)
        return well_sol

    def set_missing_limits(self, lims: Dict[str, float], modes: Sequence[str],
                           val: float) -> Dict[str, float]:
        lims = dict(lims)
        for mode in modes:
            lims.setdefault(mode, val)
        return lims

    def update_resv_controls(self, state: Any, state0: Any):
        """Turn `resv_history` targets into `resv` rates at reservoir state."""
        mask = self.get_well_status_mask(state["wellSol"])
        active_models = [wm for wm, a in zip(self.WellModels, mask) if a]
        is_hist = np.array([wm.W.type.lower() == "resv_history" for wm in active_models], dtype=bool)
        is_now = np.array([wm.W.type.lower() == "resv" for wm in active_models], dtype=bool)
        is_resv = is_hist | is_now
        if not np.any(is_resv):
            return self, state
        W = [wm.W for wm in active_models]
        hist = is_hist[is_resv]
        compi = np.vstack([np.asarray(w.compi, dtype=float).ravel() for w in W])[is_resv]
        rates = np.array([float(w.val) for w, r in zip(W, is_resv) if r])
        qs = rates[:, None]*compi
        rmodel = self.ReservoirModel
        disgas = bool(getattr(rmodel, "disgas", False))
        vapoil = bool(getattr(rmodel, "vapoil", False))
        oix, gix = rmodel.get_phase_index("O", "G")
        pvt_reg = rmodel.PVTPropertyFunctions.get_state_function("Density").regions
        if pvt_reg is None:
            pvt_reg = np.ones(rmodel.G.cells.num, dtype=int)
        pvt_reg = np.asarray(pvt_reg)
        cells = np.array([np.asarray(w.cells).ravel()[0] for w, r in zip(W, is_resv) if r])
        nc = cells.size
        reg_no = pvt_reg[cells]
        rs, rv, pm = np.zeros(nc), np.zeros(nc), np.zeros(nc)
        pv = np.asarray(value(rmodel.get_prop(state0, "PoreVolume")), dtype=float)
        if rmodel.water:
            pv = pv*(1 - np.asarray(value(rmodel.get_prop(state0, "sw")), dtype=float))
        for reg in range(1, int(reg_no.max()) + 1):
            subs = reg_no == reg
            pvi = pv*(pvt_reg == reg)
            pm[subs] = np.sum(state0["pressure"]*pvi)/np.sum(pvi)
            if disgas:
                rs[subs] = np.sum(state0["rs"]*pvi)/np.sum(pvi)
            if vapoil:
                rv[subs] = np.sum(state0["rv"]*pvi)/np.sum(pvi)
        from pymrst.modules.ad_core.statefunctions.state_function import State
        # MATLAB copies into the struct: the shrinkage factors below are taken
        # at the averaged rs/rv, before the history wells' are limited.
        substate = State(pressure=pm, s=np.tile([1.0, 0.0, 0.0], (nc, 1)), rs=rs.copy(),
                         rv=rv.copy())
        rs0, rv0 = rs.copy(), rv.copy()
        if np.any(hist):
            if disgas:
                rs[hist] = np.minimum(qs[hist, gix]/qs[hist, oix], rs[hist])
            if vapoil:
                rv[hist] = np.minimum(qs[hist, oix]/qs[hist, gix], rv[hist])
        minimodel = copy(rmodel)
        minimodel.PVTPropertyFunctions = deepcopy(rmodel.PVTPropertyFunctions).subset(cells)
        minimodel.FlowPropertyFunctions = deepcopy(rmodel.FlowPropertyFunctions).subset(cells)
        shrinkage = minimodel.PVTPropertyFunctions.get_state_function("ShrinkageFactors")
        if hasattr(shrinkage, "useSaturatedFlag"):
            shrinkage.useSaturatedFlag = True
        minimodel.G = _subset_grid(rmodel.G, nc)
        substate = minimodel.init_state_function_containers(substate)
        b = minimodel.get_prop(substate, "ShrinkageFactors")
        shrink = 1 - rs*rv
        shrink0 = 1 - rs0*rv0
        nph = self.get_number_of_phases()
        factors = np.zeros((nc, nph))
        new_rates = 0.0
        if rmodel.water:
            b_w = np.asarray(value(b[0]), dtype=float)
            factors[:, 0] = 1/b_w
            new_rates = new_rates + qs[:, 0]/b_w
        if rmodel.oil:
            b_o = np.asarray(value(b[oix]), dtype=float)
            orat = qs[:, oix]
            factors[:, oix] = 1/(b_o*shrink0)
            if vapoil:
                orat = orat - rv*qs[:, gix]
                factors[:, gix] = factors[:, gix] - rv/(b_o*shrink0)
            new_rates = new_rates + orat/(b_o*shrink)
        if rmodel.gas:
            b_g = np.asarray(value(b[gix]), dtype=float)
            grat = qs[:, gix]
            factors[:, gix] = factors[:, gix] + 1/(b_g*shrink0)
            if vapoil:
                grat = grat - rs*qs[:, oix]
                factors[:, oix] = factors[:, oix] - rs/(b_g*shrink0)
            new_rates = new_rates + grat/(b_g*shrink)
        act_ix = np.flatnonzero(mask)
        for i, I in enumerate(np.flatnonzero(is_resv)):
            g = act_ix[I]
            if is_hist[I]:
                self.WellModels[g].W.val = float(new_rates[i])
                state["wellSol"][g].val = float(new_rates[i])
                self.WellModels[g].W.type = "resv"
                state["wellSol"][g].type = "resv"
            self.WellModels[g].ControlDensity = factors[i, :]
        return self, state

    getNumberOfComponents = get_number_of_components
    getNumberOfPhases = get_number_of_phases
    getComponentSources = get_component_sources
    getSurfaceRates = get_surface_rates
    getEquations = get_equations
    getModelEquations = get_model_equations
    applyWellLimits = apply_well_limits
    validateModel = validate_model
    prepareReportstep = prepare_report_step
    updateState = update_state
    prepareTimestep = prepare_timestep
    updateAfterConvergence = update_after_convergence
    getStateFunctionGroupings = get_state_function_groupings
    getSurfaceConditions = get_surface_conditions
    applyWellLimitsWellSol = apply_well_limits_well_sol
    setMissingLimits = set_missing_limits
    updateRESVControls = update_resv_controls


def _matmul(matrix: Any, x: Any) -> Any:
    from pymrst.utils.adi import ad_matmul
    if x is None:
        # An empty flux (no wells): MATLAB's product with `[]` is empty.
        return np.zeros(matrix.shape[0])
    return ad_matmul(matrix, x)


def _assign_rows(x: Any, rows: np.ndarray, new: Any) -> Any:
    out = x + 0.0 if hasattr(x, "jac") else np.array(x, dtype=float, copy=True)
    out[rows] = new
    return out


def _subset_grid(G: Any, nc: int) -> Any:
    """A grid stand-in whose cell count is the subset's (the property
    functions size their output from `G.cells.num`)."""
    return SimpleNamespace(cells=SimpleNamespace(num=nc), griddim=getattr(G, "griddim", 3))


__all__ = ["GenericFacilityModel"]


if __name__ == "__main__":
    def test_well_switches_to_bhp_when_its_limit_is_breached():
        fm = GenericFacilityModel()
        fm.ReservoirModel = SimpleNamespace(get_phase_names=lambda: "WO", water=True,
                                            oil=True, gas=False)
        W = SimpleNamespace(lims={"bhp": 100e5}, status=True)
        well = SimpleNamespace(allowControlSwitching=True, W=W,
                               is_injector=lambda: False)
        ws = SimpleNamespace(type="orat", val=-1.0, bhp=90e5, qWs=-0.1, qOs=-1.0,
                             name="P")
        ws = fm.apply_well_limits_well_sol(well, ws)
        assert ws.type == "bhp" and ws.val == 100e5
        print("ok test_well_switches_to_bhp_when_its_limit_is_breached")

    def test_a_well_within_limits_keeps_its_control():
        fm = GenericFacilityModel()
        fm.ReservoirModel = SimpleNamespace(get_phase_names=lambda: "WO", water=True,
                                            oil=True, gas=False)
        W = SimpleNamespace(lims={"bhp": 50e5}, status=True)
        well = SimpleNamespace(allowControlSwitching=True, W=W,
                               is_injector=lambda: False)
        ws = SimpleNamespace(type="orat", val=-1.0, bhp=90e5, qWs=0.0, qOs=-1.0,
                             name="P")
        assert fm.apply_well_limits_well_sol(well, ws).type == "orat"
        print("ok test_a_well_within_limits_keeps_its_control")

    def test_facility_does_not_impose_targets_on_well_solutions():
        assert GenericFacilityModel().setWellTargets is False
        print("ok test_facility_does_not_impose_targets_on_well_solutions")

    test_well_switches_to_bhp_when_its_limit_is_breached()
    test_a_well_within_limits_keeps_its_control()
    test_facility_does_not_impose_targets_on_well_solutions()
    print("\nAll tests passed!")
