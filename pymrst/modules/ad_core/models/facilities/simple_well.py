"""MRST ``ad-core/models/facilities/SimpleWell`` compatibility model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from pymrst.params.wells_and_bc import Well
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.utils.adi import value


class SimpleWell:
    """Single, instantaneous-equilibrium well model.

    This mirrors MRST's ``SimpleWell`` object surface: a per-well model owns the
    schedule well struct, advertises any extra variables/equations introduced by
    the reservoir model, validates ``wellSol`` rows, and supplies per-well
    equation/source hooks for ``FacilityModel``.
    """

    def __init__(self, W: Well, **kwargs: Any):
        self.W = W
        self.allowCrossflow = True
        self.allowSignChange = False
        self.allowControlSwitching = True
        self.doUpdatePressureDrop = True
        self.simplePressureDrop = False
        self.dpMaxRel = np.inf
        self.dpMaxAbs = np.inf
        self.dsMaxAbs = np.inf
        self.VFPTable = getattr(W, "vfp_table", None)
        self.ControlDensity = getattr(W, "control_density", None)
        for key, item in kwargs.items():
            setattr(self, key, item)

    @property
    def allow_crossflow(self) -> bool:
        return self.allowCrossflow

    @allow_crossflow.setter
    def allow_crossflow(self, value_k: bool) -> None:
        self.allowCrossflow = bool(value_k)

    @property
    def allow_sign_change(self) -> bool:
        return self.allowSignChange

    @allow_sign_change.setter
    def allow_sign_change(self, value_k: bool) -> None:
        self.allowSignChange = bool(value_k)

    @property
    def allow_control_switching(self) -> bool:
        return self.allowControlSwitching

    @allow_control_switching.setter
    def allow_control_switching(self, value_k: bool) -> None:
        self.allowControlSwitching = bool(value_k)

    @property
    def vfp_table(self) -> Any:
        return self.VFPTable

    @vfp_table.setter
    def vfp_table(self, value_k: Any) -> None:
        self.VFPTable = value_k

    def updateWell(self, W: Well):
        """Update the schedule/control well struct."""
        self.W = W
        if getattr(W, "vfp_table", None) is not None:
            self.VFPTable = W.vfp_table
        return self

    update_well = updateWell

    @staticmethod
    def _container_field(container: Any, name: str, default: Any = None) -> Any:
        if container is None:
            return default
        if isinstance(container, dict):
            return container.get(name, default)
        return getattr(container, name, default)

    @staticmethod
    def _set_container_field(container: Any, name: str, item: Any) -> Any:
        if isinstance(container, dict):
            container[name] = item
        else:
            setattr(container, name, item)
        return container

    @staticmethod
    def _call(owner: Any, names: Sequence[str], *args: Any, default: Any = None) -> Any:
        if owner is None:
            return default
        for name in names:
            if hasattr(owner, name):
                return getattr(owner, name)(*args)
        return default

    def _phase_names(self, resmodel: Any = None) -> list[str]:
        names = self._call(resmodel, ("getPhaseNames", "get_phase_names"))
        if names is not None:
            return [str(item) for item in names]
        nph = max(np.asarray(getattr(self.W, "compi", [])).size, 1)
        base = ["W", "O", "G"]
        return base[:nph] if nph <= len(base) else base + [
            f"P{i}" for i in range(len(base) + 1, nph + 1)
        ]

    def _number_of_components(self, resmodel: Any = None) -> int:
        count = self._call(resmodel, ("getNumberOfComponents", "get_number_of_components"))
        if count is not None:
            return int(count)
        components = getattr(self.W, "components", None)
        return max(np.asarray(components if components is not None else self.W.compi).size, 1)

    def getExtraPrimaryVariableNames(self, resmodel: Any = None):
        """Return extra well variables requested by the reservoir model."""
        names = self._call(resmodel, ("getExtraWellPrimaryVariableNames", "get_extra_well_primary_variable_names"))
        names = [] if names is None else list(names)
        return names, [True] * len(names)

    get_extra_primary_variable_names = getExtraPrimaryVariableNames

    def getExtraEquationNames(self, resmodel: Any = None):
        """Return extra well equations requested by the reservoir model."""
        result = self._call(resmodel, ("getExtraWellEquationNames", "get_extra_well_equation_names"))
        if result is None:
            return [], []
        if isinstance(result, tuple):
            names = list(result[0])
            types = list(result[1]) if len(result) > 1 else ["well"] * len(names)
        else:
            names = list(result)
            types = ["well"] * len(names)
        return names, types

    get_extra_equation_names = getExtraEquationNames

    def getExtraPrimaryVariables(self, wellSol: Any, resmodel: Any = None):
        names, from_res_model = self.getExtraPrimaryVariableNames(resmodel)
        variables: list[Any] = []
        for name, from_res in zip(names, from_res_model):
            owner = resmodel if from_res else self
            variable = self._call(owner, ("getProp", "get_prop"), wellSol, name, default=None)
            if variable is None:
                variable = self._container_field(wellSol, str(name), 0.0)
            variables.append(variable)
        return variables, names

    get_extra_primary_variables = getExtraPrimaryVariables

    def getVariableCounts(self, fld: str) -> int:
        return 1 if str(fld) else 0

    get_variable_counts = getVariableCounts

    def getVariableField(self, name: str, *_, strict: bool = True):
        """Map MRST well variable names to ``wellSol`` fields."""
        key = str(name).lower()
        mapping = {
            "bhp": ("bhp", 0),
            "qos": ("qOs", 0),
            "qgs": ("qGs", 0),
            "qws": ("qWs", 0),
            "qss": ("qSs", 0),
            "well_temperature": ("T", 0),
            "t": ("T", 0),
            "well_massfractions": ("massfractions", slice(None)),
        }
        if key in mapping:
            return mapping[key]
        if strict:
            raise KeyError(f"unknown SimpleWell variable field {name!r}")
        return str(name), slice(None)

    get_variable_field = getVariableField

    @staticmethod
    def setMissingLimits(lims: Any, modes: Sequence[str], val: float):
        """Fill missing MRST ``W.lims`` modes with a default value."""
        out = {} if lims is None else dict(lims)
        for mode in modes:
            out.setdefault(str(mode), float(val))
        return out

    set_missing_limits = setMissingLimits

    @staticmethod
    def wb2in(w: Well) -> np.ndarray:
        """Return MRST's wellbore-to-inflow incidence matrix."""
        nperf = np.asarray(w.cells).size
        topo = getattr(w, "topo", None)
        if topo is None:
            topo = np.column_stack([np.arange(nperf), np.arange(1, nperf + 1)])
        topo = np.asarray(topo, dtype=int)
        conn = topo[1:, :] if topo.size else np.zeros((0, 2), dtype=int)
        C = np.eye(nperf)
        for left, right in conn:
            if 1 <= left <= nperf and 1 <= right <= nperf:
                C[left - 1, right - 1] -= 1.0
            elif 0 <= left < nperf and 0 <= right < nperf:
                C[left, right] -= 1.0
        return C

    @staticmethod
    def compVolRat(mixs: Any, p: Any, b: Any, model: Any) -> np.ndarray:
        """Compute reservoir/surface volume ratio for independent phases."""
        del p, model
        mix = np.asarray(value(mixs), dtype=float)
        shrinkage = np.asarray(value(b), dtype=float)
        if shrinkage.ndim == 1:
            shrinkage = np.broadcast_to(shrinkage, mix.shape)
        return np.sum(mix / np.maximum(shrinkage, 1e-30), axis=1)

    comp_vol_rat = compVolRat

    @staticmethod
    def getTotalRate(sol: Any) -> np.ndarray:
        """Sum MRST surface-rate fields from one or more ``wellSol`` rows."""
        if sol is None:
            return np.zeros(0)
        if isinstance(sol, (dict, SimpleNamespace)):
            items = [sol]
        else:
            try:
                items = list(sol)
            except TypeError:
                items = [sol]
        rates = np.zeros(len(items), dtype=float)
        for row, item in enumerate(items):
            for field in ("qWs", "qOs", "qGs", "qSs"):
                field_value = SimpleWell._container_field(item, field, None)
                if field_value is not None:
                    rates[row] += float(np.asarray(value(field_value)).reshape(-1)[0])
        return rates

    get_total_rate = getTotalRate

    def validateWellSol(self, resmodel: Any, wsol: Any, state: Any = None):
        """Ensure a ``wellSol`` row has the fields this well model expects."""
        del state
        if wsol is None:
            wsol = SimpleNamespace()
        w = self.W
        nperf = np.asarray(w.cells).size
        ncomp = self._number_of_components(resmodel)
        phase_names = self._phase_names(resmodel)
        defaults = {
            "name": w.name,
            "status": bool(w.status),
            "type": str(w.type),
            "val": float(np.asarray(w.val).reshape(-1)[0]),
            "sign": int(w.sign),
            "bhp": float(np.asarray(w.val).reshape(-1)[0]),
            "pressure": float(np.asarray(w.val).reshape(-1)[0]),
            "flux": np.zeros(nperf),
            "cdp": np.zeros(nperf),
            "cstatus": np.ones(nperf, dtype=bool) if w.cstatus is None else np.asarray(w.cstatus, dtype=bool),
            "components": np.asarray(w.components if w.components is not None else w.compi, dtype=float).copy(),
            "massfractions": np.ones(ncomp, dtype=float) / ncomp,
        }
        for phase in phase_names:
            defaults[f"q{phase.upper()}s"] = 0.0
        for name in self.getExtraPrimaryVariableNames(resmodel)[0]:
            defaults[str(name)] = 0.0
        for key, item in defaults.items():
            if self._container_field(wsol, key, None) is None:
                self._set_container_field(wsol, key, item)
        return wsol

    validate_well_sol = validateWellSol

    def isInjector(self) -> bool:
        return int(getattr(self.W, "sign", 0)) > 0

    is_injector = isInjector

    def getWellEquationNames(self, resmodel: Any = None):
        names = [f"{phase}Wells" for phase in self._phase_names(resmodel)]
        return names, ["perf"] * len(names)

    get_well_equation_names = getWellEquationNames

    def updateConnectionPressureDrop(self, wellSol0: Any, wellSol: Any, model: Any,
                                     q_s: Sequence[Any], bhp: Any, packed: Any,
                                     dt: float | None, iteration: float | int | None):
        del wellSol0, model, q_s, bhp, packed, dt
        if iteration != 1 or not self.doUpdatePressureDrop:
            return wellSol
        nperf = np.asarray(self.W.cells).size
        cdp = self._container_field(wellSol, "cdp", None)
        if cdp is None or np.asarray(cdp).size != nperf:
            self._set_container_field(wellSol, "cdp", np.zeros(nperf))
        return wellSol

    update_connection_pressure_drop = updateConnectionPressureDrop

    def updateConnectionPressureDropState(self, model: Any, wellSol: Any,
                                          rho_res: Any, rho_well: Any = None,
                                          mob_res: Any = None):
        """Update ``wellSol.cdp`` from a reservoir-state pressure-drop column."""
        del rho_well
        if not self.doUpdatePressureDrop:
            return wellSol
        if hasattr(model, "connection_pressure_drop"):
            rho = rho_res if isinstance(rho_res, (list, tuple)) else [rho_res]
            mob = mob_res if isinstance(mob_res, (list, tuple)) else [mob_res if mob_res is not None else np.ones_like(value(rho[0]))]
            nph = len(rho)
            p = np.asarray(self._container_field(wellSol, "pressure", np.zeros(np.asarray(self.W.cells).size)))
            cdp = model.connection_pressure_drop(model, self.W, p, mob, [np.ones_like(value(rho[0]))] * nph, nph, rho=rho)
        else:
            gravity = float(getattr(model, "gravity_magnitude", getattr(model, "gravity", NORM_GRAVITY)))
            rho_arr = np.asarray(value(rho_res[0] if isinstance(rho_res, (list, tuple)) else rho_res), dtype=float)
            wc = np.asarray(self.W.cells, dtype=int)
            rho_c = rho_arr[wc] if rho_arr.size > np.max(wc, initial=0) else np.full(wc.size, float(np.mean(rho_arr)))
            cdp = np.cumsum(gravity * rho_c * np.diff(np.concatenate([[0.0], np.asarray(self.W.dZ, dtype=float).ravel()])))
        self._set_container_field(wellSol, "cdp", np.asarray(cdp, dtype=float))
        return wellSol

    update_connection_pressure_drop_state = updateConnectionPressureDropState

    def _mode_value(self, mode: str, q_s: Sequence[Any], bhp: Any) -> float:
        rates = np.asarray([float(np.asarray(value(q)).reshape(-1)[0]) for q in q_s], dtype=float)
        mode = str(mode).lower()
        if mode == "bhp":
            return float(np.asarray(value(bhp)).reshape(-1)[0])
        if mode in ("rate", "nrat", "vrat"):
            return float(np.sum(rates))
        if mode == "wrat":
            return float(rates[0]) if rates.size else 0.0
        if mode == "orat":
            return float(rates[1]) if rates.size > 1 else 0.0
        if mode == "grat":
            return float(rates[2]) if rates.size > 2 else 0.0
        if mode == "lrat":
            return float(rates[0] + (rates[1] if rates.size > 1 else 0.0)) if rates.size else 0.0
        raise ValueError(f"unsupported well control mode {mode!r}")

    def _violated_limit(self, wellSol: Any, q_s: Sequence[Any], bhp: Any) -> str | None:
        lims = getattr(self.W, "lims", None) or {}
        if not lims:
            return None
        active_type = str(self._container_field(wellSol, "type", self.W.type)).lower()
        if self.isInjector():
            modes = ("bhp", "rate", "vrat")
            for mode in modes:
                if mode == active_type or mode not in lims:
                    continue
                current = self._mode_value(mode, q_s, bhp)
                target = float(lims[mode])
                if (mode == "vrat" and current < target) or (mode != "vrat" and current > target):
                    return mode
        else:
            modes = ("bhp", "orat", "lrat", "grat", "wrat", "vrat")
            for mode in modes:
                if mode == active_type or mode not in lims:
                    continue
                current = self._mode_value(mode, q_s, bhp)
                target = float(lims[mode])
                if (mode == "vrat" and current > target) or (mode != "vrat" and current < target):
                    return mode
        return None

    def updateLimits(self, wellSol0: Any, wellSol: Any, model: Any,
                     q_s: Sequence[Any], bhp: Any, packed: Any,
                     dt: float | None, iteration: float | int | None):
        del wellSol0, model, packed, dt, iteration
        within_limits = True
        if not self.allowControlSwitching or not bool(getattr(self.W, "status", True)):
            return q_s, bhp, wellSol, within_limits
        violated = self._violated_limit(wellSol, q_s, bhp)
        if violated is None:
            return q_s, bhp, wellSol, within_limits
        within_limits = False
        self.W.type = violated
        self.W.val = float((getattr(self.W, "lims", {}) or {})[violated])
        self._set_container_field(wellSol, "type", violated)
        self._set_container_field(wellSol, "val", self.W.val)
        if violated == "bhp":
            bhp = np.asarray([self.W.val], dtype=float)
            self._set_container_field(wellSol, "bhp", self.W.val)
        elif violated in ("rate", "nrat", "vrat"):
            compi = np.asarray(getattr(self.W, "compi", []), dtype=float).reshape(-1)
            q_s = [np.asarray([self.W.val * float(compi[ix])]) for ix in range(min(len(q_s), compi.size))]
        elif violated == "wrat" and len(q_s) > 0:
            q_s = list(q_s)
            q_s[0] = np.asarray([self.W.val], dtype=float)
        elif violated == "orat" and len(q_s) > 1:
            q_s = list(q_s)
            q_s[1] = np.asarray([self.W.val], dtype=float)
        elif violated == "grat" and len(q_s) > 2:
            q_s = list(q_s)
            q_s[2] = np.asarray([self.W.val], dtype=float)
        return q_s, bhp, wellSol, within_limits

    update_limits = updateLimits

    def updateWellSol(self, wellSol: Any, variables: Sequence[str], dx: Sequence[Any], resmodel: Any = None):
        """Apply MRST-style increments to primary variables in ``wellSol``."""
        for name, increment in zip(variables, dx):
            field, index = self.getVariableField(name, strict=False)
            old = self._container_field(wellSol, field, 0.0)
            new_value = np.asarray(value(old), dtype=float) + np.asarray(value(increment), dtype=float)
            if str(name).lower() == "bhp":
                new_value = np.asarray(old, dtype=float) + np.clip(
                    np.asarray(value(increment), dtype=float),
                    -float(self.dpMaxAbs),
                    float(self.dpMaxAbs),
                )
            if field == "massfractions":
                new_value = np.clip(new_value, 0.0, 1.0)
                total = float(np.sum(new_value))
                if total > 0.0:
                    new_value = new_value / total
            if index == 0 and np.asarray(new_value).size == 1:
                new_value = float(np.asarray(new_value).reshape(-1)[0])
            self._set_container_field(wellSol, field, new_value)
        return wellSol

    update_well_sol = updateWellSol

    def updateWellSolAfterStep(self, resmodel: Any, wellSol: Any, wellSol0: Any):
        """Book-keeping after a converged step, including sign-change shutdown."""
        del resmodel
        well_shut = False
        if not bool(getattr(self.W, "status", True)):
            return wellSol, well_shut
        total_rate = float(self.getTotalRate(wellSol)[0]) if self.getTotalRate(wellSol).size else 0.0
        schedule_sign = int(getattr(self.W, "sign", 0))
        solved_sign = int(np.sign(total_rate))
        closed = schedule_sign != solved_sign and schedule_sign != 0
        closed = closed and str(getattr(self.W, "type", "")).lower() != "bhp" and float(getattr(self.W, "val", 0.0)) != 0.0
        if closed and not self.allowSignChange and self.allowControlSwitching:
            self._set_container_field(wellSol, "status", False)
            well_shut = True
        if self._container_field(wellSol, "type", None) != self._container_field(wellSol0, "type", None):
            self._set_container_field(wellSol, "switched", True)
        return wellSol, well_shut

    update_well_sol_after_step = updateWellSolAfterStep

    def ensureWellSolConsistency(self, wellSol: Any):
        return wellSol

    ensure_well_sol_consistency = ensureWellSolConsistency

    def _connection_weights(self) -> np.ndarray:
        nperf = np.asarray(self.W.cells).size
        if nperf == 0:
            return np.zeros(0)
        cstatus = np.ones(nperf, dtype=bool) if self.W.cstatus is None else np.asarray(self.W.cstatus, dtype=bool)
        wi = np.asarray(self.W.WI, dtype=float).reshape(-1)
        if wi.size != nperf or np.sum(np.abs(wi[cstatus])) <= 0.0:
            weights = cstatus.astype(float)
        else:
            weights = np.where(cstatus, np.abs(wi), 0.0)
        if weights.sum() <= 0.0:
            weights = np.ones(nperf, dtype=float)
        return weights / weights.sum()

    def computeWellEquations(self, wellSol0: Any, wellSol: Any, resmodel: Any,
                             q_s: Sequence[Any], bh: Any, packed: Any,
                             dt: float | None, iteration: float | int | None):
        del wellSol0, packed, dt, iteration
        phase_names = self._phase_names(resmodel)
        nphase = len(q_s)
        weights = self._connection_weights()
        q_vals = [np.asarray(value(q), dtype=float).reshape(-1)[0] for q in q_s]
        qVol = [weights * float(q_vals[ph]) for ph in range(nphase)]
        qMass = [arr.copy() for arr in qVol]
        weqs = [np.asarray([np.sum(qVol[ph]) - float(q_vals[ph])]) for ph in range(nphase)]
        control = str(self._container_field(wellSol, "type", self.W.type)).lower()
        target = float(self._container_field(wellSol, "val", self.W.val))
        if control == "bhp":
            ctrl_eq = np.asarray([float(np.asarray(value(bh)).reshape(-1)[0]) - target])
        else:
            ctrl_eq = np.asarray([self._mode_value(control, q_s, bh) - target])
        cqs = np.zeros((weights.size, nphase))
        for ph in range(nphase):
            cqs[:, ph] = qVol[ph]
            if ph < len(phase_names):
                self._set_container_field(wellSol, f"q{phase_names[ph].upper()}s", float(q_vals[ph]))
        self._set_container_field(wellSol, "cqs", cqs)
        self._set_container_field(wellSol, "cstatus", weights > 0.0)
        self._set_container_field(wellSol, "status", bool(getattr(self.W, "status", True)))
        return weqs, ctrl_eq, [], [], qMass, qVol, wellSol

    compute_well_equations = computeWellEquations

    def computeComponentContributions(self, wellSol0: Any, wellSol: Any, resmodel: Any,
                                      q_s: Sequence[Any], bh: Any, packed: Any,
                                      qMass: Sequence[Any], qVol: Sequence[Any],
                                      dt: float | None, iteration: float | int | None):
        result = self._call(
            resmodel,
            ("getExtraWellContributions", "get_extra_well_contributions"),
            self,
            wellSol0,
            wellSol,
            q_s,
            bh,
            packed,
            qMass,
            qVol,
            dt,
            iteration,
            default=None,
        )
        if result is not None:
            return result
        ncomp = self._number_of_components(resmodel)
        components = np.asarray(
            self._container_field(wellSol, "components", self.W.components if self.W.components is not None else self.W.compi),
            dtype=float,
        ).reshape(-1)
        if components.size < ncomp:
            components = np.pad(components, (0, ncomp - components.size))
        components = components[:ncomp]
        total_mass = np.zeros_like(np.asarray(qMass[0], dtype=float)) if qMass else np.zeros(0)
        for item in qMass:
            total_mass = total_mass + np.asarray(item, dtype=float)
        comp_src = [total_mass * float(components[ix]) for ix in range(ncomp)]
        return [], comp_src, [], wellSol

    compute_component_contributions = computeComponentContributions
