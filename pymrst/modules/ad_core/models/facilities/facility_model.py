"""MRST `ad-core/models/facilities/FacilityModel.m` well model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import ad_sum, value, vertcat_adi


class FacilityLinearizedProblem:
    """Small MRST-like equation container for standalone facility solves."""

    def __init__(
        self,
        equations: Sequence[Any],
        equation_names: Sequence[str],
        types: Sequence[str],
        primary_variables: Sequence[str],
        state: Any,
        *,
        sources: Any = None,
        well_system: Any = None,
        iteration_no: float | int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.equations = list(equations)
        self.equationNames = [str(item) for item in equation_names]
        self.equation_names = self.equationNames
        self.types = [str(item) for item in types]
        self.primaryVariables = [str(item) for item in primary_variables]
        self.primary_variables = self.primaryVariables
        self.state = state
        self.sources = sources
        self.wellSystem = well_system
        self.well_system = well_system
        self.iterationNo = iteration_no
        self.iteration_no = iteration_no
        self.metadata = metadata or {}
        self.A = None
        self.b: np.ndarray | None = None
        self.assembled = False

    def indexOfType(self, name: str) -> np.ndarray:
        key = str(name).lower()
        return np.asarray([str(item).lower() == key for item in self.types], dtype=bool)

    index_of_type = indexOfType

    def assembleSystem(self):
        rows = [np.asarray(value(eq), dtype=float).reshape(-1) for eq in self.equations]
        self.b = np.concatenate(rows) if rows else np.zeros(0)
        self.A = None
        self.assembled = True
        return self

    assemble_system = assembleSystem

    def norm(self, ord=np.inf) -> float:
        rows = [np.asarray(value(eq), dtype=float).reshape(-1) for eq in self.equations]
        if not rows:
            return 0.0
        return float(np.linalg.norm(np.concatenate(rows), ord=ord))


class FacilityModel:
    """The wells (MRST ad-core/facilities).

    One well model, generic over phase count and dissolved gas - the same
    algorithm the two hand-rolled models each carried a copy of.
    """

    def __init__(self, wells: Sequence[Well] | Well | Any = (), disgas: bool = False,
                 verbose: bool = False, vapoil: bool = False,
                 vfp_tables_injector: Sequence[Any] = (),
                 vfp_tables_producer: Sequence[Any] = (),
                 group_controls: Sequence[Any] = (),
                 primary_variable_set: str = "standard"):
        reservoir_model = None
        if wells is None:
            well_list: list[Well] = []
        elif isinstance(wells, Well):
            well_list = [wells]
        else:
            try:
                candidate = list(wells)
            except TypeError:
                reservoir_model = wells
                well_list = []
            else:
                well_list = candidate
        self.wells = well_list
        self.disgas = disgas
        self.vapoil = vapoil
        self.verbose = verbose
        self.vfp_tables_injector = list(vfp_tables_injector)
        self.vfp_tables_producer = list(vfp_tables_producer)
        self.group_controls = list(group_controls)
        self.primary_variable_set = str(primary_variable_set)
        self.setWellTargets = True
        self.tolerance_well_bhp = 1e5
        self.tolerance_well_rate = 1.0 / 86400.0
        self.tolerance_well_ms = 1e-6
        self.toleranceWellBHP = self.tolerance_well_bhp
        self.toleranceWellRate = self.tolerance_well_rate
        self.toleranceWellMS = self.tolerance_well_ms
        self._bind_vfp_tables()
        # The connection rates of the last assembled state - MRST's
        # `wellSol.cqs`. A converged step passes these to the next one.
        self.last_flux: Optional[List[np.ndarray]] = None
        # Set false whenever a well has just changed control mode: the Newton
        # step it happened in is solving a different problem from the one it
        # started, so it cannot be declared converged.
        self.within_limits = True
        self.last_group_switches: list[dict[str, Any]] = []
        self.WellModels: list[Any] = []
        self.well_models = self.WellModels
        self.addedPrimaryVarNames: list[str] = []
        self.addedPrimaryVarNamesIsFromResModel: list[bool] = []
        self.addedEquationNames: list[str] = []
        self.addedEquationTypes: list[str] = []
        self.FacilityFlowDiscretization = None
        if reservoir_model is not None:
            self.set_reservoir_model(reservoir_model)

    def setup_state_function_groupings(self, use_defaults: bool | None = None):
        """MRST ``setupStateFunctionGroupings`` compatibility hook.

        PyMRST's generic facility model does not need a separate facility
        state-function grouping for the currently implemented well equations,
        but keeping the hook and storage slot makes exact ports able to call
        the same setup chain.
        """
        if use_defaults is None:
            use_defaults = self.FacilityFlowDiscretization is None
        if use_defaults and self.FacilityFlowDiscretization is None:
            self.FacilityFlowDiscretization = {"model": self, "status": "placeholder"}
        return self

    setupStateFunctionGroupings = setup_state_function_groupings

    def remove_state_function_groupings(self):
        """Clear facility state-function grouping storage."""
        self.FacilityFlowDiscretization = None
        return self

    removeStateFunctionGroupings = remove_state_function_groupings

    def set_reservoir_model(self, resmodel: Any):
        """Attach the parent reservoir model, mirroring MRST ``setReservoirModel``."""
        self.ReservoirModel = resmodel
        self.reservoir_model = resmodel
        backend = getattr(resmodel, "AutoDiffBackend", getattr(resmodel, "auto_diff_backend", None))
        if backend is not None:
            self.AutoDiffBackend = backend
            self.auto_diff_backend = backend
        return self

    setReservoirModel = set_reservoir_model

    @staticmethod
    def _driving_force_wells(driving_forces: Any):
        if driving_forces is None:
            return None
        if isinstance(driving_forces, dict):
            return driving_forces.get("W", driving_forces.get("wells", None))
        return getattr(driving_forces, "W", getattr(driving_forces, "wells", None))

    @staticmethod
    def _metadata_list(items: Any) -> list[Any]:
        if items is None:
            return []
        if isinstance(items, str):
            return [items]
        if isinstance(items, np.ndarray):
            return list(items.ravel())
        try:
            return list(items)
        except TypeError:
            return [items]

    @staticmethod
    def _metadata_call(method_owner: Any, method_names: Sequence[str], *args):
        for method_name in method_names:
            if not hasattr(method_owner, method_name):
                continue
            method = getattr(method_owner, method_name)
            try:
                return method(*args)
            except TypeError:
                return method()
        return None

    def refresh_added_well_metadata(self):
        """Collect extra primary variables/equations from attached well models.

        MRST's ``setupWells`` lets individual well models advertise additional
        primary variables and equations, and the facility model keeps one
        canonical ordered list.  PyMRST mirrors that behavior here while
        preserving the first occurrence when several wells report the same
        name.
        """
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        primary_names: list[str] = []
        primary_flags: list[bool] = []
        equation_names: list[str] = []
        equation_types: list[str] = []
        seen_primary: set[str] = set()
        seen_equations: set[str] = set()

        for well_model in self.WellModels:
            result = self._metadata_call(
                well_model,
                ("getExtraPrimaryVariableNames", "get_extra_primary_variable_names"),
                reservoir_model,
            )
            if result is not None:
                if isinstance(result, tuple):
                    names = self._metadata_list(result[0])
                    flags = self._metadata_list(result[1]) if len(result) > 1 else []
                else:
                    names = self._metadata_list(result)
                    flags = []
                if len(flags) < len(names):
                    flags = flags + [False] * (len(names) - len(flags))
                for name, flag in zip(names, flags):
                    key = str(name).lower()
                    if key in seen_primary:
                        continue
                    seen_primary.add(key)
                    primary_names.append(str(name))
                    primary_flags.append(bool(flag))

            result = self._metadata_call(
                well_model,
                ("getExtraEquationNames", "get_extra_equation_names"),
                reservoir_model,
            )
            if result is None:
                continue
            if isinstance(result, tuple):
                names = self._metadata_list(result[0])
                types = self._metadata_list(result[1]) if len(result) > 1 else []
            else:
                names = self._metadata_list(result)
                types = []
            if len(types) < len(names):
                types = types + ["well"] * (len(names) - len(types))
            for name, eq_type in zip(names, types):
                key = str(name).lower()
                if key in seen_equations:
                    continue
                seen_equations.add(key)
                equation_names.append(str(name))
                equation_types.append(str(eq_type))

        self.addedPrimaryVarNames = primary_names
        self.addedPrimaryVarNamesIsFromResModel = primary_flags
        self.addedEquationNames = equation_names
        self.addedEquationTypes = equation_types
        return self

    refreshAddedWellMetadata = refresh_added_well_metadata

    def setup_wells(self, wells: Sequence[Well] | None = None, wellmodels: Sequence[Any] | None = None):
        """Set or refresh facility wells in MRST ``setupWells`` style.

        PyMRST stores schedule wells directly.  When explicit per-well models
        are supplied, they are retained in ``WellModels`` for advanced callers;
        otherwise the direct well list is the authoritative facility state.
        """
        if wells is None:
            wells = self.wells
        new_wells = list(wells)
        if self.wells and len(self.wells) != len(new_wells):
            raise AssertionError("Number of wells in facility model has changed during simulation")
        for ix, well in enumerate(new_wells):
            if bool(getattr(well, "status", True)) and np.asarray(getattr(well, "val", [])).size != 1:
                raise AssertionError(f"Bad controls for well {ix + 1}: One value must be specified for .val")
        self.wells = new_wells
        self._bind_vfp_tables()

        if wellmodels is None:
            self.WellModels = []
            # Direct-well PyMRST path: no extra per-well primary variables.
            self.addedPrimaryVarNames = []
            self.addedPrimaryVarNamesIsFromResModel = []
            self.addedEquationNames = []
            self.addedEquationTypes = []
        else:
            self.WellModels = list(wellmodels)
            if len(self.WellModels) != len(new_wells):
                raise AssertionError("wellmodels must have the same length as wells")
            self.refresh_added_well_metadata()
        self.well_models = self.WellModels
        return self

    setupWells = setup_wells

    def validate_model(self, driving_forces: Any = None, *_, **__):
        """MRST ``validateModel`` compatibility surface for facility wells."""
        wells = self._driving_force_wells(driving_forces)
        if wells is not None:
            self.setup_wells(wells)
        return self

    validateModel = validate_model

    def _number_of_components(self) -> int:
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        for method in ("getNumberOfComponents", "get_number_of_components"):
            if reservoir_model is not None and hasattr(reservoir_model, method):
                return int(getattr(reservoir_model, method)())
        return max((np.asarray(w.compi).size for w in self.wells), default=0)

    def _default_well_sol(self, well: Well) -> SimpleNamespace:
        nperf = np.asarray(well.cells).size
        ncomp = max(np.asarray(well.compi).size, self._number_of_components(), 1)
        phase_names = self._phase_names()
        data: dict[str, Any] = {
            "name": well.name,
            "status": bool(well.status),
            "type": well.type,
            "val": float(np.asarray(well.val).reshape(-1)[0]),
            "sign": int(well.sign),
            "bhp": float(np.asarray(well.val).reshape(-1)[0]),
            "pressure": float(np.asarray(well.val).reshape(-1)[0]),
            "flux": np.zeros(nperf),
            "cdp": np.zeros(nperf),
            "components": np.asarray(well.components if well.components is not None else well.compi, dtype=float).copy(),
            "massfractions": np.ones(ncomp, dtype=float) / ncomp,
        }
        for phase in phase_names:
            data[f"q{phase.upper()}s"] = 0.0
        if well.type.lower() in ("rate", "wrat", "orat", "grat", "lrat", "vrat", "nrat"):
            compi = np.asarray(well.compi, dtype=float).ravel()
            total = float(np.asarray(well.val).reshape(-1)[0])
            for ix, phase in enumerate(phase_names[:compi.size]):
                data[f"q{phase.upper()}s"] = total * float(compi[ix])
        return SimpleNamespace(**data)

    def validate_state(self, state: Any):
        """Ensure a state carries MRST-like ``wellSol`` rows."""
        well_sol = self._state_field(state, "wellSol", None)
        missing = well_sol is None or len(list(well_sol)) == 0
        if not missing:
            first = list(well_sol)[0]
            missing = self._wellsol_field(first, "bhp", None) is None
        if missing:
            well_sol = [self._default_well_sol(well) for well in self.wells]
            self._set_state_field(state, "wellSol", well_sol)

        if self.primary_variable_set.lower() == "bhp_massfractions":
            ncomp = max(self._number_of_components(), 1)
            for item in self._state_field(state, "wellSol", []):
                self._set_wellsol_field(item, "massfractions", np.ones(ncomp, dtype=float) / ncomp)

        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        items = list(self._state_field(state, "wellSol", []))
        for wno, well_model in enumerate(self.WellModels):
            if wno >= len(items):
                continue
            result = self._call_well_model(
                well_model,
                ("validateWellSol", "validate_well_sol"),
                reservoir_model,
                items[wno],
                state,
            )
            if result is None:
                continue
            fields = result.items() if isinstance(result, dict) else vars(result).items()
            for key, field_value in fields:
                self._set_wellsol_field(items[wno], key, field_value)
        return state

    validateState = validate_state

    def init_state_ad(
        self,
        state: Any,
        variables: Sequence[Any],
        names: Sequence[str],
        origin: Sequence[Any] | None = None,
    ):
        """Store facility primary variables in MRST ``FacilityState`` layout."""
        del origin
        names = list(names)
        variables = list(variables)
        comp_ix = [ix for ix, name in enumerate(names) if str(name).lower() == "well_massfractions"]
        mass_fraction_var = variables[comp_ix[0]] if comp_ix else None
        keep = [ix for ix in range(len(names)) if ix not in comp_ix]
        kept_names = [names[ix] for ix in keep]
        kept_vars = [variables[ix] for ix in keep]

        well_sol = self._state_field(state, "wellSol", None)
        if well_sol is None:
            state = self.validate_state(state)
            well_sol = self._state_field(state, "wellSol", [])
        active_items = self._active_well_sol_items(well_sol)

        surface_rates = []
        for phase in self._phase_names():
            qname = f"q{phase.upper()}s"
            match = next((ix for ix, name in enumerate(kept_names) if str(name).lower() == qname.lower()), None)
            if match is not None:
                surface_rates.append(kept_vars[match])
            else:
                surface_rates.append(np.asarray([
                    self._wellsol_field(item, qname, 0.0)
                    for item in active_items
                ], dtype=float).ravel())

        facility_state: dict[str, Any] = {
            "primaryVariables": kept_vars,
            "primary_variables": kept_vars,
            "surfacePhaseRates": surface_rates,
            "surface_phase_rates": surface_rates,
            "names": kept_names,
        }
        if mass_fraction_var is not None:
            ncomp = max(self._number_of_components(), 1)
            mf = np.asarray(value(mass_fraction_var), dtype=float).ravel()
            if ncomp > 1:
                nrow = mf.size // (ncomp - 1)
                rows = mf.reshape(nrow, ncomp - 1)
                fill = 1.0 - rows.sum(axis=1)
                full = np.column_stack([rows, fill])
                facility_state["massfractions"] = [full[:, ix] for ix in range(ncomp)]
                facility_state["massfractions_matrix"] = full
            else:
                facility_state["massfractions"] = [np.ones(mf.size)]
                facility_state["massfractions_matrix"] = np.ones((mf.size, 1))
        self._set_state_field(state, "FacilityState", facility_state)
        self._set_state_field(state, "facility_state", facility_state)
        return state

    initStateAD = init_state_ad

    def update_state(self, state: Any, problem: Any = None, dx: Any = None, driving_forces: Any = None):
        """MRST ``updateState`` compatibility wrapper for well primary variables."""
        well_sol = self._state_field(state, "wellSol", None)
        if well_sol is not None and problem is not None and dx is not None:
            updated, _ = self.update_well_sol(well_sol, problem, dx, driving_forces)
            self._set_state_field(state, "wellSol", updated)
        return state, []

    updateState = update_state

    @staticmethod
    def _parse_name_value_options(args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        opts = dict(kwargs)
        if len(args) == 1 and isinstance(args[0], dict):
            opts.update(args[0])
        elif args:
            if len(args) % 2:
                raise TypeError("options must be name/value pairs or a single dictionary")
            opts.update({str(key): val for key, val in zip(args[0::2], args[1::2])})
        aliases = {
            "iteration": "iteration",
            "Iteration": "iteration",
            "resOnly": "res_only",
            "res_only": "res_only",
        }
        parsed = {"iteration": np.nan, "res_only": False}
        for key, val in opts.items():
            parsed[aliases.get(str(key), str(key))] = val
        return parsed

    def get_equations(self, state0: Any, state: Any, dt: float, driving_forces: Any = None, *args, **kwargs):
        """Standalone MRST-style facility equation assembly.

        This is the facility-only analogue of MRST ``FacilityModel.getEquations``:
        reservoir properties are treated as fixed inputs and the returned
        problem contains well equations plus the control equation.
        """
        del driving_forces
        opts = self._parse_name_value_options(args, kwargs)
        state = self.validate_state(state)
        state0 = self.validate_state(state0)
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if reservoir_model is None:
            raise ValueError("getEquations requires an attached reservoir model")

        well_sol = self._state_field(state, "wellSol", [])
        well_sol0 = self._state_field(state0, "wellSol", well_sol)
        well_vars, primary_vars, well_map = self.get_all_primary_variables(well_sol)

        p = self._state_field(state, "pressure", None)
        if p is None:
            p = self._state_field(state, "p", None)
        if p is None:
            raise KeyError("state must contain pressure for FacilityModel.getEquations")

        mob = self._state_field(state, "mob", None)
        rho = self._state_field(state, "rho", None)
        if mob is None:
            try:
                mob = reservoir_model.get_props(state, "Mobility")
            except Exception:
                mob = None
        if rho is None:
            try:
                rho = reservoir_model.get_props(state, "Density")
            except Exception:
                rho = None

        rs = self._state_field(state, "rs", None)
        rv = self._state_field(state, "rv", None)
        if hasattr(reservoir_model, "getDissolutionMatrix"):
            dissolution = reservoir_model.getDissolutionMatrix(rs, rv)
        elif hasattr(reservoir_model, "get_dissolution_matrix"):
            dissolution = reservoir_model.get_dissolution_matrix(rs, rv)
        else:
            dissolution = None

        components = self._state_field(state, "components", None)
        if components is None:
            components = self._state_field(state, "z", None)
        if isinstance(components, np.ndarray) and components.ndim == 2:
            components = [components[:, ix] for ix in range(components.shape[1])]
        elif components is None:
            ncomp = int(getattr(reservoir_model, "getNumberOfComponents", lambda: 0)())
            components = [np.zeros_like(np.asarray(value(p), dtype=float).reshape(-1)) for _ in range(ncomp)]

        sources, well_system, updated_well_sol = self.get_well_contributions(
            well_sol0,
            well_sol,
            well_vars,
            well_map,
            p,
            mob,
            rho,
            dissolution,
            components,
            dt,
            opts["iteration"],
            state=state,
            model=reservoir_model,
        )
        self._set_state_field(state, "wellSol", updated_well_sol)

        equations = list(well_system.wellEquations) + [well_system.controlEquation]
        equation_names = list(well_system.names) + ["control"]
        types = list(well_system.types) + ["well"]
        problem = FacilityLinearizedProblem(
            equations,
            equation_names,
            types,
            primary_vars,
            state,
            sources=sources,
            well_system=well_system,
            iteration_no=opts["iteration"],
            metadata={"res_only": bool(opts["res_only"]), "model": self},
        )
        return problem, state

    getEquations = get_equations

    @property
    def n_wells(self) -> int:
        return len(self.wells)

    @property
    def primaryVariableSet(self) -> str:
        """MRST-compatible alias for ``primary_variable_set``."""
        return self.primary_variable_set

    @primaryVariableSet.setter
    def primaryVariableSet(self, value: str) -> None:
        self.primary_variable_set = str(value)

    @property
    def VFPTablesInjector(self) -> list[Any]:
        """MRST-compatible alias for injector VFP tables."""
        return self.vfp_tables_injector

    @VFPTablesInjector.setter
    def VFPTablesInjector(self, value: Sequence[Any]) -> None:
        self.vfp_tables_injector = list(value)
        self._bind_vfp_tables()

    @property
    def VFPTablesProducer(self) -> list[Any]:
        """MRST-compatible alias for producer VFP tables."""
        return self.vfp_tables_producer

    @VFPTablesProducer.setter
    def VFPTablesProducer(self, value: Sequence[Any]) -> None:
        self.vfp_tables_producer = list(value)
        self._bind_vfp_tables()

    @staticmethod
    def _container_field(container: Any, name: str, default: Any = None) -> Any:
        if container is None:
            return default
        if isinstance(container, dict):
            return container.get(name, default)
        return getattr(container, name, default)

    def _phase_names(self) -> list[str]:
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if reservoir_model is not None:
            for method in ("getPhaseNames", "get_phase_names"):
                if hasattr(reservoir_model, method):
                    return [str(x) for x in getattr(reservoir_model, method)()]
            if hasattr(reservoir_model, "phase_names"):
                return [str(x) for x in reservoir_model.phase_names]
        nph = max((np.asarray(w.compi).size for w in self.wells), default=3)
        default = ["W", "O", "G"]
        return default[:nph] if nph <= len(default) else default + [
            f"P{i}" for i in range(len(default) + 1, nph + 1)
        ]

    def get_variable_field(self, name: str, strict: bool = True):
        """Map MRST facility variable names to ``wellSol`` fields.

        MRST recognizes ``bhp`` and phase surface rates (``qWs``, ``qOs``,
        ``qGs``...) directly on the facility layer.  Unknown names can either
        fail loudly or be returned as direct field names for compatibility with
        the base-model fallback path.
        """
        key = str(name)
        if key.lower() == "bhp":
            return "bhp", slice(None)
        qnames = {f"q{phase.upper()}s".lower(): f"q{phase.upper()}s" for phase in self._phase_names()}
        if key.lower() in qnames:
            return qnames[key.lower()], slice(None)
        if strict:
            raise KeyError(f"unknown facility variable field {name!r}")
        return key, slice(None)

    getVariableField = get_variable_field

    @staticmethod
    def _select_variable_index(field_value: Any, index: Any) -> np.ndarray:
        arr = np.asarray(value(field_value))
        if isinstance(index, slice):
            return arr
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr if int(index) == 0 else arr[int(index):int(index) + 1]
        return arr[:, int(index)]

    def _facility_state_primary(self, state: Any, name: str):
        facility_state = self._container_field(state, "FacilityState", None)
        if facility_state is None:
            facility_state = self._container_field(state, "facility_state", None)
        if facility_state is None:
            return None
        names = list(self._container_field(facility_state, "names", ()) or ())
        variables = list(
            self._container_field(
                facility_state,
                "primaryVariables",
                self._container_field(facility_state, "primary_variables", ()),
            ) or ()
        )
        for candidate, variable in zip(names, variables):
            if str(candidate).lower() == str(name).lower():
                return variable
        return None

    def _well_sol_values(self, state: Any, field: str, index: Any):
        well_sol = self._container_field(state, "wellSol", None)
        if well_sol is None:
            well_sol = self._container_field(state, "well_sol", None)
        if well_sol is None:
            return None
        items = list(well_sol)
        rows: list[np.ndarray] = []
        for item, active in zip(items, self.get_well_status_mask(items)):
            if not active:
                continue
            field_value = self._wellsol_field(item, field, None)
            if field_value is None:
                return None
            rows.append(self._select_variable_index(field_value, index).reshape(-1))
        if not rows:
            return np.zeros(0)
        return np.concatenate(rows)

    def get_prop(self, state: Any, name: str):
        """Return a facility property using MRST's lookup priority."""
        key = str(name)
        if key.lower() in ("surfacerates", "q_s"):
            return [self.get_prop(state, f"q{phase.upper()}s") for phase in self._phase_names()]

        primary = self._facility_state_primary(state, key)
        if primary is not None:
            return primary

        field, index = self.get_variable_field(key, strict=False)
        primary = self._facility_state_primary(state, field)
        if primary is not None:
            return primary

        well_values = self._well_sol_values(state, field, index)
        if well_values is not None:
            return well_values

        direct = self._container_field(state, field, None)
        if direct is not None:
            return direct
        raise KeyError(f"'{name}' is neither a facility primary variable nor a wellSol field")

    getProp = get_prop

    def get_props(self, state: Any, *names: str):
        out = [self.get_prop(state, name) for name in names]
        return out[0] if len(out) == 1 else out

    getProps = get_props

    def get_number_of_wells(self) -> int:
        """Return the number of wells known to the facility model.

        Mirrors MRST ``FacilityModel.getNumberOfWells``.  PyMRST stores the
        well structs directly rather than inside per-well model objects, so the
        count is simply the length of ``self.wells``.
        """
        return self.n_wells

    getNumberOfWells = get_number_of_wells

    def get_well_struct(self, subs: Sequence[int] | slice | None = None) -> list[Well]:
        """Return well structs in facility order.

        This is the Python counterpart of MRST ``getWellStruct``.  ``subs`` may
        be ``None``/``slice``/an index sequence and follows Python's zero-based
        indexing conventions.
        """
        if subs is None:
            return list(self.wells)
        if isinstance(subs, slice):
            return list(self.wells[subs])
        return [self.wells[int(ix)] for ix in subs]

    getWellStruct = get_well_struct

    @staticmethod
    def _wellsol_status(well_sol: Any, ix: int) -> bool:
        if well_sol is None:
            return True
        try:
            item = well_sol[ix]
        except (TypeError, KeyError, IndexError):
            item = well_sol
        if isinstance(item, dict):
            return bool(item.get("status", True))
        return bool(getattr(item, "status", True))

    def get_well_status_mask(self, well_sol: Any = None) -> np.ndarray:
        """Return active status for each well.

        MRST combines the schedule well's ``W.status`` with the simulator's
        ``wellSol.status``.  PyMRST follows the same rule when a ``well_sol``
        sequence is supplied, and otherwise reports schedule status only.
        """
        return np.asarray([
            bool(w.status) and self._wellsol_status(well_sol, ix)
            for ix, w in enumerate(self.wells)
        ], dtype=bool)

    getWellStatusMask = get_well_status_mask

    def get_indices_of_active_wells(self, well_sol: Any = None) -> np.ndarray:
        """Return zero-based indices of wells active in both W and well_sol."""
        return np.flatnonzero(self.get_well_status_mask(well_sol))

    getIndicesOfActiveWells = get_indices_of_active_wells

    def get_number_of_active_wells(self, well_sol: Any = None) -> int:
        """Return the number of currently active wells."""
        return int(np.count_nonzero(self.get_well_status_mask(well_sol)))

    getNumberOfActiveWells = get_number_of_active_wells

    def _normalise_well_subs(self, subs: Sequence[int] | slice | None = None) -> list[int]:
        if subs is None:
            return list(range(len(self.wells)))
        if isinstance(subs, slice):
            return list(range(len(self.wells)))[subs]
        return [int(ix) for ix in subs]

    def get_well_cells(self, subs: Sequence[int] | slice | None = None) -> np.ndarray:
        """Return perforated cells for selected wells, regardless of status."""
        indices = self._normalise_well_subs(subs)
        if not indices:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([
            np.asarray(self.wells[ix].cells, dtype=np.int64).ravel()
            for ix in indices
        ])

    getWellCells = get_well_cells

    def get_active_well_cells(self, well_sol: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """Return cells in active wells and a perforation-to-well map.

        The first output mirrors MRST ``getActiveWellCells``.  The second
        output maps each returned perforation row to the zero-based well index
        it came from, which is the Python analogue of MRST's ``p2w`` map.
        """
        cells: list[np.ndarray] = []
        p2w: list[np.ndarray] = []
        for ix in self.get_indices_of_active_wells(well_sol):
            wc = np.asarray(self.wells[int(ix)].cells, dtype=np.int64).ravel()
            cells.append(wc)
            p2w.append(np.full(wc.size, int(ix), dtype=np.int64))
        if not cells:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        return np.concatenate(cells), np.concatenate(p2w)

    getActiveWellCells = get_active_well_cells

    def get_basic_primary_variable_names(self, phase_names: Sequence[str] | None = None) -> list[str]:
        """Return MRST's standard well primary variables.

        MRST builds ``qWs``, ``qOs`` and friends from reservoir phase names,
        followed by ``bhp``.  If no phase names are supplied, use W/O/G up to
        the largest well composition length present.
        """
        variable_set = self.primary_variable_set.lower()
        if variable_set in ("none", ""):
            return []
        if variable_set == "bhp":
            return ["bhp"]
        if variable_set == "bhp_massfractions":
            return ["bhp", "well_massfractions"]
        if variable_set not in ("standard", "explicit"):
            raise ValueError(f"invalid facility primary variable set '{self.primary_variable_set}'")
        if phase_names is None:
            phase_names = self._phase_names()
        return [f"q{str(name).upper()}s" for name in phase_names] + ["bhp"]

    getBasicPrimaryVariableNames = get_basic_primary_variable_names

    def get_primary_variable_names(self, phase_names: Sequence[str] | None = None) -> list[str]:
        """Return facility primary variable names in MRST order."""
        return self.get_basic_primary_variable_names(phase_names) + list(self.addedPrimaryVarNames)

    getPrimaryVariableNames = get_primary_variable_names

    @staticmethod
    def _wellsol_field(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    def _active_well_sol_items(self, well_sol: Any = None) -> list[Any]:
        if well_sol is None:
            return [
                {"status": bool(w.status)}
                for w in self.wells
            ]
        if isinstance(well_sol, dict):
            # State-style dictionaries may carry a `wellSol` member.
            if "wellSol" in well_sol:
                well_sol = well_sol["wellSol"]
            else:
                return [well_sol] if self._wellsol_status(well_sol, 0) else []
        items = list(well_sol)
        mask = self.get_well_status_mask(items)
        return [item for item, active in zip(items, mask) if active]

    def get_basic_primary_variables(
        self,
        well_sol: Any,
        phase_names: Sequence[str] | None = None,
    ) -> tuple[list[np.ndarray], list[str], dict[str, np.ndarray]]:
        """Return active well primary variables in MRST order.

        Outputs mirror MRST ``getBasicPrimaryVariables``: a list of vectors,
        the corresponding names, and a map marking BHP versus rate variables.
        """
        names = self.get_basic_primary_variable_names(phase_names)
        active_items = self._active_well_sol_items(well_sol)
        if not active_items or not names:
            empty = np.zeros(0, dtype=bool)
            return [], [], {"isBHP": empty, "isRate": empty}

        variables: list[np.ndarray] = []
        is_bhp = np.zeros(len(names), dtype=bool)
        for ix, name in enumerate(names):
            if name == "well_massfractions":
                rows = [
                    np.asarray(self._wellsol_field(item, "massfractions", []), dtype=float).ravel()
                    for item in active_items
                ]
                if rows and rows[0].size:
                    variables.append(np.concatenate([row[:-1] for row in rows]))
                else:
                    variables.append(np.zeros(0))
            else:
                variables.append(np.asarray([
                    self._wellsol_field(item, name, 0.0)
                    for item in active_items
                ], dtype=float).ravel())
            if name == "bhp":
                is_bhp[ix] = True

        return variables, names, {"isBHP": is_bhp, "isRate": ~is_bhp}

    getBasicPrimaryVariables = get_basic_primary_variables

    @staticmethod
    def _call_well_model(method_owner: Any, method_names: Sequence[str], *args):
        for method in method_names:
            if hasattr(method_owner, method):
                return getattr(method_owner, method)(*args)
        return None

    def get_extra_primary_variables(self, well_sol: Any):
        """Return MRST-style extra well primary variables.

        Basic PyMRST wells do not add extra variables.  If explicit per-well
        model objects are attached through ``setupWells(..., wellmodels=...)``,
        this method asks those models for their extra variables and stacks them
        in the canonical ``addedPrimaryVarNames`` order.
        """
        active_ix = self.get_indices_of_active_wells(well_sol)
        names = list(self.addedPrimaryVarNames)
        if active_ix.size == 0 or not names:
            return [], [], np.zeros((0, 0), dtype=np.int64)

        variables_by_name: dict[str, list[np.ndarray]] = {name: [] for name in names}
        wellmap = np.zeros((int(active_ix.size), len(names)), dtype=np.int64)
        items = list(well_sol)
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))

        for row, well_ix in enumerate(active_ix):
            well_ix = int(well_ix)
            if well_ix >= len(self.WellModels):
                continue
            result = self._call_well_model(
                self.WellModels[well_ix],
                ("getExtraPrimaryVariables", "get_extra_primary_variables"),
                items[well_ix],
                reservoir_model,
            )
            if result is None:
                continue
            values, value_names = result[:2]
            for value_k, name_k in zip(values, value_names):
                for name_ix, canonical in enumerate(names):
                    if str(canonical).lower() == str(name_k).lower():
                        variables_by_name[canonical].append(np.asarray(value(value_k), dtype=float).ravel())
                        wellmap[row, name_ix] = name_ix + 1
                        break

        variables = [
            np.concatenate(rows) if rows else np.zeros(0)
            for rows in variables_by_name.values()
        ]
        return variables, names, wellmap

    getExtraPrimaryVariables = get_extra_primary_variables

    def get_all_primary_variables(self, well_sol: Any):
        """Return basic and extra facility primary variables in MRST order."""
        basic, bnames, wellmap = self.get_basic_primary_variables(well_sol)
        extra, enames, extra_map = self.get_extra_primary_variables(well_sol)
        out_map = {
            "isBHP": np.concatenate([
                np.asarray(wellmap.get("isBHP", np.zeros(0, dtype=bool)), dtype=bool),
                np.zeros(len(extra), dtype=bool),
            ]),
            "isRate": np.concatenate([
                np.asarray(wellmap.get("isRate", np.zeros(0, dtype=bool)), dtype=bool),
                np.zeros(len(extra), dtype=bool),
            ]),
            "extraMap": extra_map,
        }
        return basic + extra, bnames + enames, out_map

    getAllPrimaryVariables = get_all_primary_variables

    def get_primary_variables(self, state: State | dict[str, Any]) -> tuple[list[np.ndarray], list[str], list[str]]:
        """Return facility primary variables from a state carrying ``wellSol``."""
        well_sol = state.get("wellSol", None) if isinstance(state, dict) else getattr(state, "wellSol", None)
        variables, names, _ = self.get_all_primary_variables(well_sol)
        return variables, names, [type(self).__name__] * len(names)

    getPrimaryVariables = get_primary_variables

    def _well_variable_count(self, well: Well, name: str) -> int:
        key = str(name).lower()
        basic = {item.lower() for item in self.get_basic_primary_variable_names()}
        if key in basic:
            return 1
        if key == "well_massfractions":
            return max(int(np.asarray(well.compi).size) - 1, 0)
        for name in self.addedPrimaryVarNames:
            if key == str(name).lower():
                return 1
        return 0

    def get_well_variable_map(self, name: str, well_sol: Any = None) -> np.ndarray:
        """Map stacked variable rows to active wells.

        MRST's ``getWellVariableMap`` returns, for a variable name such as
        ``qOs`` or ``bhp``, the active well that owns each row of the stacked
        variable vector.  PyMRST returns zero-based active-well ordinals.
        """
        owners: list[int] = []
        for active_ordinal, well_ix in enumerate(self.get_indices_of_active_wells(well_sol)):
            count = self._well_variable_count(self.wells[int(well_ix)], name)
            owners.extend([active_ordinal] * count)
        return np.asarray(owners, dtype=np.int64)

    getWellVariableMap = get_well_variable_map

    def _bind_vfp_tables(self) -> None:
        for w in self.wells:
            ix = getattr(w, "vfp_index", None)
            if ix is None or int(ix) <= 0:
                continue
            tables = self.vfp_tables_injector if w.sign > 0 else self.vfp_tables_producer
            pos = int(ix) - 1
            if pos < 0 or pos >= len(tables):
                kind = "injector" if w.sign > 0 else "producer"
                raise IndexError(f"{kind} VFP table index {ix} is not available for well {w.name!r}")
            if tables[pos] is None:
                kind = "injector" if w.sign > 0 else "producer"
                raise IndexError(f"{kind} VFP table index {ix} is empty for well {w.name!r}")
            w.vfp_table = tables[pos]

    # -- control switching ---------------------------------------------------
    def update_switched_controls(self, state: State,
                                 q_names: Sequence[str]) -> bool:
        """Switch any well that has violated a limit (MRST's
        `updateSwitchedWellControls`).

        A real schedule is not "inject at 100 m3/day" but "inject at
        100 m3/day *unless* that needs more than 300 bar, in which case hold
        300 bar". A well runs on one control until it breaks a limit, then
        switches to the limit it broke. Without this a rate-controlled
        injector will happily demand an unphysical bottom-hole pressure.

        Rates are signed - production is negative - so a producer violates a
        rate limit by going *below* it, which is why the comparisons look
        upside down. Absent limits are never violated.

        Returns True if every well is within its limits.
        """
        bhp = np.asarray(value(state["bhp"]), dtype=float).ravel()
        q = np.column_stack([np.asarray(value(state[n]), float).ravel()
                             for n in q_names])

        all_ok = True
        self.last_group_switches = []
        for k, w in enumerate(self.wells):
            if not w.status or not w.lims:
                continue
            mode = self._violated_limit(w, bhp[k], q[k])
            if mode is None:
                continue
            all_ok = False
            if self.verbose:
                print(f"  well {w.name}: control changed from "
                      f"{w.type} to {mode}")
            w.type = mode
            w.val = float(w.lims[mode])

        if self.group_controls:
            all_ok = self._update_group_controls(q) and all_ok

        self.within_limits = all_ok
        return all_ok

    @staticmethod
    def _problem_list(problem: Any, name: str, default=()):
        if isinstance(problem, dict):
            value_k = problem.get(name, default)
        else:
            value_k = getattr(problem, name, default)
        if value_k is None:
            return []
        return list(value_k)

    @staticmethod
    def _problem_index_of_type(problem: Any, type_name: str, types: list[str]) -> np.ndarray:
        if hasattr(problem, "indexOfType"):
            return np.asarray(problem.indexOfType(type_name), dtype=bool).ravel()
        if hasattr(problem, "index_of_type"):
            return np.asarray(problem.index_of_type(type_name), dtype=bool).ravel()
        return np.asarray([str(t).lower() == str(type_name).lower() for t in types], dtype=bool)

    def get_convergence_values_wells(self, problem: Any):
        """Return MRST ``getConvergenceValuesWells`` data.

        The method evaluates equation types ``perf``, ``well``, ``seg``,
        ``node`` and ``alpha``.  Values are infinity norms of residual vectors,
        tolerances follow MRST's facility tolerances, and ``evaluated`` is a
        boolean mask into the full equation list.
        """
        equations = self._problem_list(problem, "equations")
        names = self._problem_list(problem, "equationNames", None)
        if names == []:
            names = self._problem_list(problem, "equation_names", ())
        types = self._problem_list(problem, "types", ())
        if not names:
            names = [f"equation_{i}" for i in range(len(equations))]
        if not types:
            types = [""] * len(equations)

        masks = {
            "perf": self._problem_index_of_type(problem, "perf", types),
            "well": self._problem_index_of_type(problem, "well", types),
            "seg": self._problem_index_of_type(problem, "seg", types),
            "node": self._problem_index_of_type(problem, "node", types),
            "alpha": self._problem_index_of_type(problem, "alpha", types),
        }
        evaluated = np.zeros(len(equations), dtype=bool)
        for mask in masks.values():
            if mask.size != len(equations):
                raise ValueError("problem type masks must match number of equations")
            evaluated |= mask

        selected = np.flatnonzero(evaluated)
        values = np.asarray([
            np.linalg.norm(np.asarray(value(equations[ix]), dtype=float).ravel(), ord=np.inf)
            if np.asarray(value(equations[ix])).size else 0.0
            for ix in selected
        ], dtype=float)

        tolerances = np.zeros(selected.size, dtype=float)
        selected_type_names: list[str] = []
        for row, ix in enumerate(selected):
            typ = str(types[ix]).lower()
            selected_type_names.append(str(types[ix]))
            if typ in ("perf", "well"):
                tolerances[row] = float(self.toleranceWellRate)
            elif typ in ("seg", "node", "alpha"):
                tolerances[row] = float(self.toleranceWellMS)
            else:
                tolerances[row] = np.inf
        conv_names = [
            f"{names[ix]} ({selected_type_names[row]})"
            for row, ix in enumerate(selected)
        ]
        return values, conv_names, tolerances, evaluated

    getConvergenceValuesWells = get_convergence_values_wells

    def check_facility_convergence(self, problem: Any):
        """Return convergence flags and residual values for facility equations."""
        values, names, tolerances, evaluated = self.get_convergence_values_wells(problem)
        return values < tolerances, values, names, evaluated

    checkFacilityConvergence = check_facility_convergence

    def get_facility_convergence_values(self, problem: Any, *_, **__):
        """MRST-shaped facility convergence values tuple."""
        values, names, tolerances, evaluated = self.get_convergence_values_wells(problem)
        return values, tolerances, names, evaluated

    getFacilityConvergenceValues = get_facility_convergence_values

    def get_convergence_values(self, problem: Any, *_, **__):
        """MRST-shaped convergence values without evaluated mask."""
        values, names, tolerances, _ = self.get_convergence_values_wells(problem)
        return values, tolerances, names

    getConvergenceValues = get_convergence_values

    def get_valid_driving_forces(self) -> dict[str, Any]:
        """Return MRST FacilityModel driving-force keys.

        MRST's facility layer accepts wells, boundary conditions and sources as
        driving forces.  PyMRST uses ``None`` placeholders, matching the local
        model helper convention.
        """
        return {"W": None, "bc": None, "src": None}

    getValidDrivingForces = get_valid_driving_forces

    @staticmethod
    def _source_field(sources: Any, name: str, default: Any = None) -> Any:
        if isinstance(sources, dict):
            return sources.get(name, default)
        return getattr(sources, name, default)

    @staticmethod
    def _set_wellsol_field(item: Any, name: str, field_value: Any) -> None:
        if isinstance(item, dict):
            item[name] = field_value
        else:
            setattr(item, name, field_value)

    @staticmethod
    def _state_field(state: Any, name: str, default: Any = None) -> Any:
        if state is None:
            return default
        if isinstance(state, dict):
            return state.get(name, default)
        return getattr(state, name, default)

    @staticmethod
    def _set_state_field(state: Any, name: str, field_value: Any) -> None:
        if isinstance(state, dict):
            state[name] = field_value
        else:
            setattr(state, name, field_value)

    @staticmethod
    def _copy_schedule_value(field_value: Any) -> Any:
        if isinstance(field_value, np.ndarray):
            return field_value.copy()
        if isinstance(field_value, (list, tuple)):
            return np.asarray(field_value).copy()
        return field_value

    def _phase_layout(self, nphase: int, reservoir_model: Any = None) -> dict[str, int | None]:
        """Return W/O/G indices for a phase-volume source list."""
        reservoir_model = reservoir_model or getattr(self, "ReservoirModel", None)
        layout: dict[str, int | None] = {"W": None, "O": None, "G": None}
        if reservoir_model is not None and hasattr(reservoir_model, "getPhaseIndex"):
            for key in layout:
                try:
                    ix = reservoir_model.getPhaseIndex(key)
                except Exception:
                    ix = None
                if ix is not None and 0 <= int(ix) < nphase:
                    layout[key] = int(ix)
            return layout

        defaults = ["W", "O", "G"]
        for ix, key in enumerate(defaults[:nphase]):
            layout[key] = ix
        return layout

    def set_well_sol_statistics(
        self,
        well_sol: Any,
        sources: Any,
        reservoir_model: Any = None,
    ):
        """Add MRST-style derived wellSol statistics.

        Ports ``FacilityModel.setWellSolStatistics``: active wells receive
        reservoir-condition phase rates (``qWr/qOr/qGr``), total surface and
        reservoir rates (``qTs/qTr``), phase cuts, GOR, and the per-perforation
        reservoir flux matrix ``flux``.  If repeated perforations or custom
        source assembly make the source rows incompatible with the active
        perforation map, the input is returned unchanged just as MRST does.
        """
        items = list(well_sol)
        src_res = self._source_field(sources, "phaseVolume", None)
        if src_res is None:
            return well_sol
        src_res = [np.asarray(value(x), dtype=float).reshape(-1) for x in src_res]
        if not src_res:
            return well_sol

        _, p2w = self.get_active_well_cells(items)
        q_r = np.column_stack(src_res)
        if q_r.shape[0] != p2w.size:
            return well_sol

        layout = self._phase_layout(len(src_res), reservoir_model)
        for well_ix in self.get_indices_of_active_wells(items):
            well_ix = int(well_ix)
            item = items[well_ix]
            mask = p2w == well_ix
            self._set_wellsol_field(item, "qTs", 0.0)
            self._set_wellsol_field(item, "qTr", 0.0)

            phase_surface: dict[str, float] = {}
            for phase, ix in layout.items():
                if ix is None:
                    continue
                res_rate = float(np.sum(src_res[ix][mask]))
                surf_name = f"q{phase}s"
                res_name = f"q{phase}r"
                surf_rate = float(self._wellsol_field(item, surf_name, 0.0))
                self._set_wellsol_field(item, res_name, res_rate)
                self._set_wellsol_field(item, "qTr", float(self._wellsol_field(item, "qTr", 0.0)) + res_rate)
                self._set_wellsol_field(item, "qTs", float(self._wellsol_field(item, "qTs", 0.0)) + surf_rate)
                phase_surface[phase] = surf_rate

            qw = phase_surface.get("W", 0.0)
            qo = phase_surface.get("O", 0.0)
            qg = phase_surface.get("G", 0.0)
            qts = float(self._wellsol_field(item, "qTs", 0.0))
            if layout.get("W") is not None and layout.get("O") is not None:
                self._set_wellsol_field(item, "wcut", qw / (qw + qo) if (qw + qo) != 0.0 else np.nan)
            if layout.get("G") is not None:
                self._set_wellsol_field(item, "gcut", qg / qts if qts != 0.0 else np.nan)
            if layout.get("O") is not None:
                self._set_wellsol_field(item, "ocut", qo / qts if qts != 0.0 else np.nan)
            if layout.get("G") is not None and layout.get("O") is not None:
                self._set_wellsol_field(item, "gor", qg / qo if qo != 0.0 else np.nan)
            self._set_wellsol_field(item, "flux", q_r[mask, :])
        return well_sol

    setWellSolStatistics = set_well_sol_statistics

    @staticmethod
    def _strip_var_names(names: Sequence[str], remove: str) -> list[str]:
        return [name for name in names if str(name).lower() != str(remove).lower()]

    @staticmethod
    def _problem_primary_variables(problem: Any) -> list[str]:
        if isinstance(problem, dict):
            value_k = problem.get("primaryVariables", problem.get("primary_variables", ()))
        else:
            value_k = getattr(problem, "primaryVariables", getattr(problem, "primary_variables", ()))
        return list(value_k or ())

    @staticmethod
    def _increment_for_variable(dx: Any, problem: Any, name: str, ordinal: int):
        if isinstance(dx, dict):
            return np.asarray(dx[name], dtype=float).ravel()
        if hasattr(problem, "getIncrement"):
            return np.asarray(problem.getIncrement(dx, name), dtype=float).ravel()
        if hasattr(problem, "get_increment"):
            return np.asarray(problem.get_increment(dx, name), dtype=float).ravel()
        if isinstance(dx, (list, tuple)):
            return np.asarray(dx[ordinal], dtype=float).ravel()
        return np.asarray(dx, dtype=float).ravel()

    def update_well_sol(
        self,
        well_sol: Any,
        problem: Any,
        dx: Any,
        driving_forces: Any = None,
        rest_vars: Sequence[str] | None = None,
    ):
        """Update ``wellSol`` fields from stacked facility increments.

        This ports the basic-variable part of MRST ``FacilityModel.updateWellSol``.
        PyMRST applies increments directly to dict/object wellSol containers for
        active wells.  The return value is ``(well_sol, rest_vars)``.
        """
        items = list(well_sol)
        if not items:
            return well_sol, list(rest_vars or self._problem_primary_variables(problem))

        rest = list(self._problem_primary_variables(problem) if rest_vars is None else rest_vars)
        well_vars = self.get_primary_variable_names()
        active_ix = self.get_indices_of_active_wells(items)
        n_active = int(active_ix.size)
        if n_active == 0:
            return well_sol, rest

        for var_ordinal, name in enumerate(well_vars):
            if name not in rest and str(name).lower() not in {str(v).lower() for v in rest}:
                continue
            inc = self._increment_for_variable(dx, problem, name, var_ordinal)
            if name == "well_massfractions":
                if n_active == 0:
                    continue
                if inc.size % n_active != 0:
                    raise ValueError("well_massfractions increment length must divide by active well count")
                rows = inc.reshape(n_active, -1)
                rows = np.column_stack([rows, -np.sum(rows, axis=1)])
                for row, well_ix in zip(rows, active_ix):
                    item = items[int(well_ix)]
                    old = np.asarray(self._wellsol_field(item, "massfractions", np.zeros(row.size)), dtype=float)
                    self._set_wellsol_field(item, "massfractions", old + row)
                rest = self._strip_var_names(rest, name)
                continue

            owner = self.get_well_variable_map(name, items)
            if inc.size != owner.size:
                raise ValueError(f"increment for {name!r} has {inc.size} rows, expected {owner.size}")
            for active_ordinal, well_ix in enumerate(active_ix):
                sub = inc[owner == active_ordinal]
                if sub.size:
                    item = items[int(well_ix)]
                    old = float(self._wellsol_field(item, name, 0.0))
                    self._set_wellsol_field(item, name, old + float(sub[0]))
            rest = self._strip_var_names(rest, name)
        if self.setWellTargets:
            self._assign_well_values_from_control(items)
        return well_sol, rest

    updateWellSol = update_well_sol

    def _assign_well_values_from_control(self, well_sol_items: Sequence[Any]) -> None:
        """Reset directly controlled wellSol values to scheduled targets."""
        phase_names = self._phase_names()
        q_names = [f"q{str(name).upper()}s" for name in phase_names]
        active_mask = self.get_well_status_mask(well_sol_items)
        for ix, (item, well) in enumerate(zip(well_sol_items, self.wells)):
            if ix >= active_mask.size or not bool(active_mask[ix]):
                continue
            control = str(well.type).lower()
            target = float(np.asarray(well.val, dtype=float).reshape(-1)[0])
            if control == "bhp":
                self._set_wellsol_field(item, "bhp", target)
                continue
            if control in ("rate", "vrat", "nrat"):
                compi = np.asarray(well.compi, dtype=float).reshape(-1)
                if compi.size < len(q_names):
                    compi = np.pad(compi, (0, len(q_names) - compi.size))
                compi = compi[:len(q_names)]
                total = float(np.sum(compi))
                if total > 0.0:
                    compi = compi / total
                elif compi.size:
                    compi[:] = 1.0 / compi.size
                for name, fraction in zip(q_names, compi):
                    self._set_wellsol_field(item, name, target * float(fraction))
                continue
            phase_control = {
                "wrat": 0,
                "orat": 1,
                "grat": 2,
            }.get(control)
            if phase_control is not None:
                if phase_control >= len(q_names):
                    raise ValueError(f"{control} control requires at least {phase_control + 1} phases")
                self._set_wellsol_field(item, q_names[phase_control], target)
                continue
            if control == "lrat":
                indices = [i for i in (0, 1) if i < len(q_names)]
                compi = np.asarray(well.compi, dtype=float).reshape(-1)
                weights = np.asarray([
                    compi[i] if i < compi.size else 0.0
                    for i in indices
                ], dtype=float)
                total = float(np.sum(weights))
                if total > 0.0:
                    weights = weights / total
                elif weights.size:
                    weights[:] = 1.0 / weights.size
                for local_ix, weight in zip(indices, weights):
                    self._set_wellsol_field(item, q_names[local_ix], target * float(weight))

    def _update_single_well_sol_after_step(self, item: Any, well: Well) -> Any:
        """Refresh one ``wellSol`` row with schedule-control metadata."""
        existing_status = bool(self._wellsol_field(item, "status", True))
        self._set_wellsol_field(item, "status", bool(well.status) and existing_status)
        cstatus = well.cstatus
        if cstatus is None:
            cstatus = np.ones(np.asarray(well.cells).size, dtype=bool)

        for name in (
            "name",
            "type",
            "val",
            "sign",
            "compi",
            "components",
            "cells",
            "WI",
            "dZ",
            "cstatus",
            "refDepth",
            "thp",
            "vfp_index",
        ):
            if name == "cstatus":
                field_value = cstatus
            else:
                field_value = getattr(well, name, None)
            if field_value is not None:
                self._set_wellsol_field(item, name, self._copy_schedule_value(field_value))
        return item

    def update_well_sol_after_step(self, well_sol: Any, well_sol0: Any = None):
        """Update ``wellSol`` after a converged step.

        MRST delegates this hook to each per-well model.  PyMRST's generic
        facility model stores schedule wells directly, so this method performs
        the shared part of that hook: keep the converged flow fields intact
        while refreshing the MRST well-control metadata (``type``, ``val``,
        ``status``, completion status and injected composition) from the active
        schedule wells.
        """
        if well_sol is None:
            return well_sol
        items = list(well_sol)
        if len(items) != len(self.wells):
            raise ValueError("wellSol length must match number of facility wells")

        well_models = getattr(self, "WellModels", getattr(self, "well_models", None))
        reservoir_model = getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if well_models is not None:
            old_items = list(well_sol0) if well_sol0 is not None else [None] * len(items)
            for ix, item in enumerate(items):
                wm = well_models[ix]
                if hasattr(wm, "updateWellSolAfterStep"):
                    items[ix] = wm.updateWellSolAfterStep(reservoir_model, item, old_items[ix])
                elif hasattr(wm, "update_well_sol_after_step"):
                    items[ix] = wm.update_well_sol_after_step(reservoir_model, item, old_items[ix])
                else:
                    self._update_single_well_sol_after_step(item, self.wells[ix])
            return well_sol

        for item, well in zip(items, self.wells):
            self._update_single_well_sol_after_step(item, well)
        return well_sol

    updateWellSolAfterStep = update_well_sol_after_step

    def update_after_convergence(
        self,
        state0: Any,
        state: Any,
        dt: float,
        driving_forces: Any = None,
    ):
        """MRST ``FacilityModel.updateAfterConvergence`` counterpart."""
        well_sol = self._state_field(state, "wellSol", None)
        if well_sol is not None:
            well_sol0 = self._state_field(state0, "wellSol", None)
            self._set_state_field(state, "wellSol", self.update_well_sol_after_step(well_sol, well_sol0))
        return state, []

    updateAfterConvergence = update_after_convergence

    @staticmethod
    def _producer_limit_floor(target: float) -> float:
        return -abs(float(target))

    @staticmethod
    def _group_name(group: Any) -> str:
        return str(getattr(group, "name", getattr(group, "groupName", "")))

    @staticmethod
    def _group_members(group: Any) -> Sequence[Any]:
        return getattr(group, "wells", getattr(group, "members", ()))

    @staticmethod
    def _group_limits(group: Any) -> dict[str, float]:
        return dict(getattr(group, "lims", getattr(group, "limits", {})) or {})

    @staticmethod
    def _group_status(group: Any) -> bool:
        return bool(getattr(group, "status", getattr(group, "isActive", True)))

    def _well_index_by_name(self, name: str) -> int | None:
        for ix, well in enumerate(self.wells):
            if well.name == name:
                return ix
        return None

    def _group_by_name(self) -> dict[str, Any]:
        return {
            self._group_name(group): group
            for group in self.group_controls
            if self._group_name(group)
        }

    def _group_member_indices(
        self,
        item: Any,
        groups: dict[str, Any],
        seen: tuple[str, ...] = (),
    ) -> list[int]:
        if isinstance(item, (int, np.integer)):
            ix = int(item)
            return [ix] if 0 <= ix < len(self.wells) else []
        if item in groups.values():
            name = self._group_name(item)
            if name in seen:
                raise ValueError(f"cyclic facility group tree: {' -> '.join(seen + (name,))}")
            out: list[int] = []
            for child in self._group_members(item):
                out.extend(self._group_member_indices(child, groups, seen + (name,)))
            return out
        if isinstance(item, str):
            well_ix = self._well_index_by_name(item)
            if well_ix is not None:
                return [well_ix]
            group_name = item.split(":", 1)[1] if item.startswith("group:") else item
            if group_name in groups:
                return self._group_member_indices(groups[group_name], groups, seen)
        return []

    def _resolve_group_well_indices(self, group: Any) -> list[int]:
        groups = self._group_by_name()
        out: list[int] = []
        seen: set[int] = set()
        for item in self._group_members(group):
            for ix in self._group_member_indices(item, groups):
                if ix not in seen:
                    out.append(ix)
                    seen.add(ix)
        return out

    def _group_rate_component(self, q_row: np.ndarray, mode: str) -> float:
        q_w = float(q_row[0]) if q_row.size > 0 else 0.0
        q_o = float(q_row[1]) if q_row.size > 1 else 0.0
        q_g = float(q_row[2]) if q_row.size > 2 else 0.0
        if mode == "wrat":
            return q_w
        if mode == "orat":
            return q_o
        if mode == "grat":
            return q_g
        if mode == "lrat":
            return q_w + q_o
        if mode in ("rate", "nrat", "vrat"):
            return float(q_row.sum())
        raise ValueError(f"unsupported group limit '{mode}'")

    def _violated_group_limit(self, group: Any, indices: Sequence[int], q: np.ndarray) -> str | None:
        lims = self._group_limits(group)
        if not lims:
            return None
        producer = bool(indices) and all(self.wells[ix].sign < 0 for ix in indices)
        q_group = q[list(indices), :].sum(axis=0)
        modes = ("orat", "lrat", "grat", "wrat", "rate", "nrat", "vrat") if producer else (
            "wrat", "orat", "grat", "lrat", "rate", "nrat", "vrat"
        )
        for mode in modes:
            if mode not in lims:
                continue
            value_k = self._group_rate_component(q_group, mode)
            target = float(lims[mode])
            if producer:
                if value_k < self._producer_limit_floor(target):
                    return mode
            elif value_k > target:
                return mode
        return None

    def _group_allocation_weights(self, group: Any, indices: Sequence[int], mode: str, q: np.ndarray) -> np.ndarray:
        if not indices:
            return np.zeros(0)
        allocation = str(getattr(group, "allocation", "rate")).lower()
        fractions = getattr(group, "fractions", None)
        if allocation in ("equal", "uniform"):
            weights = np.ones(len(indices), dtype=float)
        elif allocation in ("fraction", "fractions", "explicit", "guide", "guided", "guide_rate", "guide_rates"):
            if fractions is None:
                extra = getattr(group, "extra", {}) if isinstance(getattr(group, "extra", {}), dict) else {}
                payload = extra.get("GUIDE", extra.get("GUIDE_ALLOCATION", {}))
                fractions = payload.get("fractions") if isinstance(payload, dict) else payload
            if fractions is None:
                weights = np.array([
                    abs(self._group_rate_component(q[ix], mode)) for ix in indices
                ], dtype=float)
            else:
                frac = np.asarray(fractions, dtype=float).ravel()
                direct_members = list(self._group_members(group))
                if frac.size == len(direct_members):
                    groups = self._group_by_name()
                    by_index = {ix: 0.0 for ix in indices}
                    for item, value_k in zip(direct_members, frac):
                        member_indices = [
                            ix for ix in self._group_member_indices(item, groups)
                            if ix in by_index
                        ]
                        if not member_indices:
                            continue
                        share = max(float(value_k), 0.0)
                        for ix in member_indices:
                            by_index[ix] += share / len(member_indices)
                    weights = np.array([by_index[ix] for ix in indices], dtype=float)
                elif frac.size == len(indices):
                    weights = np.maximum(frac, 0.0)
                else:
                    raise ValueError(
                        f"group '{self._group_name(group)}' has {frac.size} allocation fractions "
                        f"for {len(direct_members)} direct members or {len(indices)} active wells"
                    )
        elif allocation in ("rate", "current", "current_rate"):
            weights = np.array([
                abs(self._group_rate_component(q[ix], mode)) for ix in indices
            ], dtype=float)
        else:
            raise ValueError(f"unsupported group allocation '{allocation}'")
        if weights.sum() <= 0.0:
            weights = np.ones(len(indices), dtype=float)
        return weights / weights.sum()

    def _update_group_controls(self, q: np.ndarray) -> bool:
        all_ok = True
        for group in self.group_controls:
            if not self._group_status(group):
                continue
            indices = [ix for ix in self._resolve_group_well_indices(group) if self.wells[ix].status]
            if not indices:
                continue
            mode = self._violated_group_limit(group, indices, q)
            if mode is None:
                continue
            all_ok = False
            target = float(self._group_limits(group)[mode])
            producer = all(self.wells[ix].sign < 0 for ix in indices)
            signed_target = self._producer_limit_floor(target) if producer else target
            weights = self._group_allocation_weights(group, indices, mode, q)
            for ix, weight in zip(indices, weights):
                self.wells[ix].type = mode
                self.wells[ix].val = signed_target * float(weight)
            self.last_group_switches.append(
                {
                    "group": self._group_name(group),
                    "mode": mode,
                    "allocation": str(getattr(group, "allocation", "rate")),
                    "wells": [self.wells[ix].name for ix in indices],
                    "indices": [int(ix) for ix in indices],
                    "allocation_weights": [float(x) for x in weights],
                    "target": signed_target,
                }
            )
        return all_ok

    def _violated_limit(self, w: Well, bhp: float,
                        q: np.ndarray) -> Optional[str]:
        """The first limit this well breaks, other than the one it is on."""
        lims = w.lims
        qt = float(q.sum())

        if w.sign > 0:                                    # injector
            checks = [("bhp", bhp > lims.get("bhp", np.inf)),
                      ("rate", qt > lims.get("rate", np.inf))]
        else:                                             # producer
            q_w = q[0]
            q_o = q[1] if q.size > 1 else 0.0
            q_g = q[2] if q.size > 2 else 0.0
            checks = [
                ("bhp", bhp < lims.get("bhp", -np.inf)),
                ("orat", q_o < lims.get("orat", -np.inf)),
                ("lrat", q_w + q_o < lims.get("lrat", -np.inf)),
                ("grat", q_g < lims.get("grat", -np.inf)),
                ("wrat", q_w < lims.get("wrat", -np.inf)),
                ("vrat", qt > -lims.get("vrat", np.inf)),
            ]

        for mode, violated in checks:
            if mode == w.type:            # never switch to the mode you are on
                continue
            if violated and mode in lims:
                return mode
        return None

    def get_well_contributions(
        self,
        wellSol0: Any = None,
        wellSol: Any = None,
        wellvars: Sequence[Any] | None = None,
        wellMap: Any = None,
        p: Any = None,
        mob: Sequence[Any] | None = None,
        rho: Sequence[Any] | None = None,
        dissolved: Any = None,
        comp: Sequence[Any] | None = None,
        dt: float | None = None,
        iteration: float | int | None = None,
        *,
        state: Any = None,
        model: Any = None,
    ):
        """MRST-shaped gateway for well source and equation contributions.

        This mirrors the return surface of MRST ``getWellContributions`` for
        PyMRST's direct-well path.  Rich MRST per-well models still need their
        own ``computeWellEquations`` implementation; this method exposes the
        common source/well-system struct layout using the existing PyMRST well
        equation assembly.
        """
        reservoir_model = model or getattr(self, "ReservoirModel", getattr(self, "reservoir_model", None))
        if reservoir_model is None:
            raise ValueError("getWellContributions requires an attached reservoir model or model=...")

        if wellSol is None:
            wellSol = self._state_field(state, "wellSol", None)
        if wellSol is None:
            state = self.validate_state({} if state is None else state)
            wellSol = self._state_field(state, "wellSol", [])
        if wellSol0 is None:
            wellSol0 = wellSol

        if wellvars is None or wellMap is None:
            wellvars, _, wellMap = self.get_all_primary_variables(wellSol)
        wellvars = list(wellvars)
        is_bhp = np.asarray(self._source_field(wellMap, "isBHP", []), dtype=bool)
        is_rate = np.asarray(self._source_field(wellMap, "isRate", []), dtype=bool)
        q_s = [var for var, flag in zip(wellvars, is_rate) if bool(flag)]
        bhp_vars = [var for var, flag in zip(wellvars, is_bhp) if bool(flag)]
        extra_wellvars = [var for var, bh, qr in zip(wellvars, is_bhp, is_rate) if not bool(bh) and not bool(qr)]

        nphase = int(getattr(reservoir_model, "n_phases", len(q_s) or len(self._phase_names())))
        if not q_s:
            q_s = [
                np.asarray([
                    self._wellsol_field(item, f"q{phase.upper()}s", 0.0)
                    for item in self._active_well_sol_items(wellSol)
                ], dtype=float)
                for phase in self._phase_names()[:nphase]
            ]
        if not bhp_vars:
            bhp = np.asarray([
                self._wellsol_field(item, "bhp", 0.0)
                for item in self._active_well_sol_items(wellSol)
            ], dtype=float)
        else:
            bhp = bhp_vars[0]

        if state is None:
            state = {}
        if p is not None:
            self._set_state_field(state, "pressure", p)

        if self.WellModels and any(hasattr(wm, "computeWellEquations") or hasattr(wm, "compute_well_equations")
                                   for wm in self.WellModels):
            return self._get_well_contributions_from_models(
                wellSol0,
                wellSol,
                extra_wellvars,
                wellMap,
                p,
                mob,
                rho,
                dissolved,
                comp,
                dt,
                iteration,
                reservoir_model,
                q_s[:nphase],
                bhp,
            )

        cq_s_all, well_eqs, ctrl_eq = self.well_equations(
            reservoir_model,
            state,
            q_s[:nphase],
            bhp,
            state0=state if wellSol0 is wellSol else None,
        )
        del cq_s_all

        active_ix = self.get_indices_of_active_wells(wellSol)
        source_cells = self.get_well_cells(active_ix)
        phase_volume: list[np.ndarray] = []
        phase_mass: list[np.ndarray] = []

        if rho is None:
            try:
                rho = reservoir_model.get_props(state, "Density")
            except Exception:
                rho = None

        for ph_no in range(nphase):
            rows: list[np.ndarray] = []
            masses: list[np.ndarray] = []
            for well_ix in active_ix:
                well_ix = int(well_ix)
                if self.last_flux is None or well_ix >= len(self.last_flux):
                    continue
                flux = np.asarray(self.last_flux[well_ix], dtype=float)
                if flux.size == 0:
                    continue
                vals = flux[:, ph_no].reshape(-1)
                rows.append(vals)
                if rho is not None and ph_no < len(rho):
                    wc = np.asarray(self.wells[well_ix].cells, dtype=np.int64).ravel()
                    masses.append(vals * np.asarray(value(rho[ph_no]), dtype=float).reshape(-1)[wc])
                else:
                    masses.append(vals.copy())
            phase_volume.append(np.concatenate(rows) if rows else np.zeros(0))
            phase_mass.append(np.concatenate(masses) if masses else np.zeros(0))

        ncomp = int(getattr(reservoir_model, "n_components", len(comp) if comp is not None else nphase))
        components: list[np.ndarray] = []
        total_mass = sum(phase_mass, np.zeros_like(phase_mass[0])) if phase_mass else np.zeros(0)
        for comp_no in range(ncomp):
            if comp is not None and comp_no < len(comp) and source_cells.size:
                comp_values = np.asarray(value(comp[comp_no]), dtype=float).reshape(-1)
                components.append(total_mass * comp_values[source_cells])
            elif comp_no < len(phase_mass):
                components.append(phase_mass[comp_no].copy())
            else:
                components.append(np.zeros_like(total_mass))

        phases = self._phase_names()[:nphase]
        names = [f"well{phase.upper()}" for phase in phases] + list(self.addedEquationNames)
        types = ["perf"] * nphase + list(self.addedEquationTypes)
        extra = [np.zeros(0) for _ in self.addedEquationNames]
        well_system = SimpleNamespace(
            wellEquations=list(well_eqs) + extra,
            well_equations=list(well_eqs) + extra,
            names=names,
            types=types,
            controlEquation=ctrl_eq,
            control_equation=ctrl_eq,
        )
        sources = SimpleNamespace(
            phaseMass=phase_mass,
            phase_mass=phase_mass,
            phaseVolume=phase_volume,
            phase_volume=phase_volume,
            components=components,
            sourceCells=source_cells,
            source_cells=source_cells,
        )
        return sources, well_system, wellSol

    getWellContributions = get_well_contributions

    @staticmethod
    def _row_value(variable: Any, row: int):
        try:
            return variable[row:row + 1]
        except Exception:
            arr = np.asarray(value(variable)).reshape(-1)
            return arr[row:row + 1]

    def _well_model_call(self, well_model: Any, names: Sequence[str], *args):
        for name in names:
            if hasattr(well_model, name):
                return getattr(well_model, name)(*args)
        return None

    def _get_well_contributions_from_models(
        self,
        well_sol0: Any,
        well_sol: Any,
        wellvars: Sequence[Any],
        well_map: Any,
        p: Any,
        mob: Sequence[Any] | None,
        rho: Sequence[Any] | None,
        dissolved: Any,
        comp: Sequence[Any] | None,
        dt: float | None,
        iteration: float | int | None,
        reservoir_model: Any,
        q_well: Sequence[Any],
        bhp: Any,
    ):
        """Assemble MRST-style contributions through attached well-model hooks."""
        active_ix = self.get_indices_of_active_wells(well_sol)
        n_active = int(active_ix.size)
        phase_names = list(getattr(reservoir_model, "getPhaseNames", self._phase_names)())
        active_phases = getattr(reservoir_model, "getActivePhases", lambda: np.ones(len(phase_names), dtype=bool))()
        nphase = int(np.count_nonzero(np.asarray(active_phases, dtype=bool)))
        ncomp = int(getattr(reservoir_model, "getNumberOfComponents", lambda: len(comp) if comp is not None else nphase)())
        if comp is not None and len(comp) != ncomp:
            raise AssertionError("Number of input components must match length of getComponentNames")

        enames = list(self.addedEquationNames)
        etypes = list(self.addedEquationTypes)
        all_base: list[Any] = []
        all_ctrl: list[Any] = []
        all_mass: list[Any] = []
        all_vol: list[Any] = []
        all_comp: list[Any] = []
        all_extra: dict[str, list[Any]] = {name.lower(): [] for name in enames}
        well_sol_items = list(well_sol)
        well_sol0_items = list(well_sol0)

        added_vars = list(self.addedPrimaryVarNames)
        varmaps = [
            self.get_well_variable_map(name, well_sol)
            for name in added_vars
        ]
        extra_map = self._source_field(well_map, "extraMap", np.zeros((n_active, len(added_vars)), dtype=np.int64))

        base_names: list[str] | None = None
        base_types: list[str] | None = None
        for row, well_ix_raw in enumerate(active_ix):
            well_ix = int(well_ix_raw)
            if well_ix >= len(self.WellModels):
                raise IndexError(f"missing well model for active well {well_ix}")
            well_model = self.WellModels[well_ix]
            ws = well_sol_items[well_ix]
            ws0 = well_sol0_items[well_ix]
            well = getattr(well_model, "W", self.wells[well_ix])
            packed = SimpleNamespace(
                W=well,
                p=p,
                mob=mob,
                rho=rho,
                dissolved=dissolved,
                comp=comp,
                wellvars=list(wellvars),
                addedVars=added_vars,
                added_vars=added_vars,
                varmaps=varmaps,
                extraMap=extra_map,
                extra_map=extra_map,
                activeOrdinal=row,
                active_ordinal=row,
            )
            qw = [self._row_value(item, row) for item in q_well[:nphase]]
            bh = self._row_value(bhp, row)

            updated = self._well_model_call(
                well_model,
                ("updateConnectionPressureDrop", "update_connection_pressure_drop"),
                ws0,
                ws,
                reservoir_model,
                qw,
                bh,
                packed,
                dt,
                iteration,
            )
            if updated is not None:
                ws = updated

            limits = self._well_model_call(
                well_model,
                ("updateLimits", "update_limits"),
                ws0,
                ws,
                reservoir_model,
                qw,
                bh,
                packed,
                dt,
                iteration,
            )
            if limits is not None:
                qw, bh, ws, _ = limits

            if base_names is None:
                names_types = self._well_model_call(
                    well_model,
                    ("getWellEquationNames", "get_well_equation_names"),
                    reservoir_model,
                )
                if names_types is None:
                    base_names = [f"well{str(name).upper()}" for name in phase_names[:nphase]]
                    base_types = ["perf"] * nphase
                else:
                    base_names = [str(item) for item in self._metadata_list(names_types[0])]
                    base_types = [str(item) for item in self._metadata_list(names_types[1])]

            result = self._well_model_call(
                well_model,
                ("computeWellEquations", "compute_well_equations"),
                ws0,
                ws,
                reservoir_model,
                qw,
                bh,
                packed,
                dt,
                iteration,
            )
            if result is None:
                raise NotImplementedError("attached well models must implement computeWellEquations")
            base_eqs, ctrl_eq, extra_eqs, extra_names, mass_src, vol_src, ws = result

            comp_result = self._well_model_call(
                well_model,
                ("computeComponentContributions", "compute_component_contributions"),
                ws0,
                ws,
                reservoir_model,
                qw,
                bh,
                packed,
                mass_src,
                vol_src,
                dt,
                iteration,
            )
            if comp_result is None:
                comp_eqs, comp_src, comp_names = [], [np.zeros_like(np.asarray(mass_src[0])) for _ in range(ncomp)], []
            else:
                comp_eqs, comp_src, comp_names, ws = comp_result

            for eq, name in zip(list(extra_eqs) + list(comp_eqs), list(extra_names) + list(comp_names)):
                key = str(name).lower()
                if key not in all_extra:
                    raise KeyError(f"well model returned undeclared extra equation {name!r}")
                all_extra[key].append(eq)

            all_base.append(base_eqs)
            all_ctrl.append(ctrl_eq)
            all_mass.append(mass_src)
            all_vol.append(vol_src)
            all_comp.append(comp_src)
            well_sol_items[well_ix] = ws

        phase_mass = [self.combine_cell_data(all_mass, ph_no) for ph_no in range(nphase)]
        phase_volume = [self.combine_cell_data(all_vol, ph_no) for ph_no in range(nphase)]
        component_sources = [self.combine_cell_data(all_comp, comp_no) for comp_no in range(ncomp)]
        well_eqs = [self.combine_cell_data(all_base, ph_no) for ph_no in range(nphase)]
        extra_eqs = [
            vertcat_adi(*all_extra[name.lower()]) if all_extra[name.lower()] else np.zeros(0)
            for name in enames
        ]
        source_cells = self.get_well_cells(active_ix)
        source_cells, phase_mass, phase_volume = self.handle_repeated_perforated_cells(
            source_cells,
            phase_mass,
            phase_volume,
        )
        well_system = SimpleNamespace(
            wellEquations=well_eqs + extra_eqs,
            well_equations=well_eqs + extra_eqs,
            names=list(base_names or []) + enames,
            types=list(base_types or []) + etypes,
            controlEquation=vertcat_adi(*all_ctrl) if all_ctrl else np.zeros(0),
            control_equation=vertcat_adi(*all_ctrl) if all_ctrl else np.zeros(0),
        )
        sources = SimpleNamespace(
            phaseMass=phase_mass,
            phase_mass=phase_mass,
            phaseVolume=phase_volume,
            phase_volume=phase_volume,
            components=component_sources,
            sourceCells=source_cells,
            source_cells=source_cells,
        )
        if getattr(reservoir_model, "extraWellSolOutput", False):
            well_sol_items = list(self.set_well_sol_statistics(well_sol_items, sources, reservoir_model))
        return sources, well_system, well_sol_items

    def well_equations(self, model: ReservoirModel, state: State,
                       q_s: List[Any], bhp: Any,
                       state0: Optional[State] = None):
        """Connection rates, rate-closure equations, and control equations.

        Returns (cq_s, well_eqs, ctrl_eqs) where cq_s[k][i] is the surface
        rate of component i through the perforations of well k.

        `state0` is used only for the connection pressure drops, which are
        held fixed over the Newton loop - see `connection_pressure_drop`.
        """
        mob = model.get_props(state, "Mobility")
        b = model.get_props(state, "ShrinkageFactors")
        p = state["pressure"]
        rs = state["rs"] if self.disgas else None
        rv = state["rv"] if self.vapoil else None
        nph = model.n_phases

        # The wellbore column is evaluated at the *previous* state and then
        # frozen for the whole Newton loop. MRST does the same, explicitly
        # ("this update only happens at the start of each nonlinear loop"):
        # making rho_mix an implicit function of the unknowns would couple
        # every perforation of a well to every other and fill in the
        # Jacobian. Re-evaluating it each iteration instead converges to a
        # slightly different (self-consistent) answer, and then the two codes
        # do not agree.
        cdp_state = state0 if state0 is not None else state
        cdp_mob = (model.get_props(cdp_state, "Mobility")
                   if state0 is not None else mob)
        cdp_rho = model.get_props(cdp_state, "Density")

        if not self.wells:
            # A model may have no wells at all - driven by its boundary, by
            # sources, or by nothing. Every well array is then empty, and
            # `vertcat` of nothing is not zero, it is an error.
            empty = np.zeros(0)
            return [], [empty] * nph, empty

        # MRST's `wellSol.flux`: the *reservoir* volumetric perforation fluxes
        # of the previous converged step, which is what the wellbore column is
        # weighed with once the well has actually flowed.
        flux_prev = cdp_state.get("well_flux")

        cq_s_all, cq_vol_all = [], []
        well_eqs, ctrls = [[] for _ in range(nph)], []
        for k, w in enumerate(self.wells):
            cdp = self.connection_pressure_drop(
                model, w, cdp_state["pressure"], cdp_mob, None, nph,
                rho=cdp_rho,
                flux=None if flux_prev is None else flux_prev[k])
            cq_s, weqs, ctrl, cq_vol = self._single_well(
                w, p, mob, b, rs, nph,
                [q[k:k + 1] for q in q_s], bhp[k:k + 1], cdp, rv=rv,
            )
            cq_s_all.append(cq_s)
            cq_vol_all.append(cq_vol)
            for i in range(nph):
                well_eqs[i].append(weqs[i])
            ctrls.append(ctrl)

        # Remembered so that a converged step can hand its perforation fluxes
        # to the next one's wellbore column (see `connection_pressure_drop`).
        self.last_flux = [
            np.column_stack([np.asarray(value(c), float).ravel()
                             for c in cq_vol[:nph]])
            for cq_vol in cq_vol_all
        ]

        return (cq_s_all,
                [vertcat_adi(*e) for e in well_eqs],
                vertcat_adi(*ctrls))

    def connection_pressure_drop(self, model: ReservoirModel, w: Well,
                                 p, mob, b, nph: int,
                                 rho=None, flux: Optional[np.ndarray] = None,
                                 ) -> np.ndarray:
        r"""The pressure drop from the bhp datum down to each perforation
        (MRST's `SimpleWell.updateConnectionPressureDropState`, which is the one
        `GenericFacilityModel` - and therefore this port's models - actually
        use; the `wb2in`-on-surface-rates variant belongs to the *classic*
        facility and is a different calculation).

        A well reports its pressure at a reference depth, but it feels the
        reservoir at its perforations, and the fluid column in between weighs
        something. `refDepth` is often *not* the perforation depth - in
        `runNorneExample.m` the wells are 32 m and 62 m below it, worth about
        3 and 6 bar - so leaving this out is not a small approximation.

        **The column is built segment by segment, not in one piece**, and that
        is what was wrong here before. A well is a chain of nodes: the datum,
        then each perforation in turn. The fluid in the segment above a
        perforation is everything flowing *past* it - the cumulative stream from
        every perforation below - so each segment has its own density and

            cdp_k = sum_{j<=k} rho_j g (dZ_j - dZ_{j-1}).

        The density of segment `j` is its **cumulative mass flux over its
        cumulative volume flux**, which is what makes it a genuine mixture
        rather than an average of averages.

        With a *single* perforation this collapses to `rho_mix g dZ` with
        `rho_mix` the mobility-weighted reservoir density - which is exactly
        what the old code computed, and exactly why nothing caught it: every
        case this port had validated against MRST had one-cell wells. A
        three-perforation producer disagreed by 1e-3.
        """
        dz_perf = np.asarray(getattr(w, "dZ", 0.0), dtype=float).ravel()
        nperf = len(w.cells)
        if not np.any(dz_perf) or model.gravity_magnitude == 0.0:
            return np.zeros(nperf)

        wc = w.cells
        wi = np.asarray(w.WI, dtype=float).ravel()
        # *Reservoir* densities at the perforations - what the wellbore fluid
        # actually weighs down there, dissolved gas included.
        rho_c = np.column_stack(
            [np.asarray(value(r), float)[wc] for r in rho[:nph]])
        mob_c = np.column_stack(
            [np.asarray(value(m), float)[wc] for m in mob[:nph]])

        if w.sign < 1 and flux is not None and np.any(flux):
            # The well has flowed: use the perforation fluxes it actually had.
            q_vol_ph = np.asarray(flux, float).reshape(nperf, nph)
        elif w.sign == -1:
            # A producer with no history: assume a uniform drawdown, so each
            # perforation delivers in proportion to `WI * mobility`.
            q_vol_ph = wi[:, None] * mob_c
        else:
            # An injector pumps what it is told to pump.
            q_vol_ph = wi[:, None] * np.asarray(w.compi[:nph], float)[None, :]

        q_mass = np.sum(q_vol_ph * rho_c, axis=1)
        q_vol = np.sum(q_vol_ph, axis=1)

        # `wb2in`: an upper-bidiagonal solve whose meaning is a reverse
        # cumulative sum - segment k carries perforations k..nperf, because
        # fluid entering at the toe passes every segment above it.
        wb_mass = np.abs(np.cumsum(q_mass[::-1])[::-1])
        wb_vol = np.abs(np.cumsum(q_vol[::-1])[::-1])

        shut = abs(q_mass.sum()) < 1e-20
        if shut:
            q_approx = wi[:, None] * mob_c
            rho_mix = (np.sum(q_approx * rho_c, axis=1)
                       / np.maximum(np.sum(q_approx, axis=1), 1e-30))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rho_mix = wb_mass / wb_vol
            rho_mix[~np.isfinite(rho_mix)] = 0.0
            # A near-zero density means a dead or disabled perforation, not a
            # weightless fluid. Back-fill from below, as MRST does.
            for i in range(nperf - 2, -1, -1):
                if rho_mix[i] < 1e-8:
                    rho_mix[i] = rho_mix[i + 1]
            bad = (rho_mix == 0.0) | ~np.isfinite(rho_mix)
            if bad.all():
                return np.zeros(nperf)
            rho_mix[bad] = rho_mix[~bad].mean()

        dz_seg = np.diff(np.concatenate([[0.0], dz_perf]))
        return np.cumsum(model.gravity_magnitude * rho_mix * dz_seg)

    def _volume_ratio(self, mix, b_c, rs_c, rv_c, nph: int):
        r"""Reservoir volume injected per unit surface volume of the mixture.

        An injector is told what to inject in *surface* volumes, and what it
        actually pushes into the rock is a *reservoir* volume. Dividing by `b`
        converts one to the other - but only if the streams are independent,
        and with dissolution they are not: a surface volume of gas is partly
        gas that will dissolve into the oil, and a surface volume of oil is
        partly condensate that will vaporise into the gas. Untangling them is a
        2x2 linear solve, and its determinant is

            d = 1 - rs * rv

        which is the whole of the black-oil dissolution matrix. It is why
        `rs*rv < 1` is a physical requirement and not a modelling convenience -
        at `rs*rv = 1` oil and gas have become the same fluid and the surface
        split is meaningless.

        With no vaporised oil `rv = 0`, `d = 1`, and this collapses to the
        familiar `(mix_g - rs*mix_o)/b_g`.
        """
        if not (self.disgas or self.vapoil):
            total = mix[0] / b_c[0]
            for i in range(1, nph):
                total = total + mix[i] / b_c[i]
            return total

        tmp = list(mix)
        rs = rs_c if self.disgas else 0.0
        rv = rv_c if self.vapoil else 0.0
        d = 1.0 - rs * rv
        # Both use the *original* mixture, not each other's updated value -
        # this is one linear solve, not two sequential corrections.
        if self.disgas:
            tmp[2] = (mix[2] - rs * mix[1]) / d
        if self.vapoil:
            tmp[1] = (mix[1] - rv * mix[2]) / d

        total = tmp[0] / b_c[0]
        for i in range(1, nph):
            total = total + tmp[i] / b_c[i]
        return total

    def _single_well(self, w: Well, p, mob, b, rs, nph: int, q_s, bhp_k,
                     cdp: np.ndarray, rv=None):
        """MRST's computeWellContributionsSingleWell."""
        wc, tw = w.cells, w.WI
        compi = w.compi[:nph]

        drawdown = -(bhp_k + cdp) + p[wc]
        inj = value(drawdown) < 0
        prod = ~inj
        mob_c = [m[wc] for m in mob]
        b_c = [x[wc] for x in b]
        rs_c = rs[wc] if self.disgas else None
        rv_c = rv[wc] if self.vapoil else None

        any_inj, any_prod = bool(inj.any()), bool(prod.any())

        if any_prod:
            con = prod * tw
            cq_p = [-con * m * drawdown for m in mob_c]
            cq_ps = [b_c[i] * cq_p[i] for i in range(nph)]
            # What a phase carries *of another component*: the oil stream
            # brings dissolved gas up with it, and a wet gas stream brings
            # condensate. Both are surface volumes of a component that is not
            # the phase's own.
            if self.disgas:
                cq_ps[2] = cq_ps[2] + rs_c * (b_c[1] * cq_p[1])
            if self.vapoil:
                cq_ps[1] = cq_ps[1] + rv_c * (b_c[2] * cq_p[2])
            wbq = [q_s[i] * float(value(q_s[i])[0] > 0) - ad_sum(cq_ps[i])
                   for i in range(nph)]
            wbqt = wbq[0]
            for i in range(1, nph):
                wbqt = wbqt + wbq[i]
            if value(wbqt)[0] == 0.0:            # dead well: fall back to compi
                mix = list(compi)
            else:
                mix = [q / wbqt for q in wbq]
        else:
            cq_ps = None
            mix = list(compi)

        if any_inj:
            mt = mob_c[0]
            for i in range(1, nph):
                mt = mt + mob_c[i]
            cqt_i = -(inj * tw) * (mt * drawdown)
            vol_rat = self._volume_ratio(mix, b_c, rs_c, rv_c, nph)
            cqt_is = cqt_i / vol_rat
            if any_prod:
                cq_s = [cq_ps[i] + mix[i] * cqt_is for i in range(nph)]
            else:
                cq_s = [mix[i] * cqt_is for i in range(nph)]
        else:
            cq_s = cq_ps

        # The *reservoir* volumetric flux through each perforation, per phase
        # (MRST's `cq_vol`): an injecting connection pushes its mixture, a
        # producing one delivers each phase in proportion to its mobility. It
        # is what the next step's wellbore column is weighed with, and it is
        # not the same thing as `cq_s`, which is in surface volumes.
        cq_vol = []
        for i in range(nph):
            term = None
            if any_inj:
                term = inj * (cqt_i * compi[i])
            if any_prod:
                pterm = prod * cq_p[i]
                term = pterm if term is None else term + pterm
            cq_vol.append(term)

        well_eqs = [q_s[i] - ad_sum(cq_s[i]) for i in range(nph)]
        ctrl = self._control_equation(w, q_s, bhp_k, nph, cq_vol=cq_vol)
        return cq_s, well_eqs, ctrl, cq_vol

    @staticmethod
    def _control_density(w: Well, nph: int) -> np.ndarray:
        """MRST's `ControlDensity` for reservoir-rate controls.

        `resv` controls close on surface phase rates converted to reservoir
        volume at the previous-step control density. PyMRST wells predate that
        field, so accept both Pythonic `control_density` and MRST-style
        `ControlDensity`; missing values reduce to a unit conversion.
        """
        rho = getattr(w, "control_density",
                      getattr(w, "ControlDensity", None))
        if rho is None:
            return np.ones(nph)
        rho = np.asarray(rho, dtype=float).ravel()
        if rho.size == 1:
            return np.repeat(rho[0], nph)
        if rho.size < nph:
            raise ValueError("resv control needs one control density per phase")
        return rho[:nph]

    @staticmethod
    def _control_equation(w: Well, q_s, bhp_k, nph: int,
                          cq_vol: Optional[Sequence[Any]] = None):
        """The one equation that closes a well: it holds its target.

        Every mode a well can *switch into* needs one of these, or a well that
        hits its liquid-rate limit has nowhere to go.
        """
        def total():
            t = q_s[0]
            for i in range(1, nph):
                t = t + q_s[i]
            return t

        qt = total()
        if w.type == "bhp":
            qt_val = np.asarray(value(qt), dtype=float).ravel()
            if w.sign != 0 and np.any(qt_val != 0.0) and np.any(np.sign(qt_val) != w.sign):
                return qt
            return bhp_k - w.val
        if w.type in ("rate", "vrat"):
            return qt - w.val
        if w.type == "resv":
            rho = FacilityModel._control_density(w, nph)
            resv = q_s[0] * rho[0]
            for i in range(1, nph):
                resv = resv + q_s[i] * rho[i]
            return resv - w.val
        if w.type == "volume":
            if cq_vol is None:
                raise ValueError("volume control requires reservoir fluxes")
            vol = ad_sum(cq_vol[0])
            for i in range(1, nph):
                vol = vol + ad_sum(cq_vol[i])
            return vol - w.val
        if w.type == "wrat":
            return q_s[0] - w.val
        if w.type == "orat":
            return q_s[1] - w.val
        if w.type == "grat":
            assert nph > 2, "a gas rate control needs a gas phase"
            return q_s[2] - w.val
        if w.type == "lrat":
            return q_s[0] + q_s[1] - w.val
        if w.type in ("thp", "vfp"):
            return bhp_k - FacilityModel._vfp_bhp(w, q_s, nph)
        raise ValueError(f"unsupported well control '{w.type}'")

    @staticmethod
    def _vfp_bhp(w: Well, q_s, nph: int):
        """Explicit VFP BHP target from the current well surface rates."""
        if w.vfp_table is None:
            raise ValueError("THP/VFP control requires w.vfp_table")
        thp = float(w.thp if w.thp is not None else w.val)
        qv = [float(np.asarray(value(q), dtype=float).ravel()[0]) for q in q_s]
        water = abs(qv[0]) if nph > 0 else 0.0
        oil = abs(qv[1]) if nph > 1 else 0.0
        gas = abs(qv[2]) if nph > 2 else 0.0
        liquid = water + oil
        total = liquid + gas
        flow_type = getattr(w.vfp_table, "flow_type", "").lower()
        if flow_type in ("liquid", "lrat"):
            flow = liquid
        elif flow_type in ("oil", "orat"):
            flow = oil
        elif flow_type in ("gas", "grat"):
            flow = gas
        else:
            flow = total
        params = []
        for name in getattr(w.vfp_table, "parameter_names", []):
            key = name.lower()
            if key == "wfr":
                params.append(water / (liquid if liquid > 0.0 else 1.0))
            elif key == "gfr":
                params.append(gas / (oil if oil > 0.0 else max(liquid, 1.0)))
            elif key == "alq":
                params.append(float(getattr(w, "alq", 0.0)))
            else:
                raise ValueError(f"unsupported VFP parameter '{name}'")
        return float(np.asarray(w.vfp_table.evaluate_bhp(flow, thp, *params)).ravel()[0])

    def apply_to_reservoir(self, eqs: List[Any], cq_s_all: List[List[Any]],
                           scale: Optional[np.ndarray] = None):
        """Subtract each well's connection rates from its perforated cells.

        A well's rates are *surface volumes* - that is what a well is metered
        in - so in the mass basis they must be weighed before they can be put
        into a mass equation. `scale` is the model's `_basis_scale()`.
        """
        for k, w in enumerate(self.wells):
            for i, eq in enumerate(eqs):
                q = cq_s_all[k][i]
                if scale is not None:
                    q = scale[i] * q
                eq[w.cells] = eq[w.cells] - q
        return eqs

    @staticmethod
    def handle_repeated_perforated_cells(wc: Sequence[int], *sources):
        """Collapse repeated perforation cells and sum matching source rows.

        Direct Python port of MRST ``FacilityModel.handleRepeatedPerforatedcells``.
        Each item in ``sources`` is expected to be a sequence of per-phase
        arrays with one row per entry of ``wc``.  The return value is
        ``(unique_wc, *collapsed_sources)`` with cells in first-occurrence
        order, matching MRST's stable unique behavior.
        """
        wc_arr = np.asarray(wc, dtype=np.int64).ravel()
        if wc_arr.size == 0:
            return (wc_arr, *sources)

        unique: list[int] = []
        inverse = np.empty(wc_arr.size, dtype=np.int64)
        by_cell: dict[int, int] = {}
        for row, cell in enumerate(wc_arr.tolist()):
            if cell not in by_cell:
                by_cell[cell] = len(unique)
                unique.append(cell)
            inverse[row] = by_cell[cell]

        if len(unique) == wc_arr.size:
            return (wc_arr, *sources)

        collapsed_sources = []
        for source in sources:
            collapsed_phases = []
            for phase_values in source:
                arr = np.asarray(phase_values)
                if arr.shape[0] != wc_arr.size:
                    raise ValueError("source arrays must have one row per perforated cell")
                out = np.zeros((len(unique),) + arr.shape[1:], dtype=arr.dtype)
                np.add.at(out, inverse, arr)
                collapsed_phases.append(out)
            collapsed_sources.append(collapsed_phases)
        return (np.asarray(unique, dtype=np.int64), *collapsed_sources)

    handleRepeatedPerforatedcells = handle_repeated_perforated_cells
    handleRepeatedPerforatedCells = handle_repeated_perforated_cells

    @staticmethod
    def combine_cell_data(data: Sequence[Sequence[Any]], ix: int) -> np.ndarray:
        """Vertically concatenate item ``ix`` from each cell-like sequence."""
        parts = [np.asarray(item[ix]) for item in data]
        return np.concatenate(parts, axis=0) if parts else np.zeros(0)

    combineCellData = combine_cell_data


__all__ = ["FacilityLinearizedProblem", "FacilityModel"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
    from pymrst.modules.ad_core.statefunctions.pvt_property_functions import simple_phase_fluid
    from pymrst.modules.ad_props.simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock, pore_volume
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.adi import init_variables_adi
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model():
        g = compute_geometry(cart_grid([4, 4, 1], [100.0, 100.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        f = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        ops = setup_operators_tpfa(g, rock)
        m = ReservoirModel(g, rock, simple_phase_fluid(f), ops,
                           pore_volume(g, rock), np.zeros(ops.T.size))
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([15]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        return m, FacilityModel(w)

    def test_validate_model_accepts_a_sound_graph():
        """The shipped groupings resolve every dependency they declare."""
        m, _ = _model()
        m.validate_model()
        print("ok test_validate_model_accepts_a_sound_graph")

    def test_validate_model_catches_a_bad_dependency():
        """A typo'd dependency is caught at setup, not deep in a solve."""
        m, _ = _model()
        m.FlowPropertyFunctions.get_state_function("Mobility").depends_on(
            "Viscocity", "PVTPropertyFunctions")      # sic
        try:
            m.validate_model()
            raise SystemExit("should have raised")
        except KeyError as exc:
            assert "Viscocity" in str(exc)
        print("ok test_validate_model_catches_a_bad_dependency")

    def test_get_props_falls_back_to_state_variables():
        """get_props resolves state variables as well as state functions."""
        m, _ = _model()
        st = State(pressure=np.full(16, 200.0 * BARSA), sW=np.full(16, 0.3))
        assert np.allclose(m.get_props(st, "pressure"), 200.0 * BARSA)
        assert len(m.get_props(st, "Mobility")) == 2
        try:
            m.get_props(st, "Nonsense")
            raise SystemExit("should have raised")
        except KeyError:
            pass
        print("ok test_get_props_falls_back_to_state_variables")

    def test_wells_inject_and_produce():
        """The injector adds mass to its cell, the producer removes it."""
        m, fac = _model()
        p, sw, qws, qos, bhp = init_variables_adi(
            np.full(16, 200.0 * BARSA), np.full(16, 0.3),
            np.array([1e-3, 0.0]), np.array([0.0, -1e-3]),
            np.array([250.0 * BARSA, 150.0 * BARSA]),   # over/under-pressured
        )
        st = State(pressure=p, sW=sw)
        cq_s, well_eqs, ctrl = fac.well_equations(m, st, [qws, qos], bhp)
        assert value(cq_s[0][0])[0] > 0, "injector must add water"
        assert value(cq_s[1][1])[0] < 0, "producer must remove oil"
        assert len(well_eqs) == 2 and value(ctrl).size == 2
        print("ok test_wells_inject_and_produce")

    def test_facility_subtracts_rates_from_the_right_cells():
        """apply_to_reservoir touches only the perforated cells."""
        m, fac = _model()
        eqs = [np.zeros(16), np.zeros(16)]
        cq_s = [[np.array([2.0]), np.array([0.0])],
                [np.array([0.0]), np.array([-3.0])]]
        eqs = fac.apply_to_reservoir(eqs, cq_s)
        assert eqs[0][0] == -2.0 and eqs[1][15] == 3.0
        assert np.count_nonzero(eqs[0]) == 1 and np.count_nonzero(eqs[1]) == 1
        print("ok test_facility_subtracts_rates_from_the_right_cells")

    def test_vfp_control_equation():
        """THP/VFP control closes on bhp minus tabulated bottom-hole pressure."""
        from pymrst.modules.ad_core.models.facilities.vfp import VFPTable

        table = VFPTable(
            np.array([0.0, 10.0]), np.array([100.0, 200.0]),
            [], [], np.array([[100.0, 200.0], [120.0, 220.0]]),
            flow_type="liquid",
        )
        _, fac = _model()
        w = fac.wells[0]
        w.type = "thp"
        w.thp = 150.0
        w.vfp_table = table
        qws = np.array([5.0])
        qos = np.array([0.0])
        bhp = np.array([160.0])
        ctrl = FacilityModel._control_equation(w, [qws, qos], bhp, 2)
        assert np.isclose(ctrl[0], 0.0), ctrl
        print("ok test_vfp_control_equation")

    def test_bhp_wrong_direction_falls_back_to_total_rate():
        """MRST GenericFacilityModel replaces bad-direction BHP equations."""
        _, fac = _model()
        w = fac.wells[1]
        w.type = "bhp"
        w.val = 150.0
        w.sign = -1
        ctrl = FacilityModel._control_equation(w, [np.array([2.0]), np.array([0.0])], np.array([150.0]), 2)
        assert np.isclose(ctrl[0], 2.0), ctrl
        ctrl = FacilityModel._control_equation(w, [np.array([0.0]), np.array([-2.0])], np.array([160.0]), 2)
        assert np.isclose(ctrl[0], 10.0), ctrl
        print("ok test_bhp_wrong_direction_falls_back_to_total_rate")

    def test_resv_control_equation_uses_control_density():
        """MRST `resv` control closes on density-weighted surface rates."""
        _, fac = _model()
        w = fac.wells[0]
        w.type = "resv"
        w.val = 7.0
        w.ControlDensity = np.array([2.0, 3.0])
        qws = np.array([1.0])
        qos = np.array([2.0])
        ctrl = FacilityModel._control_equation(w, [qws, qos],
                                               np.array([0.0]), 2)
        assert np.isclose(ctrl[0], 1.0), ctrl

        w.control_density = np.array([5.0])
        ctrl = FacilityModel._control_equation(w, [qws, qos],
                                               np.array([0.0]), 2)
        assert np.isclose(ctrl[0], 8.0), ctrl
        print("ok test_resv_control_equation_uses_control_density")

    def test_volume_control_equation_uses_reservoir_fluxes():
        """MRST `volume` control closes on current reservoir perforation flux."""
        _, fac = _model()
        w = fac.wells[0]
        w.type = "volume"
        w.val = 4.0
        cq_vol = [np.array([1.0, 0.5]), np.array([2.0, 0.25])]
        ctrl = FacilityModel._control_equation(
            w, [np.array([0.0]), np.array([0.0])], np.array([0.0]), 2,
            cq_vol=cq_vol,
        )
        assert np.isclose(ctrl, -0.25), ctrl

        try:
            FacilityModel._control_equation(
                w, [np.array([0.0]), np.array([0.0])], np.array([0.0]), 2)
            raise SystemExit("should have raised")
        except ValueError as exc:
            assert "reservoir fluxes" in str(exc)
        print("ok test_volume_control_equation_uses_reservoir_fluxes")

    def test_connection_pressure_drop():
        """cdp = rho_mix g dZ, and it is zero exactly when it should be.

        A well's bhp datum is typically tens of metres from its perforations,
        and the fluid column in between is worth several bar - so this is not a
        rounding term.

        The mixture is what the well *carries*: an injector's is its `compi`, a
        producer's is weighted by mobility (or, once it has flowed, by the
        fluxes it actually had). And the density is the **reservoir** one - what
        the fluid weighs down there, dissolved gas included - not a surface
        density.
        """
        m, fac = _model()
        st = State(pressure=np.full(16, 200.0 * BARSA), sW=np.full(16, 0.3))
        mob = m.get_props(st, "Mobility")
        b = m.get_props(st, "ShrinkageFactors")
        rho = m.get_props(st, "Density")
        inj, prod = fac.wells

        # no gravity -> no column, whatever the geometry
        assert np.all(fac.connection_pressure_drop(
            m, inj, st["pressure"], mob, b, 2, rho=rho) == 0.0)

        m.gravity_magnitude = 9.80665
        inj.dZ = np.array([50.0])            # 50 m below the datum

        # a water injector's wellbore holds water (compi = [1, 0])
        cdp = fac.connection_pressure_drop(m, inj, st["pressure"], mob, b, 2,
                                           rho=rho)
        rho_w = float(value(rho[0])[inj.cells[0]])
        want = rho_w * 9.80665 * 50.0
        assert np.isclose(cdp[0], want, rtol=1e-12), (cdp[0], want)
        assert cdp[0] > 4e5, "50 m of water is worth ~5 bar"

        # a producer's column is what its perforations deliver: with no history,
        # the mobility-weighted reservoir density - and at sW = 0.3 that is
        # mostly oil, so it is lighter than the injector's water.
        prod.dZ = np.array([50.0])
        cdp_p = fac.connection_pressure_drop(m, prod, st["pressure"], mob, b, 2,
                                             rho=rho)
        mob_c = np.array([float(value(x)[prod.cells[0]]) for x in mob[:2]])
        rho_c = np.array([float(value(x)[prod.cells[0]]) for x in rho[:2]])
        want_p = (mob_c @ rho_c) / mob_c.sum() * 9.80665 * 50.0
        assert np.isclose(cdp_p[0], want_p, rtol=1e-12), (cdp_p[0], want_p)
        assert cdp_p[0] < cdp[0], "an oil column is lighter than a water one"

        # a well at its own datum feels nothing
        inj.dZ = np.array([0.0])
        assert np.all(fac.connection_pressure_drop(
            m, inj, st["pressure"], mob, b, 2, rho=rho) == 0.0)
        print("ok test_connection_pressure_drop")

    def test_the_column_is_built_segment_by_segment():
        """THE test for a multi-perforation well, and the one nothing in this
        port had ever done.

        A well with three perforations is a chain of three segments, and each
        carries a *different* stream: the deepest carries only its own inflow,
        the shallowest carries all three. So each segment has its own density,
        and the column is their running sum - not one density times the total
        depth, which is what this used to compute and is only right when there
        is a single perforation.

        Make the phases as different as possible (water below, oil above) and
        the two answers cannot agree.
        """
        m, fac = _model()
        m.gravity_magnitude = 9.80665
        st = State(pressure=np.full(16, 200.0 * BARSA), sW=np.full(16, 0.3))
        b = m.get_props(st, "ShrinkageFactors")
        mob = m.get_props(st, "Mobility")
        rho = m.get_props(st, "Density")

        w = fac.wells[1]                       # the producer
        w.cells = np.array([0, 1, 2])
        w.WI = np.full(3, float(np.atleast_1d(w.WI)[0]))
        w.dZ = np.array([10.0, 20.0, 30.0])    # 10 m apart, down the hole

        # the toe produces oil, the heel water: every segment sees a different
        # mixture, because each carries everything below it
        flux = np.array([[0.0, -1e-3],         # heel: oil
                         [0.0, -1e-3],
                         [-1e-3, 0.0]])        # toe: water
        cdp = fac.connection_pressure_drop(m, w, st["pressure"], mob, b, 2,
                                           rho=rho, flux=flux)

        # segment 1 carries all three perforations (2 oil + 1 water),
        # segment 2 carries two (1 oil + 1 water), segment 3 only the water
        assert cdp[0] < cdp[1] < cdp[2], "the column must accumulate downward"
        # and the increments must *differ*, because the segments differ - a
        # single-density column would give three equal 10 m increments
        d = np.diff(np.concatenate([[0.0], cdp]))
        assert not np.allclose(d, d[0], rtol=1e-6), (
            "the segments all had the same density - the column is not being "
            "built segment by segment")
        # the deepest segment is pure water, so its increment is the heaviest
        assert d[2] > d[1] > d[0], d
        print(f"ok test_the_column_is_built_segment_by_segment "
              f"(segment increments {d[0]/1e5:.3f}, {d[1]/1e5:.3f}, "
              f"{d[2]/1e5:.3f} bar - a single-density column would give three "
              f"equal ones)")

    def test_cdp_shifts_the_drawdown():
        """A perforation below the datum sees a higher effective well
        pressure, so an injector at fixed bhp injects more."""
        m, fac = _model()
        m.gravity_magnitude = 9.80665
        p, sw, qws, qos, bhp = init_variables_adi(
            np.full(16, 200.0 * BARSA), np.full(16, 0.3),
            np.array([1e-3, 0.0]), np.array([0.0, -1e-3]),
            np.array([250.0 * BARSA, 150.0 * BARSA]),
        )
        st = State(pressure=p, sW=sw)

        rates = []
        for dz in (0.0, 100.0):
            fac.wells[0].dZ = np.array([dz])
            cq_s, _, _ = fac.well_equations(m, st, [qws, qos], bhp)
            rates.append(float(value(cq_s[0][0])[0]))
        assert rates[1] > rates[0], (
            "a deeper perforation has a heavier column above it and so a "
            f"larger drawdown: {rates}"
        )
        fac.wells[0].dZ = np.array([0.0])
        print("ok test_cdp_shifts_the_drawdown")

    test_validate_model_accepts_a_sound_graph()
    test_validate_model_catches_a_bad_dependency()
    test_get_props_falls_back_to_state_variables()
    test_wells_inject_and_produce()
    test_facility_subtracts_rates_from_the_right_cells()
    test_vfp_control_equation()
    test_bhp_wrong_direction_falls_back_to_total_rate()
    test_resv_control_equation_uses_control_density()
    test_volume_control_equation_uses_reservoir_fluxes()
    test_connection_pressure_drop()
    test_the_column_is_built_segment_by_segment()
    test_cdp_shifts_the_drawdown()
    print("\nAll tests passed!")
