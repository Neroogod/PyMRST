"""Base contract for sequential mechanics/fluid splitting models."""

from __future__ import annotations

from typing import Any, Iterable

from pymrst.modules.ad_mechanics.models.poromechanics import FixedStressPoreVolume


def _method(owner: Any, *names: str):
    for name in names:
        value = getattr(owner, name, None)
        if value is not None:
            return value
    return None


def _vars(owner: Any) -> list[str]:
    method = _method(owner, "get_all_vars_names", "getAllVarsNames")
    return list(method()) if method is not None else []


class MechFluidSplitModel:
    """MRST ``MechFluidSplitModel`` state and delegation infrastructure."""

    mechanics_fields = ("xd", "uu", "u", "stress", "strain", "vdiv")

    def __init__(
        self,
        grid: Any,
        rock: Any,
        fluid: Any,
        mech_model: Any,
        fluid_model: Any,
        *,
        fluid_model_type: str = "water",
        splitting_tolerance: float = 1.0e-6,
        splitting_verbose: bool = False,
        mech_solver: Any = None,
        fluid_solver: Any = None,
    ):
        self.G, self.rock, self.fluid = grid, rock, fluid
        self.mechModel, self.fluidModel = mech_model, fluid_model
        self.mechfds, self.fluidfds = _vars(mech_model), _vars(fluid_model)
        tokens = set(fluid_model_type.lower().replace("-", " ").split())
        blackoil = "blackoil" in tokens or "black" in tokens and "oil" in tokens
        self.water = "water" in tokens or blackoil
        self.oil = "oil" in tokens or blackoil
        self.gas = "gas" in tokens or blackoil
        self.splittingTolerance = float(splitting_tolerance)
        self.splittingVerbose = bool(splitting_verbose)
        self.mech_solver, self.fluid_solver = mech_solver, fluid_solver

    def setup_fluid_model(self, *args: Any, **kwargs: Any):
        raise RuntimeError("MechFluidSplitModel is abstract: concrete class must set up fluid model")

    setupFluidModel = setup_fluid_model

    def step_function(self, *args: Any, **kwargs: Any):
        raise RuntimeError("MechFluidSplitModel is abstract: concrete class must implement split step")

    stepFunction = step_function

    def compute_mech_term(self, *args: Any, **kwargs: Any):
        raise RuntimeError("MechFluidSplitModel is abstract: concrete class must compute mechanics term")

    computeMechTerm = compute_mech_term

    def get_variable_field(self, name: str, *args: Any, **kwargs: Any):
        if name in self.mechanics_fields:
            aliases = {"uu": "u", "vdiv": "volumetric_strain"}
            return aliases.get(name, name), None if name == "xd" else slice(None)
        method = _method(self.fluidModel, "get_variable_field", "getVariableField")
        if method is None:
            raise KeyError(f"Unknown split-model variable {name!r}")
        return method(name, *args, **kwargs)

    getVariableField = get_variable_field

    @staticmethod
    def _copy_fields(target: Any, source: Any, fields: Iterable[str]):
        out = {} if target is None else target.copy()
        for field in fields:
            try:
                out[field] = source[field]
            except (KeyError, TypeError):
                continue
        return out

    def sync_w_state_from_state(self, state: Any):
        return self._copy_fields(None, state, self.fluidfds)

    syncWStateFromState = sync_w_state_from_state

    def sync_m_state_from_state(self, state: Any):
        return self._copy_fields(None, state, self.mechfds)

    syncMStateFromState = sync_m_state_from_state

    def sync_state_from_w_state(self, state: Any, wstate: Any):
        return self._copy_fields(state, wstate, self.fluidfds)

    syncStateFromWState = sync_state_from_w_state

    def sync_state_from_m_state(self, state: Any, mstate: Any):
        return self._copy_fields(state, mstate, self.mechfds)

    syncStateFromMState = sync_state_from_m_state

    def validate_model(self, *args: Any, **kwargs: Any):
        method = _method(self.fluidModel, "validate_model", "validateModel")
        if method is not None:
            result = method(*args, **kwargs)
            if result is not None:
                self.fluidModel = result
        return self

    validateModel = validate_model

    def check_convergence(self, problem: Any, *args: Any, **kwargs: Any):
        if hasattr(self.mechModel, "nonlinearTolerance"):
            self.mechModel.nonlinearTolerance = self.splittingTolerance
        method = _method(self.mechModel, "check_convergence", "checkConvergence")
        if method is None:
            raise AttributeError("mechanics model does not expose a convergence check")
        return method(problem, *args, **kwargs)

    checkConvergence = check_convergence

    def validate_state(self, state: Any):
        for owner in (self.fluidModel, self.mechModel):
            method = _method(owner, "validate_state", "validateState")
            if method is not None:
                state = method(state)
        return state

    validateState = validate_state

    def update_after_convergence(self, state0: Any, state: Any, dt: float, driving_forces: Any = None):
        # Concrete solvers may add a report; this base guarantees synchronized
        # mechanics diagnostics are present in the accepted state.
        if "u" in state and "pressure" in state and hasattr(self.mechModel, "diagnostics"):
            diagnostics = self.mechModel.diagnostics(state)
            state = state.copy()
            state["strain"] = diagnostics["strain"]
            state["stress"] = diagnostics["effective_stress"]
            state["vdiv"] = diagnostics["volumetric_strain"]
        return state, {}

    updateAfterConvergence = update_after_convergence


__all__ = ["FixedStressPoreVolume", "MechFluidSplitModel"]
