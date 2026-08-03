"""Shared state contract for ad-mechanics fluid helper models."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class CoupledFluidHelper:
    primary_variable_names: tuple[str, ...] = ()
    variable_names: tuple[str, ...] = ()

    def get_equations(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            f"{type(self).__name__} is a coupled helper and cannot assemble "
            "a standalone timestep problem"
        )

    getEquations = get_equations

    def _well_variable_names(self, state: Any | None) -> list[str]:
        facility = getattr(self, "FacilityModel", None) or getattr(self, "facility", None)
        if facility is None or state is None or "wellSol" not in state:
            return []
        method = getattr(facility, "get_all_primary_variables", None) or getattr(
            facility, "getAllPrimaryVariables", None
        )
        if method is None:
            return []
        result = method(state["wellSol"])
        if isinstance(result, tuple) and len(result) >= 2:
            return list(result[1])
        return list(result)

    def get_all_primary_variables(self, state: Any | None = None) -> list[str]:
        return [*self.primary_variable_names, *self._well_variable_names(state)]

    getAllPrimaryVariables = get_all_primary_variables

    def get_all_vars_names(self) -> list[str]:
        return list(self.variable_names)

    getAllVarsNames = get_all_vars_names

    def get_variable_field(self, name: str):
        aliases = {
            "water": ("s", 0), "oil": ("s", 1), "gas": ("s", 2),
            "sW": ("sW", None), "sO": ("sO", None), "sG": ("sG", None),
        }
        if name not in self.variable_names:
            raise KeyError(f"Unknown fluid variable {name!r}")
        return aliases.get(name, (name, None))

    getVariableField = get_variable_field

    @staticmethod
    def _problem_names(problem: Any) -> list[str]:
        names = getattr(problem, "primary_variables", None)
        if names is None:
            names = getattr(problem, "primaryVariables", None)
        if names is None:
            raise ValueError("coupled problem does not declare primary variables")
        return list(names)

    def update_state(
        self,
        state: Any,
        problem: Any,
        increments: Any,
        driving_forces: Any = None,
    ):
        names = self._problem_names(problem)
        owned = self.get_all_primary_variables(state)
        missing = [name for name in owned if name not in names]
        if missing:
            raise ValueError(f"coupled problem is missing fluid variables: {missing}")
        if not isinstance(increments, (list, tuple)):
            raise TypeError("fluid helper updates require one increment block per primary variable")
        if len(increments) != len(names):
            raise ValueError("increment block count does not match primary variables")
        out = state.copy()
        for name in owned:
            dx = np.asarray(increments[names.index(name)], dtype=float)
            if name in out:
                out[name] = np.asarray(out[name], dtype=float) + dx
            else:
                out[name] = dx.copy()
        if "sW" in out and "sO" in self.variable_names and "sG" not in out:
            out["sO"] = 1.0 - np.asarray(out["sW"])
        clear = getattr(out, "clear_cache", None)
        if clear is not None:
            clear()
        return out, {}

    updateState = update_state


if __name__ == "__main__":
    from types import SimpleNamespace

    class _Helper(CoupledFluidHelper):
        primary_variable_names = ("pressure", "sW")
        variable_names = ("pressure", "sW", "sO", "water")

    def _problem(*names):
        return SimpleNamespace(primary_variables=list(names))

    def test_a_coupled_helper_refuses_to_assemble_alone():
        """The point of the class: it is half of a coupled system.

        Returning an empty equation list instead would let a caller assemble a
        timestep that silently omits the mechanics.
        """
        try:
            _Helper().get_equations(None, None, None)
        except RuntimeError as error:
            assert "cannot assemble" in str(error), error
        else:
            raise AssertionError("expected the helper to refuse")
        print("ok test_a_coupled_helper_refuses_to_assemble_alone")

    def test_increments_are_matched_by_name_not_by_position():
        """The whole reason `update_state` takes the problem's variable list.

        The coupled problem orders its unknowns to suit the mechanics, so the
        fluid block is not at index 0. Applying increments positionally would
        add the displacement update to the pressure and still produce a state
        of the right shape.
        """
        helper = _Helper()
        state = {"pressure": np.array([10.0, 20.0]), "sW": np.array([0.3, 0.4])}
        # The fluid variables sit *after* the mechanics block here.
        problem = _problem("displacement", "pressure", "sW")
        increments = [np.array([99.0, 99.0]),      # mechanics, must be ignored
                      np.array([1.0, 2.0]),        # pressure
                      np.array([0.01, 0.02])]      # sW
        out, _ = helper.update_state(state, problem, increments)
        assert np.allclose(out["pressure"], [11.0, 22.0]), out["pressure"]
        assert np.allclose(out["sW"], [0.31, 0.42]), out["sW"]
        print("ok test_increments_are_matched_by_name_not_by_position")

    def test_a_problem_missing_a_fluid_variable_is_rejected():
        """Silently skipping it would freeze that unknown at its old value."""
        helper = _Helper()
        state = {"pressure": np.array([10.0]), "sW": np.array([0.3])}
        try:
            helper.update_state(state, _problem("displacement", "pressure"),
                                [np.array([0.0]), np.array([1.0])])
        except ValueError as error:
            assert "missing fluid variables" in str(error), error
            assert "sW" in str(error), error
        else:
            raise AssertionError("expected the missing variable to be reported")
        print("ok test_a_problem_missing_a_fluid_variable_is_rejected")

    def test_a_mismatched_increment_count_is_rejected():
        helper = _Helper()
        state = {"pressure": np.array([1.0]), "sW": np.array([0.2])}
        try:
            helper.update_state(state, _problem("pressure", "sW"), [np.array([1.0])])
        except ValueError as error:
            assert "increment block count" in str(error), error
        else:
            raise AssertionError("expected a short increment list to be rejected")
        print("ok test_a_mismatched_increment_count_is_rejected")

    def test_oil_saturation_is_closed_from_water():
        """Two-phase saturations must sum to one after every update."""
        helper = _Helper()
        state = {"pressure": np.array([1.0]), "sW": np.array([0.3])}
        out, _ = helper.update_state(state, _problem("pressure", "sW"),
                                     [np.array([0.0]), np.array([0.1])])
        assert np.allclose(out["sW"], 0.4)
        assert np.allclose(out["sO"], 0.6), out["sO"]
        assert np.allclose(out["sW"] + out["sO"], 1.0)
        print("ok test_oil_saturation_is_closed_from_water")

    def test_the_source_state_is_not_mutated():
        helper = _Helper()
        state = {"pressure": np.array([1.0]), "sW": np.array([0.3])}
        helper.update_state(state, _problem("pressure", "sW"),
                            [np.array([5.0]), np.array([0.1])])
        assert np.allclose(state["pressure"], 1.0), "the input state was modified"
        print("ok test_the_source_state_is_not_mutated")

    def test_variable_fields_resolve_phase_aliases():
        helper = _Helper()
        assert helper.get_variable_field("water") == ("s", 0)
        assert helper.get_variable_field("sW") == ("sW", None)
        try:
            helper.get_variable_field("temperature")
        except KeyError:
            pass
        else:
            raise AssertionError("expected an unknown variable to be rejected")
        print("ok test_variable_fields_resolve_phase_aliases")

    def test_well_variables_join_the_primary_list():
        """Well unknowns belong to the fluid side of the coupled system."""
        helper = _Helper()
        helper.FacilityModel = SimpleNamespace(
            get_all_primary_variables=lambda well_sol: ([], ["bhp", "qWs"]))
        names = helper.get_all_primary_variables({"wellSol": [{}]})
        assert names == ["pressure", "sW", "bhp", "qWs"], names
        assert helper.get_all_primary_variables(None) == ["pressure", "sW"]
        print("ok test_well_variables_join_the_primary_list")

    test_a_coupled_helper_refuses_to_assemble_alone()
    test_increments_are_matched_by_name_not_by_position()
    test_a_problem_missing_a_fluid_variable_is_rejected()
    test_a_mismatched_increment_count_is_rejected()
    test_oil_saturation_is_closed_from_water()
    test_the_source_state_is_not_mutated()
    test_variable_fields_resolve_phase_aliases()
    test_well_variables_join_the_primary_list()
    print("\nAll tests passed!")
