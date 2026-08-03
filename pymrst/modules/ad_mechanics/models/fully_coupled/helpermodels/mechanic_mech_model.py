"""Mechanical helper model for fully coupled flow/mechanics systems."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel, MechanicOperators


class MechanicMechModel(MechanicModel):
    """MRST ``MechanicMechModel`` behavior on PyMRST mechanics operators.

    The helper is not a standalone timestep model.  It declares the mechanics
    portion of the coupled state and updates only those variables after the
    coupled linear system has been solved.
    """

    primary_variable_names = ("xd",)
    variable_names = ("xd", "uu", "u", "stress", "strain", "vdiv")

    def get_equations(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "MechanicMechModel is a coupled helper and cannot assemble a "
            "standalone timestep problem"
        )

    getEquations = get_equations

    def get_all_vars_names(self) -> list[str]:
        return list(self.variable_names)

    getAllVarsNames = get_all_vars_names

    def get_all_primary_variables(self) -> list[str]:
        return list(self.primary_variable_names)

    getAllPrimaryVariables = get_all_primary_variables

    def get_variable_field(self, name: str):
        """Return the state field and optional component index for a variable."""
        aliases = {"uu": "u", "vdiv": "volumetric_strain"}
        field = aliases.get(name, name)
        if name not in self.variable_names:
            raise KeyError(f"Unknown mechanics variable {name!r}")
        return field, None

    getVariableField = get_variable_field

    def update_state(
        self,
        state: Mapping[str, Any],
        problem_or_increment: Any,
        increment: Any | None = None,
        driving_forces: Any = None,
    ):
        """Update mechanics unknowns and rematerialize all derived fields."""
        out = state.copy()
        coupled = increment is not None
        if increment is None:
            raw = problem_or_increment
        else:
            raw = increment
            names = getattr(problem_or_increment, "primary_variables", None)
            if names is None:
                names = getattr(problem_or_increment, "primaryVariables", None)
            if names is not None and isinstance(raw, (list, tuple)):
                try:
                    raw = raw[list(names).index("xd")]
                except ValueError as error:
                    raise ValueError("coupled problem has no 'xd' primary variable") from error
        dx = np.asarray(raw, dtype=float).ravel()
        if coupled and dx.size > self.n_free:
            # Coupled Newton vectors place the mechanics block last.
            dx = dx[-self.n_free:]
        if dx.size != self.n_free:
            raise ValueError(
                f"mechanics increment has {dx.size} entries, expected {self.n_free}"
            )
        old = np.asarray(out.get("xd", np.zeros(self.n_free)), dtype=float)
        out["xd"] = old + dx
        out["u"] = self.full_displacement(out["xd"])
        pressure = np.asarray(out.get("pressure", np.zeros(self.G.cells.num)), dtype=float)
        derived = self.diagnostics({"u": out["u"], "pressure": pressure})
        out["strain"] = derived["strain"]
        out["stress"] = derived["effective_stress"]
        out["vdiv"] = derived["volumetric_strain"]
        return out, {}

    updateState = update_state


__all__ = ["MechanicMechModel", "MechanicModel", "MechanicOperators"]
