"""Shared fixed-stress driving-force contract."""

from __future__ import annotations

from typing import Any


class FixedStressFluidMixin:
    equation_assembler: Any = None

    def get_valid_driving_forces(self) -> dict[str, Any]:
        return {"W": None, "bc": None, "src": None, "fixedStressTerms": None}

    getValidDrivingForces = get_valid_driving_forces

    @staticmethod
    def mechanical_strain_terms(state0: Any, state: Any, driving_forces: Any):
        if isinstance(driving_forces, dict):
            terms = driving_forces.get("fixedStressTerms")
        else:
            terms = getattr(driving_forces, "fixedStressTerms", None)
        if terms is None:
            raise ValueError("fixed-stress equations require 'fixedStressTerms'")
        new = terms["new"] if isinstance(terms, dict) else terms.new
        old = terms["old"] if isinstance(terms, dict) else terms.old

        def field(owner: Any, name: str):
            return owner[name] if isinstance(owner, dict) else getattr(owner, name)

        p, p0 = state["pressure"], state0["pressure"]
        mech_new = field(new, "sTerm") + field(new, "pTerm") * p
        mech_old = field(old, "sTerm") + field(old, "pTerm") * p0
        return {"new": mech_new, "old": mech_old}

    def get_equations(
        self,
        state0: Any,
        state: Any,
        dt: float,
        driving_forces: Any,
        **kwargs: Any,
    ):
        mech_term = self.mechanical_strain_terms(state0, state, driving_forces)
        assembler = getattr(self, "equation_assembler", None)
        if assembler is None:
            raise RuntimeError(
                f"{type(self).__name__} needs an equation_assembler supplied "
                "by the concrete flow backend"
            )
        return assembler(state0, state, dt, mech_term, driving_forces, **kwargs)

    getEquations = get_equations
