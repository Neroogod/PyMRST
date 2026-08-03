"""Fully coupled single-phase water poromechanics."""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_mechanics.models.poromechanics import MechWaterModel as _MechWaterKernel


class MechWaterModel(_MechWaterKernel):
    """Native MRST-layout owner for the operational water/mechanics solver."""

    phase_names = ("water",)
    primary_variable_names = ("pressure", "xd")

    def compute_coupling_terms(self, state0: Any, state: Any):
        if "u" not in state0 or "u" not in state:
            raise KeyError("water mechanics states require full displacement 'u'")
        return {
            "old": self.mech.volumetric_strain_full(state0["u"]),
            "new": self.mech.volumetric_strain_full(state["u"]),
        }, state["pressure"]

    computeCouplingTerms = compute_coupling_terms

    def get_equations(self, state0: Any, state: Any, dt: float, **kwargs: Any):
        from pymrst.modules.ad_core.statefunctions.state_function import State
        from pymrst.utils.adi import init_variables_adi

        p, xd = init_variables_adi(state["pressure"], state["xd"])
        current = State(pressure=p, xd=xd)
        return [self.flow_equation(state0, current, dt), self.mech.equations(xd, p)]

    getEquations = get_equations
    get_adjoint_equations = get_equations
    getAdjointEquations = get_equations


__all__ = ["MechWaterModel"]
