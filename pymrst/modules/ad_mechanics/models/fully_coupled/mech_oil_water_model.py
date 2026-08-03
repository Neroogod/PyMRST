"""Fully coupled two-phase oil-water poromechanics.

This is the native owner corresponding to MRST ``MechOilWaterModel.m``.  It
uses the shared black-oil conservation machinery, but deliberately fixes the
model contract to two immiscible phases and exposes the coupling operation
owned by the MRST oil-water class.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import (
    MechBlackOilModel,
)


class MechOilWaterModel(MechBlackOilModel):
    """Two-phase oil-water flow fully coupled to VEM mechanics.

    Unlike the former alias, this class rejects gas/live-oil configurations,
    has an oil-water primary-variable contract, and owns the coupling-term
    calculation used by equation and adjoint assembly.
    """

    phase_names = ("water", "oil")
    primary_variable_names = ("pressure", "sW", "well_variables", "xd")

    def __init__(self, *args: Any, disgas: bool = False, **kwargs: Any):
        if disgas:
            raise ValueError("MechOilWaterModel does not support dissolved gas")
        super().__init__(*args, disgas=False, **kwargs)
        if self.n_phases != 2:
            raise ValueError(
                f"MechOilWaterModel requires exactly two phases, got {self.n_phases}"
            )

    def compute_coupling_terms(self, state0, state) -> tuple[dict[str, Any], Any]:
        """Return old/new volumetric strain and the mechanics fluid pressure.

        MRST's ``computeCouplingTerms`` reconstructs displacement derivatives
        before applying its volume-integrated divergence.  PyMRST states store
        the full displacement, and ``volumetric_strain_full`` performs the
        equivalent operation while preserving AD values.
        """
        if "u" not in state0 or "u" not in state:
            raise KeyError("oil-water mechanics states require full displacement 'u'")
        old = self.mech.volumetric_strain_full(state0["u"])
        new = self.mech.volumetric_strain_full(state["u"])
        return {"old": old, "new": new}, state["pressure"]

    # MRST-compatible spelling used by translated workflows.
    computeCouplingTerms = compute_coupling_terms

    def get_adjoint_equations(self, state0, state, dt: float, **kwargs: Any):
        """Assemble the oil-water residual for reverse/adjoint consumers.

        PyMRST's AD objects retain the Jacobian required by the generic adjoint
        driver, so the numerical residual is the same as the forward assembly.
        ``reverse_mode`` is accepted for MRST call-site compatibility.
        """
        kwargs.pop("reverse_mode", None)
        kwargs.pop("reverseMode", None)
        return self.get_equations(state0, state, dt)

    getAdjointEquations = get_adjoint_equations


__all__ = ["MechOilWaterModel"]
