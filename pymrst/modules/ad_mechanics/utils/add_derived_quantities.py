"""MRST ``addDerivedQuantities.m`` compatibility helper."""

from __future__ import annotations

import numpy as np


def add_derived_quantities(model, state, *, copy: bool = True):
    """Add mechanical diagnostic fields to a state.

    The MRST helper stores full displacement, cell divergence, strain and
    effective stress derived from the mechanics operators. PyMRST keeps the
    same information available through ``MechanicModel.diagnostics``; this
    function materializes it on the state for MRST-shaped workflows.
    """
    out = state.copy() if copy and hasattr(state, "copy") else dict(state)
    mech = getattr(model, "mech", None) or getattr(model, "mechModel", None)
    if mech is None and hasattr(model, "diagnostics"):
        mech = model
    if mech is None:
        raise AttributeError("model must expose a 'mech' or 'mechModel'")

    if "u" not in out:
        if "xd" not in out:
            raise KeyError("state must contain 'u' or 'xd'")
        out["u"] = mech.full_displacement(out["xd"])

    diag = mech.diagnostics(out)
    out["u"] = np.asarray(out["u"], dtype=float)
    out["uu"] = diag["displacement"]
    out["vdiv"] = diag["volumetric_strain"]
    out["divu"] = diag["volumetric_strain"]
    out["strain"] = diag["strain"]
    out["stress"] = diag["effective_stress"]
    out["effective_stress"] = diag["effective_stress"]
    out["total_stress"] = diag["total_stress"]
    out["displacement_magnitude"] = diag["displacement_magnitude"]
    return out


__all__ = ["add_derived_quantities"]


if __name__ == "__main__":
    assert callable(add_derived_quantities)
    print("ok add_derived_quantities import")
