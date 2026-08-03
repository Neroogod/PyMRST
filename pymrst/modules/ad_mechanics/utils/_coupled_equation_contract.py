"""Shared output contract for translated ad-mechanics equation functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def equation_metadata(model: Any, phases: Sequence[str], eqs: Sequence[Any]):
    """Build MRST-compatible names/types for a coupled residual list.

    Native PyMRST models order equations as cell conservation, phase well
    fluxes, well control, and mechanics.  Models without facilities simply
    end after the cell and mechanics equations.
    """
    names = list(phases)
    types = ["cell"] * len(phases)
    remaining = len(eqs) - len(phases)
    if remaining <= 0:
        return names[: len(eqs)], types[: len(eqs)]

    has_facility = hasattr(model, "facility") or hasattr(model, "FacilityModel")
    if has_facility and remaining >= 2:
        q_names = (list(model._q_names()) if hasattr(model, "_q_names")
                   else [f"q{phase[:1].upper()}s" for phase in phases])
        n_well_eq = max(0, remaining - 2)
        names.extend(q_names[:n_well_eq])
        if n_well_eq > len(q_names):
            names.extend(f"well_{i + 1}" for i in range(len(q_names), n_well_eq))
        types.extend(["well"] * n_well_eq)
        names.append("control")
        types.append("well")
        remaining = len(eqs) - len(names)

    if remaining > 0:
        names.extend(["mechanics"] + [f"equation_{i + 1}" for i in range(remaining - 1)])
        types.extend(["mechanics"] + ["unknown"] * (remaining - 1))
    return names, types


def output_state(model: Any, state: Any):
    """Return an output state with full displacement when it can be formed."""
    out = state.copy() if hasattr(state, "copy") else state
    if not hasattr(out, "__contains__") or "xd" not in out or "u" in out:
        return out
    mech = getattr(model, "mech", getattr(model, "mechModel", None))
    if mech is not None and hasattr(mech, "full_displacement"):
        out["u"] = mech.full_displacement(out["xd"])
    return out


def format_equation_output(model, state, eqs, phases, return_metadata=False):
    """Return the native equation list or MRST's four-output tuple."""
    if not return_metadata:
        return eqs
    names, types = equation_metadata(model, phases, eqs)
    return eqs, names, types, output_state(model, state)

