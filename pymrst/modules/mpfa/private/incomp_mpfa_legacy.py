"""MRST-shaped state/fluid wrapper for ``private/incompMPFAlegacy.m``."""

import numpy as np

from pymrst.modules.mpfa.incomp_mpfa import incomp_mpfa


def incomp_mpfa_legacy(
    state,
    g,
    mpfa_t,
    fluid,
    *,
    bc=None,
    src=None,
    wells=None,
    gravity=None,
    matrix_output=False,
):
    """Update and return an MRST-style reservoir ``state``.

    The numerical assembly lives in :func:`pymrst.modules.mpfa.incomp_mpfa`;
    this compatibility layer obtains dynamic mobility/density from the
    incompressible fluid exactly where MRST's legacy entry point does.
    """
    phase_mobility = fluid.mobility(state.s)
    total_mobility = np.sum(phase_mobility, axis=1)
    omega = (
        phase_mobility * np.asarray(fluid.rho, dtype=float)
    ).sum(axis=1) / total_mobility
    result = incomp_mpfa(
        g,
        mpfa_t,
        bc=bc,
        src=src,
        wells=wells,
        mobility=total_mobility,
        density=omega,
        gravity=gravity,
        return_result=True,
    )
    state.pressure = result.pressure
    state.flux = result.flux
    state.wellSol = result.well_sol
    # MRST calls these subface pressure unknowns boundaryPressure.
    state.boundaryPressure = result.boundary_pressure
    if matrix_output:
        # Kept as an explicit unsupported edge instead of silently returning
        # a differently reduced matrix.
        raise NotImplementedError("MatrixOutput is not yet exposed by the MPFA kernel")
    return state


# MRST-compatible spelling within the matching private module.
incompMPFAlegacy = incomp_mpfa_legacy

__all__ = ["incomp_mpfa_legacy", "incompMPFAlegacy"]
