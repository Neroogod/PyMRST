"""MRST ``computeInitAquifer.m`` compatibility entry point."""

import numpy as np

from pymrst.modules.ad_core.models.aquifers.aquifer import (
    equilibrium_aquifer_pressure,
)


def compute_init_aquifer(model, initial_state, initial_aquifer_volumes):
    """Seed equilibrium aquifer pressures and the supplied initial volumes.

    MRST 2024a computes the zero-flux least-squares pressure but its final line
    refers to undefined ``V_aq``. PyMRST implements the evident intended value,
    ``initaqvolumes``, while retaining the same equilibrium calculation.
    """
    aquifer = getattr(model, "aquifer", None)
    if aquifer is None:
        raise AttributeError("model must define an aquifer")
    volumes = np.asarray(initial_aquifer_volumes, dtype=float).ravel()
    p_w, b_w = aquifer._water_props(model, initial_state)
    fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
    rho = getattr(fluid, "rhoS", getattr(model.fluid, "rho_s", None))
    rho_ws = float(np.asarray(rho).ravel()[0])
    gravity = float(getattr(model, "gravity", 0.0))
    pressures = equilibrium_aquifer_pressure(
        aquifer.connections, p_w, b_w, rho_ws, gravity,
    )
    if volumes.size != pressures.size:
        raise ValueError("one initial aquifer volume is required per aquifer")
    keys = getattr(aquifer, "state_arg_keys", ("aquifer_p", "aquifer_V"))
    initial_state[keys[0]] = pressures
    initial_state[keys[1]] = volumes.copy()
    aquifer.pressures = pressures.copy()
    aquifer.volumes = volumes.copy()
    return initial_state


computeInitAquifer = compute_init_aquifer

__all__ = ["compute_init_aquifer", "computeInitAquifer"]
