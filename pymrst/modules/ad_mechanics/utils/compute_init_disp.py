"""Initial mechanics displacement helper (MRST ``computeInitDisp.m`` layout)."""

import numpy as np


def compute_init_disp(model, state, pressure=None, *, copy: bool = True):
    """Seed a state with mechanics equilibrium displacement.

    Parameters mirror the common MRST use: provide a coupled model with a
    mechanics object and a pressure field, and the state receives ``xd`` and
    full nodal displacement ``u``.
    """
    out = state.copy() if copy and hasattr(state, "copy") else dict(state)
    mech = getattr(model, "mech", None) or getattr(model, "mechModel", None)
    if mech is None:
        raise AttributeError("model must expose a 'mech' or 'mechModel'")
    if pressure is None:
        pressure = out["pressure"]
    out["xd"] = mech.solve(np.asarray(pressure, dtype=float))
    out["u"] = mech.full_displacement(out["xd"])
    return out


__all__ = ["compute_init_disp"]


if __name__ == "__main__":
    assert callable(compute_init_disp)
    print("ok compute_init_disp import")
