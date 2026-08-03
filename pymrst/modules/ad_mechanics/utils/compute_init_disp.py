"""Initial mechanics displacement helper (MRST ``computeInitDisp.m`` layout)."""

import numpy as np


def compute_init_disp(
    model,
    state,
    uu=None,
    pressure=None,
    *,
    mech_equil: bool = True,
    copy: bool = True,
):
    """Seed a state with mechanics equilibrium displacement.

    Parameters mirror the common MRST use: provide a coupled model with a
    mechanics object and a pressure field, and the state receives ``xd`` and
    full nodal displacement ``u``.
    """
    out = state.copy() if copy and hasattr(state, "copy") else dict(state)
    mech = getattr(model, "mech", None) or getattr(model, "mechModel", None)
    if mech is None:
        raise AttributeError("model must expose a 'mech' or 'mechModel'")
    if mech_equil:
        if pressure is None:
            pressure = out.get("pressure", np.zeros(mech.G.cells.num))
        out["xd"] = mech.solve(np.asarray(pressure, dtype=float))
        out["u"] = mech.full_displacement(out["xd"])
    else:
        if uu is None:
            raise ValueError("uu is required when mech_equil is false")
        full = np.asarray(uu, dtype=float)
        if full.ndim == 2:
            full = full.reshape(-1)
        if full.size != mech.operators.is_dir.size:
            raise ValueError("uu has the wrong number of displacement degrees of freedom")
        prescribed = np.asarray(mech.operators.u_dir, dtype=float)
        if not np.allclose(full[mech.operators.is_dir], prescribed[mech.operators.is_dir]):
            raise ValueError("uu conflicts with prescribed Dirichlet displacement")
        out["u"] = full.copy()
        out["xd"] = full[~mech.operators.is_dir]

    from pymrst.modules.ad_mechanics.utils.add_derived_quantities import add_derived_quantities

    return add_derived_quantities(mech, out, copy=False)


__all__ = ["compute_init_disp"]


if __name__ == "__main__":
    assert callable(compute_init_disp)
    print("ok compute_init_disp import")
