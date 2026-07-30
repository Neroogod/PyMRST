"""MRST ``getPolymerShearMultiplier.m`` nonlinear shear correction."""

from time import perf_counter
import warnings

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.utils.adi import init_variables_adi, value


def get_polymer_shear_multiplier(model, water_velocity, viscosity_multiplier):
    """Solve MRST's local nonlinear polymer shear-velocity equation."""
    started = perf_counter()
    fluid = getattr(model, "raw_fluid", getattr(model, "fluid", None))
    shear = getattr(fluid, "plyshear_mult", getattr(fluid, "plyshearMult", None))
    if shear is None:
        raise AttributeError("model fluid must define plyshear_mult or plyshearMult")
    velocity0 = np.abs(np.asarray(value(water_velocity), dtype=float).ravel())
    mu_mult = np.asarray(value(viscosity_multiplier), dtype=float).ravel()
    if velocity0.shape != mu_mult.shape:
        raise ValueError("water_velocity and viscosity_multiplier must have equal shape")

    velocity = velocity0.copy()
    residual = np.inf
    iterations = 0
    for iterations in range(31):
        (velocity_ad,) = init_variables_adi(velocity)
        equation = velocity_ad*(1.0 + shear(velocity_ad)*(mu_mult-1.0)) \
            - velocity0*mu_mult
        residual = float(np.linalg.norm(value(equation), ord=np.inf))
        if residual < 1e-15:
            break
        jacobian = -equation.jac[0]
        velocity = value(velocity_ad) + spla.spsolve(jacobian.tocsc(), value(equation))
    converged = residual <= 1e-15
    if not converged:
        warnings.warn(
            f"polymer shear convergence failure: residual={residual:.2e}",
            RuntimeWarning, stacklevel=2,
        )
    multiplier = mu_mult/(1.0 + np.asarray(value(shear(velocity)), dtype=float)*(mu_mult-1.0))
    report = {
        "Iterations": iterations,
        "Residual": residual,
        "Converged": converged,
        "Time": perf_counter()-started,
    }
    return multiplier, velocity, report


getPolymerShearMultiplier = get_polymer_shear_multiplier

__all__ = ["get_polymer_shear_multiplier", "getPolymerShearMultiplier"]
