"""Shale additions to MRST's natural-variable compositional equations."""

import numpy as np

from pymrst.modules.compositional.models.natvars.equations_natural_variables import (
    equations_natural_variables,
)
from pymrst.utils.adi import value


def _composition_components(vapor_mole_fractions, n_components):
    if isinstance(vapor_mole_fractions, (list, tuple)):
        if len(vapor_mole_fractions) != n_components:
            raise ValueError("composition and isotherm component counts differ")
        return list(vapor_mole_fractions), False
    values = np.asarray(vapor_mole_fractions)
    if values.ndim != 2 or values.shape[1] != n_components:
        raise ValueError("composition must have shape (n_cells, n_components)")
    return [values[:, i] for i in range(n_components)], True


def competitive_langmuir_components(pressure, vapor_mole_fractions, isotherm, grain_volume):
    """Return one sorbed-mass vector per component, preserving AD objects."""
    iso = np.asarray(isotherm, dtype=float)
    if iso.ndim != 2 or iso.shape[0] != 2:
        raise ValueError("isotherm must have shape (2, n_components)")
    if np.any(iso[0] <= 0):
        raise ValueError("Langmuir pressures must be positive")
    components, _ = _composition_components(vapor_mole_fractions, iso.shape[1])
    denominator = 1 + pressure * sum(
        components[i] / iso[0, i] for i in range(iso.shape[1])
    )
    return [
        grain_volume * (iso[1, i] / iso[0, i])
        * (components[i] * pressure) / denominator
        for i in range(iso.shape[1])
    ]


def langmuir_sorbed_mass(pressure, vapor_mole_fractions, isotherm, grain_volume):
    """Return competitive Langmuir sorbed mass for every cell/component."""
    iso = np.asarray(isotherm, dtype=float)
    if iso.ndim != 2:
        raise ValueError("isotherm must have shape (2, n_components)")
    components, was_matrix = _composition_components(vapor_mole_fractions, iso.shape[1])
    result = competitive_langmuir_components(pressure, components, iso, grain_volume)
    if not was_matrix or any(hasattr(item, "jac") for item in result):
        return result
    return np.column_stack(result)


def sorbed_accumulation(p0, y0, p, y, isotherm, grain_volume, dt):
    if dt <= 0:
        raise ValueError("dt must be positive")
    current = competitive_langmuir_components(p, y, isotherm, grain_volume)
    previous = competitive_langmuir_components(p0, y0, isotherm, grain_volume)
    result = [(now - old) / dt for now, old in zip(current, previous)]
    if isinstance(y, np.ndarray) and not any(hasattr(item, "jac") for item in result):
        return np.column_stack(result)
    return result


def add_sorbed_accumulation(equations, accumulation, *, water=False):
    """Add componentwise sorption to a base natural-variable residual list."""
    offset = 1 if water else 0
    if len(equations) < offset + len(accumulation):
        raise ValueError("equation list does not contain every component residual")
    out = list(equations)
    for component, correction in enumerate(accumulation):
        out[offset + component] = out[offset + component] + correction
    return out


def gas_diffusion_flux(diffusion, tortuosity, sg_rho_poro_face, composition_density_gradient):
    """Gas-only Fickian component flux used by the MRST shale equation."""
    diffusion = np.asarray(diffusion, dtype=float).reshape(1, -1)
    if tortuosity <= 0:
        raise ValueError("tortuosity must be positive")
    gradient = composition_density_gradient
    if gradient.shape[1] != diffusion.shape[1]:
        raise ValueError("diffusion and gradient component counts differ")
    return -(diffusion / tortuosity) * sg_rho_poro_face[:, None] * gradient


def non_darcy_velocity(darcy_velocity, rho_mobility_permeability_beta, *, positive_root=True):
    """Apply the shale Forchheimer velocity correction."""
    argument = 1 + 4 * rho_mobility_permeability_beta * darcy_velocity
    if not positive_root:
        argument = 1 - 4 * rho_mobility_permeability_beta * darcy_velocity
    if np.any(value(argument) < 0):
        raise ValueError("non-Darcy square-root argument is negative")
    return 2 * darcy_velocity / (1 + argument**0.5)


def apply_gangi_velocity(velocity, face_multiplier):
    return velocity * face_multiplier


def equations_shale_natural_vars(state0, state, model, dt, driving_forces=None, *args, **kwargs):
    """Assemble the base problem and identify active native shale mechanisms.

    Residual integration is performed by model-level assembly hooks; retaining
    the base equation owner avoids a divergent copy of PyMRST flash/fugacity
    logic while the shale hooks are certified independently.
    """
    problem, updated = equations_natural_variables(
        state0, state, model, dt, driving_forces, *args, **kwargs
    )
    rock = getattr(model, "shale_rock", None)
    mechanisms = {} if rock is None else dict(rock.shale_mechanisms)
    problem.metadata["shale_mechanisms"] = mechanisms
    problem.metadata["shale_kernels"] = {
        "sorption": sorbed_accumulation,
        "diffusion": gas_diffusion_flux,
        "non_darcy": non_darcy_velocity,
        "gangi": apply_gangi_velocity,
    }
    return problem, updated


__all__ = [
    "add_sorbed_accumulation",
    "apply_gangi_velocity",
    "competitive_langmuir_components",
    "equations_shale_natural_vars",
    "gas_diffusion_flux",
    "langmuir_sorbed_mass",
    "non_darcy_velocity",
    "sorbed_accumulation",
]
