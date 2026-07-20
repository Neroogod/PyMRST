"""Small-strain plasticity helpers for geomechanics material updates."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlasticityState:
    """Cell-wise plastic state for von Mises material updates."""

    plastic_strain: np.ndarray
    eq_plastic_strain: np.ndarray
    yielded: np.ndarray


def empty_plasticity_state(n: int) -> PlasticityState:
    """Return a zero plasticity state for ``n`` material points."""
    return PlasticityState(
        plastic_strain=np.zeros((n, 6)),
        eq_plastic_strain=np.zeros(n),
        yielded=np.zeros(n, dtype=bool),
    )


def _as_strain6(strain: np.ndarray) -> tuple[np.ndarray, bool]:
    strain = np.asarray(strain, dtype=float)
    single = strain.ndim == 1
    strain = np.atleast_2d(strain)
    if strain.shape[1] == 3:
        strain = np.column_stack([
            strain[:, 0], strain[:, 1], np.zeros(strain.shape[0]),
            np.zeros(strain.shape[0]), np.zeros(strain.shape[0]), strain[:, 2],
        ])
    if strain.shape[1] != 6:
        raise ValueError("strain must have 3 plane-strain or 6 3D components")
    return strain, single


def _deviatoric(stress: np.ndarray) -> np.ndarray:
    out = stress.copy()
    mean = np.mean(out[:, :3], axis=1)
    out[:, :3] -= mean[:, None]
    return out


def _j2_inner(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]
        + 2.0 * (a[:, 3] * b[:, 3] + a[:, 4] * b[:, 4] + a[:, 5] * b[:, 5])
    )


def von_mises_equivalent(stress: np.ndarray) -> np.ndarray:
    """Return von Mises equivalent stress for Voigt stress vectors.

    Component order is ``[xx, yy, zz, yz, xz, xy]``.
    """
    stress = np.atleast_2d(np.asarray(stress, dtype=float))
    s = _deviatoric(stress)
    return np.sqrt(np.maximum(1.5 * _j2_inner(s, s), 0.0))


def isotropic_elastic_stress(
    strain: np.ndarray,
    young: float,
    poisson: float,
) -> np.ndarray:
    """Linear isotropic elastic stress in 3D Voigt form.

    Strain uses engineering shear components in order
    ``[xx, yy, zz, yz, xz, xy]``. A 3-component input is interpreted as 2D
    plane strain ``[xx, yy, gamma_xy]``.
    """
    eps, single = _as_strain6(strain)
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    tr = np.sum(eps[:, :3], axis=1)
    stress = np.zeros_like(eps)
    stress[:, :3] = lam * tr[:, None] + 2.0 * mu * eps[:, :3]
    stress[:, 3:] = mu * eps[:, 3:]
    return stress[0] if single else stress


def von_mises_return_map(
    strain: np.ndarray,
    young: float,
    poisson: float,
    yield_stress: float,
    hardening: float = 0.0,
    state: Optional[PlasticityState] = None,
    tol: float = 1e-10,
) -> tuple[np.ndarray, PlasticityState, np.ndarray]:
    """Backward-Euler radial return for J2 plasticity.

    Returns ``(stress, new_state, plastic_multiplier)``. The algorithm is the
    standard small-strain isotropic-hardening update used as the material-point
    core for elastoplastic mechanics workflows.
    """
    eps, single = _as_strain6(strain)
    n = eps.shape[0]
    if state is None:
        state = empty_plasticity_state(n)
    if state.plastic_strain.shape != (n, 6):
        raise ValueError("plastic state size does not match strain")

    elastic_strain = eps - state.plastic_strain
    trial = isotropic_elastic_stress(elastic_strain, young, poisson)
    trial = np.atleast_2d(trial)
    seq = von_mises_equivalent(trial)
    flow = seq - (yield_stress + hardening * state.eq_plastic_strain)

    mu = young / (2.0 * (1.0 + poisson))
    dg = np.zeros(n)
    plastic = flow > tol * max(abs(yield_stress), 1.0)
    dg[plastic] = flow[plastic] / (3.0 * mu + hardening)

    stress = trial.copy()
    new_plastic_strain = state.plastic_strain.copy()
    new_eq = state.eq_plastic_strain.copy()
    if np.any(plastic):
        s_trial = _deviatoric(trial[plastic])
        seq_p = seq[plastic]
        scale = np.maximum(1.0 - 3.0 * mu * dg[plastic] / seq_p, 0.0)
        s_new = s_trial * scale[:, None]
        mean = np.mean(trial[plastic, :3], axis=1)
        stress[plastic, :3] = s_new[:, :3] + mean[:, None]
        stress[plastic, 3:] = s_new[:, 3:]

        direction = np.zeros_like(s_trial)
        direction[:, :3] = 1.5 * s_trial[:, :3] / seq_p[:, None]
        direction[:, 3:] = 3.0 * s_trial[:, 3:] / seq_p[:, None]
        new_plastic_strain[plastic] += dg[plastic, None] * direction
        new_eq[plastic] += dg[plastic]

    new_state = PlasticityState(
        plastic_strain=new_plastic_strain,
        eq_plastic_strain=new_eq,
        yielded=plastic,
    )
    if single:
        return stress[0], PlasticityState(
            plastic_strain=new_state.plastic_strain[:1],
            eq_plastic_strain=new_state.eq_plastic_strain[:1],
            yielded=new_state.yielded[:1],
        ), dg[:1]
    return stress, new_state, dg
