"""Mass fraction from non-unit mole fractions.

Port of MRST ``getNonUnitMassFraction.m``.
"""

from __future__ import annotations

import numpy as np


def _molar_mass(model) -> np.ndarray:
    eos = getattr(model, "EOSModel", model)
    mixture = getattr(eos, "CompositionalMixture", None)
    if mixture is None:
        mixture = getattr(eos, "fluid", None)
    if mixture is None:
        mixture = getattr(eos, "mixture", None)
    if mixture is None:
        raise ValueError("model must expose EOSModel/CompositionalMixture with molarMass")
    return np.asarray(mixture.molarMass, float)


def get_non_unit_mass_fraction(model, molfraction, molfraction_normalized):
    """Convert non-unit mole fractions to normalized mass fractions."""
    mw = _molar_mass(model)
    if isinstance(molfraction, (list, tuple)):
        masses = [molfraction[i] * mw[i] for i in range(len(molfraction))]
        total = masses[0] * 0.0
        for i in range(len(molfraction_normalized)):
            total = total + molfraction_normalized[i] * mw[i]
        return [mass / total for mass in masses]

    mol = np.asarray(molfraction, float)
    mol_norm = np.asarray(molfraction_normalized, float)
    mass = mol * mw
    total = np.sum(mol_norm * mw, axis=1, keepdims=True)
    return mass / total


def getNonUnitMassFraction(model, molfraction, molfraction_normalized):
    """MRST camelCase alias for :func:`get_non_unit_mass_fraction`."""
    return get_non_unit_mass_fraction(model, molfraction, molfraction_normalized)


__all__ = ["getNonUnitMassFraction", "get_non_unit_mass_fraction"]
