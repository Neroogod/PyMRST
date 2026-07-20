"""Mole fraction from non-unit mass fractions.

Port of MRST ``getNonUnitMoleFraction.m``.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.utils.get_non_unit_mass_fraction import _molar_mass


def get_non_unit_mole_fraction(model, massfraction, massfraction_normalized):
    """Convert non-unit mass fractions to normalized mole fractions."""
    mw = _molar_mass(model)
    if isinstance(massfraction, (list, tuple)):
        moles = [massfraction[i] / mw[i] for i in range(len(massfraction))]
        total = moles[0] * 0.0
        for i in range(len(massfraction_normalized)):
            total = total + massfraction_normalized[i] / mw[i]
        return [mole / total for mole in moles]

    mass = np.asarray(massfraction, float)
    mass_norm = np.asarray(massfraction_normalized, float)
    mole = mass / mw
    total = np.sum(mass_norm / mw, axis=1, keepdims=True)
    return mole / total


def getNonUnitMoleFraction(model, massfraction, massfraction_normalized):
    """MRST camelCase alias for :func:`get_non_unit_mole_fraction`."""
    return get_non_unit_mole_fraction(model, massfraction, massfraction_normalized)


__all__ = ["getNonUnitMoleFraction", "get_non_unit_mole_fraction"]
