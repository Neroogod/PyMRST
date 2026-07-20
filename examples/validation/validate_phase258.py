"""Phase 258: compositional fraction utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture  # noqa: E402
from pymrst.modules.compositional.utils import (  # noqa: E402
    ensureMinimumFraction,
    ensure_minimum_fraction,
    getNonUnitMassFraction,
    getNonUnitMoleFraction,
    get_non_unit_mass_fraction,
    get_non_unit_mole_fraction,
)


class _Model:
    def __init__(self, eos):
        self.EOSModel = eos


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _mixture() -> Mixture:
    return Mixture(
        names=("A", "B", "C"),
        Tcrit=np.array([190.0, 370.0, 620.0]),
        Pcrit=np.array([4.6e6, 4.2e6, 2.1e6]),
        Vcrit=np.array([1.0e-4, 2.0e-4, 6.0e-4]),
        acentric=np.array([0.01, 0.15, 0.49]),
        molarMass=np.array([0.02, 0.04, 0.10]),
    )


def _columns(mat: np.ndarray) -> list[np.ndarray]:
    return [mat[:, i] for i in range(mat.shape[1])]


def main() -> int:
    mix = _mixture()
    model = _Model(EquationOfStateModel(None, mix))

    z = np.array([[0.0, 0.2, 0.8], [0.3, -0.1, 0.8]])
    floored, tol = ensureMinimumFraction(z, 1.0e-3)
    expected_floor = np.maximum(z, 1.0e-3)
    expected_floor = expected_floor / expected_floor.sum(axis=1, keepdims=True)

    mol = np.array([[0.2, 0.3, 0.1], [0.5, 0.2, 0.1]])
    mol_norm = np.array([[0.25, 0.50, 0.25], [0.60, 0.30, 0.10]])
    expected_mass = mol * mix.molarMass
    expected_mass = expected_mass / np.sum(mol_norm * mix.molarMass, axis=1, keepdims=True)
    mass = getNonUnitMassFraction(model, mol, mol_norm)
    mass_list = get_non_unit_mass_fraction(model, _columns(mol), _columns(mol_norm))

    mass_in = np.array([[0.1, 0.2, 0.4], [0.5, 0.3, 0.1]])
    mass_norm = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    expected_mole = mass_in / mix.molarMass
    expected_mole = expected_mole / np.sum(mass_norm / mix.molarMass, axis=1, keepdims=True)
    mole = getNonUnitMoleFraction(model, mass_in, mass_norm)
    mole_list = get_non_unit_mole_fraction(model, _columns(mass_in), _columns(mass_norm))

    one_d, one_d_tol = ensure_minimum_fraction([0.0, 0.2, 0.8])
    expected_one_d = np.maximum(np.array([0.0, 0.2, 0.8]), 1.0e-8)
    expected_one_d = expected_one_d / expected_one_d.sum()

    print("\nPhase 258: compositional fraction utility parity")
    failures = 0
    failures += _check("ensureMinimumFraction floors and row-normalizes",
                       np.allclose(floored, expected_floor) and tol == 1.0e-3)
    failures += _check("ensure_minimum_fraction handles vector input",
                       np.allclose(one_d, expected_one_d)
                       and np.isclose(one_d_tol, 1.0e-8))
    failures += _check("getNonUnitMassFraction matrix path matches MRST formula",
                       np.allclose(mass, expected_mass))
    failures += _check("get_non_unit_mass_fraction list path matches matrix columns",
                       all(np.allclose(mass_list[i], expected_mass[:, i]) for i in range(3)))
    failures += _check("getNonUnitMoleFraction matrix path matches MRST formula",
                       np.allclose(mole, expected_mole))
    failures += _check("get_non_unit_mole_fraction list path matches matrix columns",
                       all(np.allclose(mole_list[i], expected_mole[:, i]) for i in range(3)))

    if failures:
        print(f"\nPhase 258 failed ({failures} checks).")
        return 1
    print("\nFraction utility layout now mirrors MRST's compositional helpers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
