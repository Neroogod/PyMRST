"""Component mass fractions for liquid/vapour phases."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.properties.component_phase_mole_fractions_lv import (
    component_phase_mole_fractions_lv,
)


def _mass_fraction_columns(mix: Mixture, cols: list[np.ndarray]) -> list[np.ndarray]:
    mass = [cols[i] * mix.molarMass[i] for i in range(mix.n)]
    total = mass[0]
    for i in range(1, mix.n):
        total = total + mass[i]
    return [m / total for m in mass]


def component_phase_mass_fractions_lv(
    mix: Mixture,
    x,
    y,
    nph: int = 2,
) -> list[list[np.ndarray | None]]:
    """Return an `ncomp x nph` matrix of liquid/vapour mass fractions."""
    mole = component_phase_mole_fractions_lv(x, y, nph)
    x_cols = [mole[i][0] for i in range(mix.n)]
    y_cols = [mole[i][1] for i in range(mix.n)]
    x_mass = _mass_fraction_columns(mix, x_cols)
    y_mass = _mass_fraction_columns(mix, y_cols)
    out: list[list[np.ndarray | None]] = [[None for _ in range(nph)] for _ in range(mix.n)]
    for i in range(mix.n):
        out[i][0] = x_mass[i]
        out[i][1] = y_mass[i]
    return out


@dataclass
class ComponentPhaseMassFractionsLV:
    """MRST `ComponentPhaseMassFractionsLV` style evaluator."""

    mix: Mixture
    nph: int = 2

    def evaluate(self, x, y) -> list[list[np.ndarray | None]]:
        return component_phase_mass_fractions_lv(self.mix, x, y, self.nph)


if __name__ == "__main__":
    from pymrst.modules.compositional.eos import simple_mixture

    def test_mass_fractions_sum_per_phase():
        mix = simple_mixture()
        x = np.array([[0.5, 0.3, 0.2]])
        y = np.array([[0.8, 0.15, 0.05]])
        mass = component_phase_mass_fractions_lv(mix, x, y)
        sx = sum(mass[i][0] for i in range(mix.n))
        sy = sum(mass[i][1] for i in range(mix.n))
        assert np.allclose(sx, 1.0)
        assert np.allclose(sy, 1.0)
        assert mass[2][0] > x[:, 2]
        print("ok test_mass_fractions_sum_per_phase")

    test_mass_fractions_sum_per_phase()
    print("\nAll tests passed!")


__all__ = ["ComponentPhaseMassFractionsLV", "component_phase_mass_fractions_lv"]
