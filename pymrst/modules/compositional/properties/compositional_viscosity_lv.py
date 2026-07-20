"""Lohrenz-Bray-Clark viscosity for liquid/vapour phases."""

from dataclasses import dataclass

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.properties import viscosity_lbc


def compositional_viscosity_lv(
    mix: Mixture,
    p,
    T,
    x,
    y,
    Z_L,
    Z_V,
    volume_shift=None,
) -> tuple:
    """Return liquid and vapour viscosities."""
    mu_l = viscosity_lbc(mix, p, T, x, Z_L, volume_shift)
    mu_v = viscosity_lbc(mix, p, T, y, Z_V, volume_shift)
    return mu_l, mu_v


@dataclass
class CompositionalViscosityLV:
    """MRST `CompositionalViscosityLV` style evaluator."""

    mix: Mixture
    volume_shift: object = None

    def evaluate(self, p, T, x, y, Z_L, Z_V) -> tuple:
        return compositional_viscosity_lv(self.mix, p, T, x, y, Z_L, Z_V, self.volume_shift)


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.compositional.eos import flash
    from pymrst.modules.compositional.eos import simple_mixture

    def test_viscosity_positive():
        mix = simple_mixture()
        p = np.array([50e5])
        T = np.array([350.0])
        fr = flash(mix, p, T, np.array([[0.5, 0.3, 0.2]]))
        mu_l, mu_v = compositional_viscosity_lv(mix, p, T, fr.x, fr.y, fr.Z_L, fr.Z_V)
        assert mu_l[0] > 0.0
        assert mu_v[0] > 0.0
        print("ok test_viscosity_positive")

    test_viscosity_positive()
    print("\nAll tests passed!")


__all__ = ["CompositionalViscosityLV", "compositional_viscosity_lv"]
