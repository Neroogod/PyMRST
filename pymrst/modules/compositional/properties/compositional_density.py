"""EOS-derived density for liquid/vapour phases."""

from dataclasses import dataclass

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.properties import mass_density


def compositional_density(
    mix: Mixture,
    p,
    T,
    x,
    y,
    Z_L,
    Z_V,
    volume_shift=None,
) -> tuple:
    """Return liquid and vapour mass densities."""
    rho_l = mass_density(mix, p, T, x, Z_L, volume_shift)
    rho_v = mass_density(mix, p, T, y, Z_V, volume_shift)
    return rho_l, rho_v


@dataclass
class CompositionalDensity:
    """MRST `CompositionalDensity` style evaluator."""

    mix: Mixture
    volume_shift: object = None

    def evaluate(self, p, T, x, y, Z_L, Z_V) -> tuple:
        return compositional_density(self.mix, p, T, x, y, Z_L, Z_V, self.volume_shift)


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.compositional.eos import flash
    from pymrst.modules.compositional.eos import simple_mixture

    def test_liquid_density_exceeds_vapour_density():
        mix = simple_mixture()
        p = np.array([50e5])
        T = np.array([350.0])
        fr = flash(mix, p, T, np.array([[0.5, 0.3, 0.2]]))
        rho_l, rho_v = compositional_density(mix, p, T, fr.x, fr.y, fr.Z_L, fr.Z_V)
        assert rho_l[0] > rho_v[0]
        print("ok test_liquid_density_exceeds_vapour_density")

    test_liquid_density_exceeds_vapour_density()
    print("\nAll tests passed!")


__all__ = ["CompositionalDensity", "compositional_density"]
