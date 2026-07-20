"""Liquid/vapour Peng-Robinson compressibility factors."""

from dataclasses import dataclass

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.eos import fugacity_coefficients


def phase_compressibility_factors_lv(mix: Mixture, p, T, x, y) -> tuple:
    """Return liquid and vapour `Z` factors."""
    _, z_l = fugacity_coefficients(mix, p, T, x, vapour=False)
    _, z_v = fugacity_coefficients(mix, p, T, y, vapour=True)
    return z_l, z_v


@dataclass
class PhaseCompressibilityFactorsLV:
    """MRST `PhaseCompressibilityFactorsLV` style evaluator."""

    mix: Mixture

    def evaluate(self, p, T, x, y) -> tuple:
        return phase_compressibility_factors_lv(self.mix, p, T, x, y)


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.compositional.eos import flash
    from pymrst.modules.compositional.eos import simple_mixture

    def test_z_matches_flash_roots():
        mix = simple_mixture()
        p = np.array([50e5])
        T = np.array([350.0])
        fr = flash(mix, p, T, np.array([[0.5, 0.3, 0.2]]))
        z_l, z_v = phase_compressibility_factors_lv(mix, p, T, fr.x, fr.y)
        assert np.allclose(z_l, fr.Z_L)
        assert np.allclose(z_v, fr.Z_V)
        print("ok test_z_matches_flash_roots")

    test_z_matches_flash_roots()
    print("\nAll tests passed!")


__all__ = ["PhaseCompressibilityFactorsLV", "phase_compressibility_factors_lv"]
