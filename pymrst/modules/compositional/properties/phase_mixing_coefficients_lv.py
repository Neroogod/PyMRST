"""Peng-Robinson phase mixing coefficients for liquid/vapour phases."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.eos_ad import _mixing_parameters
from pymrst.modules.compositional.eos_ad import _phase_mix


@dataclass
class PhaseMixingCoefficients:
    """The `Si`, `A`, `B`, `Bi`, and `A_ij` fields used by MRST."""

    Si: list
    A: object
    B: object
    Bi: list
    A_ij: list[list]


def _columns(values) -> list:
    if isinstance(values, (list, tuple)):
        return list(values)
    arr = np.asarray(values, float)
    return [arr[:, i] for i in range(arr.shape[1])]


def phase_mixing_coefficients_lv(mix: Mixture, p, T, x, y) -> tuple[PhaseMixingCoefficients, PhaseMixingCoefficients]:
    """Return liquid and vapour Peng-Robinson mixing coefficients."""
    x_cols = _columns(x)
    y_cols = _columns(y)
    a_ij, bi = _mixing_parameters(mix, p, T)
    si_l, a_l, b_l = _phase_mix(x_cols, a_ij, bi)
    si_v, a_v, b_v = _phase_mix(y_cols, a_ij, bi)
    return (
        PhaseMixingCoefficients(si_l, a_l, b_l, bi, a_ij),
        PhaseMixingCoefficients(si_v, a_v, b_v, bi, a_ij),
    )


@dataclass
class PhaseMixingCoefficientsLV:
    """MRST `PhaseMixingCoefficientsLV` style evaluator."""

    mix: Mixture

    def evaluate(self, p, T, x, y) -> tuple[PhaseMixingCoefficients, PhaseMixingCoefficients]:
        return phase_mixing_coefficients_lv(self.mix, p, T, x, y)


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.compositional.eos import simple_mixture

    def test_phase_mixing_shapes():
        mix = simple_mixture()
        p = np.array([50e5, 60e5])
        T = np.array([350.0, 350.0])
        x = np.tile([0.4, 0.3, 0.3], (2, 1))
        y = np.tile([0.8, 0.15, 0.05], (2, 1))
        liq, vap = phase_mixing_coefficients_lv(mix, p, T, x, y)
        assert len(liq.Si) == mix.n
        assert len(vap.Bi) == mix.n
        assert len(liq.A_ij) == mix.n
        print("ok test_phase_mixing_shapes")

    test_phase_mixing_shapes()
    print("\nAll tests passed!")


__all__ = ["PhaseMixingCoefficients", "PhaseMixingCoefficientsLV", "phase_mixing_coefficients_lv"]
