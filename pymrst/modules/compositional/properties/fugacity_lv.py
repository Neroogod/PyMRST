"""Liquid/vapour fugacity property (MRST `FugacityLV`)."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.compositional.eos import Mixture, fugacity_coefficients


def fugacity_lv(mix: Mixture, p, T, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Return liquid and vapour component fugacities.

    The output arrays have shape `(ncells, ncomponents)` and correspond to
    MRST's cell array `f(:, liquidIndex)` and `f(:, vaporIndex)`.
    """
    p_arr = np.atleast_1d(np.asarray(p, dtype=float))
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    y_arr = np.atleast_2d(np.asarray(y, dtype=float))
    ln_phi_l, _ = fugacity_coefficients(mix, p_arr, T_arr, x_arr, vapour=False)
    ln_phi_v, _ = fugacity_coefficients(mix, p_arr, T_arr, y_arr, vapour=True)
    return np.exp(ln_phi_l) * p_arr[:, None] * x_arr, np.exp(ln_phi_v) * p_arr[:, None] * y_arr


@dataclass
class FugacityLV:
    """State-function style evaluator for liquid/vapour fugacity."""

    mixture: Mixture

    def evaluate(self, p, T, x, y) -> tuple[np.ndarray, np.ndarray]:
        return fugacity_lv(self.mixture, p, T, x, y)

    __call__ = evaluate


if __name__ == "__main__":
    from pymrst.modules.compositional.fluids import get_benchmark_mixture
    from pymrst.modules.compositional.eos import flash

    mixture, _ = get_benchmark_mixture("simple")
    pressure = np.array([5.0e6])
    temperature = np.array([340.0])
    overall = np.array([[0.45, 0.35, 0.20]])
    result = flash(mixture, pressure, temperature, overall)
    f_l, f_v = fugacity_lv(mixture, pressure, temperature, result.x, result.y)
    assert f_l.shape == (1, 3)
    assert f_v.shape == (1, 3)
    assert np.all(np.isfinite(f_l))
    assert np.all(np.isfinite(f_v))
