"""Component mole fractions for liquid/vapour phases."""

from dataclasses import dataclass

import numpy as np


def _columns(values) -> list[np.ndarray]:
    if isinstance(values, (list, tuple)):
        return [np.asarray(v, float) for v in values]
    arr = np.asarray(values, float)
    return [arr[:, i] for i in range(arr.shape[1])]


def component_phase_mole_fractions_lv(x, y, nph: int = 2) -> list[list[np.ndarray | None]]:
    """Return an `ncomp x nph` matrix of liquid/vapour mole fractions."""
    x_cols = _columns(x)
    y_cols = _columns(y)
    ncomp = len(x_cols)
    out: list[list[np.ndarray | None]] = [[None for _ in range(nph)] for _ in range(ncomp)]
    for i in range(ncomp):
        out[i][0] = x_cols[i]
        out[i][1] = y_cols[i]
    return out


@dataclass
class ComponentPhaseMoleFractionsLV:
    """MRST `ComponentPhaseMoleFractionsLV` style evaluator."""

    nph: int = 2

    def evaluate(self, x, y) -> list[list[np.ndarray | None]]:
        return component_phase_mole_fractions_lv(x, y, self.nph)


if __name__ == "__main__":
    def test_mole_fraction_matrix_shape():
        x = np.array([[0.7, 0.3], [0.8, 0.2]])
        y = np.array([[0.9, 0.1], [0.95, 0.05]])
        mf = component_phase_mole_fractions_lv(x, y)
        assert len(mf) == 2
        assert len(mf[0]) == 2
        assert np.allclose(mf[0][0], x[:, 0])
        assert np.allclose(mf[1][1], y[:, 1])
        print("ok test_mole_fraction_matrix_shape")

    test_mole_fraction_matrix_shape()
    print("\nAll tests passed!")


__all__ = ["ComponentPhaseMoleFractionsLV", "component_phase_mole_fractions_lv"]
