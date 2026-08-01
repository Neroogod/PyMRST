"""Young's modulus and Poisson ratio to Lame parameters (MRST `ENu2LMu_3D`)."""

from typing import Tuple

import numpy as np


def enu2lmu_3d(E, nu) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Lame parameters ``(lambda, mu)`` for ``E`` and ``nu``.

    Elementwise, so scalars or per-cell arrays both work. `nu = 0.5` is
    incompressible and sends `lambda` to infinity; MRST does not guard it and
    neither does this, so the caller sees the same divide-by-zero.
    """
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


ENu2LMu_3D = enu2lmu_3d


if __name__ == "__main__":

    def test_scalar_against_closed_form() -> None:
        lam, mu = enu2lmu_3d(1.0, 0.25)
        assert np.isclose(lam, 0.4), lam
        assert np.isclose(mu, 0.4), mu
        print("ok test_scalar_against_closed_form")

    def test_vectorizes_over_cells() -> None:
        E = np.array([1.0, 2.0, 3.0])
        nu = np.array([0.25, 0.3, 0.2])
        lam, mu = enu2lmu_3d(E, nu)
        assert lam.shape == (3,) and mu.shape == (3,)
        assert np.allclose(mu, E / (2.0 * (1.0 + nu)))
        print("ok test_vectorizes_over_cells")

    def test_zero_poisson_ratio_gives_no_coupling() -> None:
        lam, mu = enu2lmu_3d(2.0, 0.0)
        assert np.isclose(lam, 0.0), lam
        assert np.isclose(mu, 1.0), mu
        print("ok test_zero_poisson_ratio_gives_no_coupling")

    def test_matches_the_bulk_modulus_identity() -> None:
        # K = lambda + 2 mu / 3 must equal E / (3 (1 - 2 nu))
        E, nu = 5.0, 0.3
        lam, mu = enu2lmu_3d(E, nu)
        assert np.isclose(lam + 2.0 * mu / 3.0, E / (3.0 * (1.0 - 2.0 * nu)))
        print("ok test_matches_the_bulk_modulus_identity")

    test_scalar_against_closed_form()
    test_vectorizes_over_cells()
    test_zero_poisson_ratio_gives_no_coupling()
    test_matches_the_bulk_modulus_identity()
    print("\nAll tests passed!")
