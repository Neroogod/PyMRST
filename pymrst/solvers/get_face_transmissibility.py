"""Face transmissibilities, ported from MRST `getFaceTransmissibility`."""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock
from pymrst.solvers.compute_trans import compute_trans


def get_face_transmissibility(
    g: Grid,
    rock: Optional[Rock] = None,
    t_half: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return harmonic reduction of one-sided transmissibilities per face."""
    if t_half is None:
        if rock is None:
            raise ValueError("give rock or t_half")
        t_half = compute_trans(g, rock)
    t_half = np.asarray(t_half, dtype=np.float64).ravel()
    return 1.0 / np.bincount(
        g.cells.faces,
        weights=1.0 / t_half,
        minlength=g.faces.num,
    )


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    def test_face_transmissibility_is_harmonic_mean() -> None:
        g = compute_geometry(cart_grid([4, 3, 2], [40.0, 30.0, 20.0]))
        rng = np.random.default_rng(0)
        perm = (10.0 + 990.0 * rng.random((g.cells.num, 1))) * MILLI * DARCY
        rock = make_rock(g, perm, np.full((g.cells.num, 1), 0.2))
        half = compute_trans(g, rock)
        face = get_face_transmissibility(g, rock)
        assert face.shape == (g.faces.num,)
        assert np.all(np.isfinite(face))
        assert np.all(face > 0)
        for f in range(g.faces.num):
            half_faces = np.flatnonzero(g.cells.faces == f)
            expected = 1.0 / np.sum(1.0 / half[half_faces])
            assert np.isclose(face[f], expected, rtol=1e-12)
        print("ok test_face_transmissibility_is_harmonic_mean")

    test_face_transmissibility_is_harmonic_mean()
    print("\nAll tests passed!")

