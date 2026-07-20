"""Pore-volume calculation, ported from MRST `poreVolume`."""

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock.rock import Rock


def pore_volume(g: Grid, rock: Rock) -> np.ndarray:
    """Return pore volume per cell, scaled by `ntg` when present."""
    pv = rock.poro * g.cells.volumes
    if rock.ntg is not None:
        pv = pv * rock.ntg
    return pv


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock.make_rock import make_rock

    def test_pore_volume() -> None:
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 10.0]))
        rock = make_rock(g, 1.0, 0.25)
        assert np.allclose(pore_volume(g, rock), 0.25 * 10.0)
        print("ok test_pore_volume")

    def test_pore_volume_ntg() -> None:
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 10.0]))
        rock = make_rock(g, 1.0, 0.25, ntg=0.5)
        assert np.allclose(pore_volume(g, rock), 0.125 * 10.0)
        print("ok test_pore_volume_ntg")

    test_pore_volume()
    test_pore_volume_ntg()
    print("\nAll tests passed!")

