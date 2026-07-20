"""Build rock properties from Norne GRDECL arrays."""

from typing import Dict

import numpy as np


def norne_rock(g, grdecl: Dict, min_poro: float = 0.1):
    """Return rock properties on active Norne grid cells."""
    from pymrst.params.rock import make_rock

    index_map = g.cells.indexMap
    perm = np.column_stack([grdecl["PERMX"], grdecl["PERMY"], grdecl["PERMZ"]])[index_map]
    poro = np.maximum(grdecl["PORO"][index_map], min_poro)
    return make_rock(g, perm, poro[:, None])


if __name__ == "__main__":

    def test_norne_rock_imports() -> None:
        assert callable(norne_rock)
        print("ok test_norne_rock_imports")

    test_norne_rock_imports()
    print("\nAll tests passed!")

