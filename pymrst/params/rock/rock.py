"""Rock data structure for MRST `params/rock` ports."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Rock:
    """Petrophysical properties: permeability, porosity, and optional ntg."""

    perm: np.ndarray
    poro: np.ndarray
    ntg: Optional[np.ndarray] = None


if __name__ == "__main__":

    def test_rock_container() -> None:
        rock = Rock(perm=np.ones((2, 1)), poro=np.full(2, 0.2))
        assert rock.perm.shape == (2, 1)
        assert np.allclose(rock.poro, 0.2)
        assert rock.ntg is None
        print("ok test_rock_container")

    test_rock_container()
    print("\nAll tests passed!")

