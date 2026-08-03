"""Reference data loader for SPE-201243-PA workflows."""

from importlib.resources import files

import numpy as np


def load_cmg_timesteps() -> np.ndarray:
    """Return the bundled CMG report timesteps in days."""
    resource = files("pymrst.modules.shale.spe201243pa").joinpath(
        "data", "CMG_timestep2.csv"
    )
    with resource.open("r", encoding="utf-8") as stream:
        values = np.loadtxt(stream, delimiter=",", skiprows=1)
    return np.asarray(values, dtype=float).reshape(-1)


__all__ = ["load_cmg_timesteps"]
