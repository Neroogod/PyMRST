"""Load the original MAT assets distributed with MRST's shale module."""

from importlib.resources import files

from scipy.io import loadmat


_ASSETS = {
    "eagle_ford_edfm_grid": "EagleFordEDFMGrid.mat",
    "eagle_ford_fractures": "EagleFordFracs.mat",
    "eagle_ford_grid": "EagleFordGrid.mat",
    "eagle_ford_full_fractures": "EagleFordfullFracs.mat",
    "stochastic_fractures": "StochasticFracs.mat",
}


def available_shale_datasets() -> tuple[str, ...]:
    return tuple(_ASSETS)


def load_shale_dataset(name: str) -> dict:
    """Load a bundled reference asset with MATLAB structs simplified."""
    try:
        filename = _ASSETS[name.lower()]
    except KeyError as exc:
        choices = ", ".join(_ASSETS)
        raise ValueError(f"unknown shale dataset {name!r}; choose from {choices}") from exc
    resource = files("pymrst.modules.shale.examples").joinpath("data", filename)
    with resource.open("rb") as stream:
        raw = loadmat(stream, simplify_cells=True)
    return {key: value for key, value in raw.items() if not key.startswith("__")}


__all__ = ["available_shale_datasets", "load_shale_dataset"]
