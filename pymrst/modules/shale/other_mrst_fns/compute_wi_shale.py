"""Fracture-completion well index (`computeWIShale`)."""

import numpy as np


def compute_wi_shale(
    permeability,
    aperture,
    radius,
    *,
    representative_radius=None,
    cell_side_lengths=None,
    skin=0.0,
    hydraulic_completion=True,
) -> np.ndarray:
    """Compute EDFM fracture well indices for rectangular fracture cells."""
    perm = np.asarray(permeability, dtype=float).reshape(-1)
    count = perm.size

    def expand(value, name):
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 1:
            array = np.full(count, array.item())
        if array.size != count:
            raise ValueError(f"{name} must be scalar or one value per completion")
        return array

    width = expand(aperture, "aperture")
    well_radius = expand(radius, "radius")
    skin_value = expand(skin, "skin")
    if representative_radius is None:
        if cell_side_lengths is None:
            raise ValueError(
                "representative_radius or rectangular cell_side_lengths is required"
            )
        sides = np.asarray(cell_side_lengths, dtype=float)
        if sides.ndim == 1:
            sides = sides.reshape(1, -1)
        if sides.shape == (1, 2):
            sides = np.repeat(sides, count, axis=0)
        if sides.shape != (count, 2):
            raise ValueError("cell_side_lengths must have shape (n, 2)")
        representative = 0.14 * np.sqrt(np.sum(sides**2, axis=1))
    else:
        representative = expand(representative_radius, "representative_radius")
    if np.any(perm <= 0) or np.any(width <= 0) or np.any(well_radius <= 0):
        raise ValueError("permeability, aperture and well radius must be positive")
    denominator = np.log(representative / well_radius) + skin_value
    if np.any(denominator <= 0):
        raise ValueError(
            "equivalent radius/skin produces a non-positive well-index denominator"
        )
    wi = 2.0 * np.pi * perm * width / denominator
    active = np.asarray(hydraulic_completion, dtype=bool).reshape(-1)
    if active.size == 1:
        active = np.full(count, active.item())
    if active.size != count:
        raise ValueError("hydraulic_completion must be scalar or per completion")
    wi[~active] = 0.0
    return wi


__all__ = ["compute_wi_shale"]
