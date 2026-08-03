"""Fracture-aligned tensor grid (MRST ``StructuredGridFractureonFace.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing.tensor_grid import tensor_grid


def structured_grid_fracture_on_face(
    physical_dimensions, fracture_centroids,
    *, num_stages: int, fractures_per_stage: int, fracture_spacing: float = 10,
    nx_refine: int = 10, nx_refine_small: int = 2, ny: int = 11, nz: int = 11,
    **_unused,
):
    """Build a structured grid whose x-faces contain every fracture centroid."""
    dims = np.asarray(physical_dimensions, dtype=float).reshape(3)
    centers = np.asarray(fracture_centroids, dtype=float).reshape(-1)
    if np.any(np.diff(centers) <= 0) or centers[0] <= 0 or centers[-1] >= dims[0]:
        raise ValueError("fracture centroids must increase strictly inside the x-domain")
    expected = int(num_stages) * int(fractures_per_stage)
    if centers.size != expected:
        raise ValueError("fracture centroid count must equal stages times fractures_per_stage")

    pieces = [np.linspace(0, centers[0], max(2, int(round(
        centers[0] / fracture_spacing * nx_refine))))]
    for index in range(1, centers.size):
        same_stage = index // fractures_per_stage == (index - 1) // fractures_per_stage
        count = nx_refine_small if same_stage else nx_refine
        pieces.append(np.linspace(centers[index - 1], centers[index], int(count)))
    pieces.append(np.linspace(centers[-1], dims[0], max(2, int(round(
        (dims[0] - centers[-1]) / fracture_spacing * nx_refine)))))
    x = np.unique(np.concatenate(pieces))
    y = np.linspace(0, dims[1], int(ny))
    z = np.linspace(0, dims[2], int(nz))
    return tensor_grid(x, y, z)


StructuredGridFractureonFace = structured_grid_fracture_on_face

__all__ = ["structured_grid_fracture_on_face", "StructuredGridFractureonFace"]
