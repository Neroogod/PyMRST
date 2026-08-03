"""Structured explicit grid with hydraulic fractures on x-faces."""

import numpy as np

from pymrst.gridprocessing import tensor_grid


def mesh_hf_system(
    physdim,
    frac_centroids,
    *,
    num_stages: int,
    num_fracs_per_stage: int,
    nx_refine: int = 10,
    nx_refine_small: int = 2,
    ny: int = 11,
    nz: int = 11,
):
    """Port the coordinate-generation behavior of MRST ``meshHFsystem``."""
    dimensions = np.asarray(physdim, dtype=float).reshape(-1)
    centers = np.asarray(frac_centroids, dtype=float).reshape(-1)
    if dimensions.size != 3:
        raise ValueError("physdim must contain three values")
    expected = num_stages * num_fracs_per_stage
    if centers.size != expected:
        raise ValueError(f"expected {expected} fracture centroids")
    if min(nx_refine, nx_refine_small, ny, nz) < 2:
        raise ValueError("grid coordinate counts/refinement must be at least two")

    xparts: list[np.ndarray] = []
    previous = 0.0
    index = 0
    for stage in range(num_stages):
        xparts.append(np.linspace(previous, centers[index], nx_refine + 1))
        previous = centers[index]
        for local in range(1, num_fracs_per_stage):
            xparts.append(np.linspace(previous, centers[index + local], nx_refine_small + 1))
            previous = centers[index + local]
        index += num_fracs_per_stage
    xparts.append(np.linspace(previous, dimensions[0], nx_refine + 1))
    x = np.unique(np.concatenate(xparts))
    y = np.linspace(0.0, dimensions[1], ny)
    z = np.linspace(0.0, dimensions[2], nz)
    return tensor_grid(x, y, z)


__all__ = ["mesh_hf_system"]
