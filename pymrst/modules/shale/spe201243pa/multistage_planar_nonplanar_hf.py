"""Multistage inclined fractures (MRST ``MulistagePlanarNonPlanarHF.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.modules.shale.data import FracturePlane


def _stage_values(value, count: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, count)
    if values.size != count:
        raise ValueError(f"{name} must be scalar or have one value per stage")
    return values


def multistage_planar_nonplanar_hf(
    *, num_stages: int, fracture_spacing, fractures_per_stage,
    fracture_half_length, fracture_height, theta: float,
    cluster_spacing, heel_coordinate=(10.0, 20.0, 15.0),
):
    """Create planar/inclined rectangular hydraulic-fracture stages."""
    stages = int(num_stages)
    if stages < 1:
        raise ValueError("num_stages must be positive")
    per_stage = _stage_values(fractures_per_stage, stages, "fractures_per_stage").astype(int)
    if np.any(per_stage < 1):
        raise ValueError("fractures_per_stage entries must be positive")
    half = _stage_values(fracture_half_length, stages, "fracture_half_length")
    height = _stage_values(fracture_height, stages, "fracture_height")
    cluster = _stage_values(cluster_spacing, stages, "cluster_spacing")
    spacing = np.asarray(fracture_spacing, dtype=float).reshape(-1)
    if stages == 1:
        spacing = np.empty(0)
    elif spacing.size == 1:
        spacing = np.repeat(spacing, stages - 1)
    if spacing.size != stages - 1:
        raise ValueError("fracture_spacing must have one value between each stage")

    centroid = np.asarray(heel_coordinate, dtype=float).reshape(3).copy()
    c, s = np.cos(float(theta)), np.sin(float(theta))
    planes, x_centers = [], []
    for stage in range(stages):
        for fracture in range(per_stage[stage]):
            if fracture:
                centroid[0] += cluster[stage]
            h2 = height[stage] / 2
            hl = half[stage]
            points = np.array([
                [centroid[0] - h2*c, centroid[1] - hl, centroid[2] + h2*s],
                [centroid[0] - h2*c, centroid[1] + hl, centroid[2] + h2*s],
                [centroid[0] + h2*c, centroid[1] + hl, centroid[2] - h2*s],
                [centroid[0] + h2*c, centroid[1] - hl, centroid[2] - h2*s],
            ])
            planes.append(FracturePlane(points, metadata={"stage": stage, "cluster": fracture}))
            x_centers.append(centroid[0])
        if stage < stages - 1:
            centroid[0] += spacing[stage]
    return planes, np.asarray(x_centers)


MulistagePlanarNonPlanarHF = multistage_planar_nonplanar_hf

__all__ = ["multistage_planar_nonplanar_hf", "MulistagePlanarNonPlanarHF"]
