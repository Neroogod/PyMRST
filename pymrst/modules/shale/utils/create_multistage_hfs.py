"""Multi-stage hydraulic-fracture geometry."""

import numpy as np

from pymrst.modules.shale.data import FracturePlane


def _per_stage(value, count: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, count)
    if values.size != count:
        raise ValueError(f"{name} must be scalar or contain {count} values")
    return values


def create_multistage_hfs(
    *,
    num_stages: int,
    frac_spacing,
    num_fracs_per_stage,
    frac_half_length,
    frac_height,
    cluster_spacing,
    frac_centroid1=(10.0, 20.0, 15.0),
    is_stencil: bool = False,
) -> tuple[list[FracturePlane], np.ndarray]:
    """Port MRST ``createMultiStageHFs`` rectangular-fracture placement."""
    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    counts_f = _per_stage(num_fracs_per_stage, num_stages, "num_fracs_per_stage")
    counts = counts_f.astype(np.int64)
    if np.any(counts < 1) or not np.allclose(counts, counts_f):
        raise ValueError("num_fracs_per_stage must contain positive integers")
    half = _per_stage(frac_half_length, num_stages, "frac_half_length")
    if is_stencil:
        half = half / 2.0
    height = _per_stage(frac_height, num_stages, "frac_height")
    cluster = _per_stage(cluster_spacing, num_stages, "cluster_spacing")
    spacing = np.asarray(frac_spacing, dtype=float).reshape(-1)
    required = max(num_stages - 1, 0)
    if required:
        if spacing.size == 1:
            spacing = np.repeat(spacing, required)
        if spacing.size != required:
            raise ValueError(f"frac_spacing must contain {required} values")

    center = np.asarray(frac_centroid1, dtype=float).copy()
    if center.shape != (3,):
        raise ValueError("frac_centroid1 must contain three coordinates")
    planes: list[FracturePlane] = []
    x_centers: list[float] = []
    for stage in range(num_stages):
        for frac in range(int(counts[stage])):
            if frac:
                center[0] += cluster[stage]
            x, y, z = center
            points = np.array([
                [x, y - half[stage], z - height[stage] / 2],
                [x, y + half[stage], z - height[stage] / 2],
                [x, y + half[stage], z + height[stage] / 2],
                [x, y - half[stage], z + height[stage] / 2],
            ])
            planes.append(FracturePlane(points, metadata={"stage": stage, "cluster": frac}))
            x_centers.append(float(x))
        if stage != num_stages - 1:
            center[0] += spacing[stage] - cluster[stage] * (counts[stage] - 1)
    return planes, np.asarray(x_centers)


__all__ = ["create_multistage_hfs"]
