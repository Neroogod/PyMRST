"""Normalized member datasets with xarray/Zarr and lazy result access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymrst.research_studio.experiments.execution import LazyMemberResult


NORMALIZED_DATASET_SCHEMA_VERSION = 1


def _xarray():
    try:
        import xarray as xr
    except ImportError as error:
        raise ImportError(
            "xarray result conversion requires xarray; install pymrst[studio]") from error
    return xr


@dataclass(frozen=True, slots=True)
class NormalizedResultDataset:
    member: LazyMemberResult

    def to_xarray(self, *, categories=("fields", "regions", "wells",
                                       "convergence", "conservation")):
        xr = _xarray()
        coordinates = {"time": self.member.times}
        variables = {}
        for category in categories:
            for name in self.member.names(category):
                values = self.member.array(category, name)
                dims = ["time"]
                for axis, size in enumerate(values.shape[1:]):
                    dimension = f"{category}__{name}__axis{axis}"
                    dims.append(dimension)
                    coordinates[dimension] = np.arange(size, dtype=np.int64)
                variables[f"{category}__{name}"] = (tuple(dims), values)
        dataset = xr.Dataset(variables, coords=coordinates)
        dataset.attrs.update({
            "pymrst_schema_version": NORMALIZED_DATASET_SCHEMA_VERSION,
            "member_id": self.member.record["coordinate"]["member_id"],
            "experiment_id": self.member.record["coordinate"]["experiment_id"],
            "comparison_group": self.member.record["coordinate"]["comparison_group"],
        })
        return dataset


def write_zarr(path: str | Path, result: NormalizedResultDataset, *, mode="w") -> Path:
    try:
        import zarr  # noqa: F401
    except ImportError as error:
        raise ImportError("Zarr export requires zarr; install pymrst[studio]") from error
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Format 2 is the mature cross-tool interchange target. Zarr 3 support can
    # be added as an explicit option once consolidated metadata is standardized.
    result.to_xarray().to_zarr(
        target, mode=mode, consolidated=True, zarr_format=2)
    return target


def read_zarr(path: str | Path, *, lazy=True):
    xr = _xarray()
    # chunks=None uses the backend's lazy indexing without requiring dask.
    dataset = xr.open_zarr(Path(path), consolidated=True, chunks=None)
    return dataset if lazy else dataset.load()


__all__ = ["NORMALIZED_DATASET_SCHEMA_VERSION", "NormalizedResultDataset",
           "read_zarr", "write_zarr"]
