"""Disk-backed, immutable grid/property snapshots for bounded UI inspection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from ..database.models import AppDatabase
from ..schemas._common import nonempty, utc_now
from ..schemas.grid import GridArraySummary, GridSnapshot
from .project_service import ProjectService


def _member(owner: Any, name: str, default=None):
    return owner.get(name, default) if isinstance(owner, Mapping) else getattr(owner, name, default)


def _array_summary(name: str, value: np.ndarray) -> GridArraySummary:
    numeric = value.astype(float, copy=False).reshape(-1)
    finite = numeric[np.isfinite(numeric)]
    if finite.size:
        minimum, maximum, mean = map(float, (finite.min(), finite.max(), finite.mean()))
        samples = np.percentile(finite, (1, 10, 50, 90, 99))
        percentiles = {key: float(item) for key, item in zip(("p01", "p10", "p50", "p90", "p99"), samples)}
    else:
        minimum = maximum = mean = None
        percentiles = {}
    return GridArraySummary(
        name, tuple(value.shape), value.dtype.str, minimum, maximum, mean,
        percentiles, int(numeric.size - finite.size),
    )


class GridInspectionService:
    def __init__(
        self, database: AppDatabase, projects: ProjectService, *,
        maximum_cells: int = 100_000_000, maximum_slice_values: int = 100_000,
        maximum_generated_cells: int = 2_000_000,
    ) -> None:
        self.database, self.projects = database, projects
        self.maximum_cells = int(maximum_cells)
        self.maximum_slice_values = int(maximum_slice_values)
        # Generation runs synchronously on the request thread (it is pure NumPy,
        # not an isolated-process run), so its bound is much tighter than the
        # storage-level maximum_cells check in register().
        self.maximum_generated_cells = int(maximum_generated_cells)

    def register(
        self, project_id: str, grid: Any, *, model_id: str = "", name: str = "Grid",
        properties: Mapping[str, Any] | None = None, findings: tuple[Mapping[str, Any], ...] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> GridSnapshot:
        project = self.projects.get(project_id)
        cells, faces = _member(grid, "cells"), _member(grid, "faces")
        nodes = _member(grid, "nodes", None)
        if cells is None or faces is None:
            raise TypeError("grid must expose cells and faces")
        cell_count = int(_member(cells, "num", 0))
        if not 0 < cell_count <= self.maximum_cells:
            raise ValueError("grid cell count is empty or exceeds configured limit")
        centroids = np.asarray(_member(cells, "centroids", []), dtype=float)
        if centroids.ndim != 2 or centroids.shape[0] != cell_count or centroids.shape[1] not in (1, 2, 3):
            raise ValueError("cell centroids must have shape (cell_count, 1..3)")
        dimension = int(centroids.shape[1])
        volumes = np.asarray(_member(cells, "volumes", []), dtype=float).reshape(-1)
        if volumes.size not in (0, cell_count):
            raise ValueError("cell volumes must align with cell count")
        index_map = np.asarray(_member(cells, "index_map", np.arange(cell_count)), dtype=np.int64).reshape(-1)
        if index_map.size != cell_count:
            raise ValueError("active-cell index map must align with cell count")
        arrays: dict[str, np.ndarray] = {
            "CELL_CENTROIDS": np.ascontiguousarray(centroids),
            "ACTIVE_INDEX": np.ascontiguousarray(index_map),
        }
        if volumes.size:
            arrays["CELL_VOLUME"] = np.ascontiguousarray(volumes)
        for raw_name, raw_value in (properties or {}).items():
            key = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw_name).upper()).strip("_")
            if not key or key in arrays:
                raise ValueError(f"invalid or duplicate grid property name: {raw_name}")
            value = np.asarray(raw_value)
            if value.ndim == 0 or value.shape[0] != cell_count:
                raise ValueError(f"property {key} must align with cell count")
            if value.dtype.kind not in "biuf":
                raise TypeError(f"property {key} must be numeric")
            arrays[key] = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        for key, value in sorted(arrays.items()):
            digest.update(key.encode() + b"\0" + value.dtype.str.encode() + b"\0")
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        content_hash = digest.hexdigest()
        snapshot_id = "grid_" + hashlib.sha256(
            f"{project_id}:{model_id}:{content_hash}".encode()
        ).hexdigest()[:24]
        bounds = tuple((float(centroids[:, axis].min()), float(centroids[:, axis].max()))
                       for axis in range(dimension))
        summaries = tuple(_array_summary(key, value) for key, value in sorted(arrays.items()))
        created = utc_now()
        snapshot = GridSnapshot(
            snapshot_id, project_id, str(model_id), nonempty(name, "grid name"),
            content_hash, cell_count, int(_member(faces, "num", 0)),
            int(_member(nodes, "num", 0) if nodes is not None else 0), dimension,
            cell_count, bounds, summaries, tuple(findings), created,
            dict(provenance) if provenance is not None else {"kind": "unknown"},
        )
        destination = Path(project.root_path) / ".pymrst" / "grid_snapshots" / snapshot_id
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text())
            if manifest["content_hash"] != content_hash:
                raise RuntimeError("grid snapshot identifier collision")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=destination.parent))
            try:
                for key, value in arrays.items():
                    np.save(staging / f"{key}.npy", value, allow_pickle=False)
                (staging / "manifest.json").write_text(
                    json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n"
                )
                for path in staging.iterdir():
                    os.chmod(path, 0o444)
                staging.replace(destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_grid_snapshots VALUES(?,?,?,?,?,?,?)",
                (snapshot_id, project_id, model_id or None, snapshot.name,
                 content_hash, json.dumps(snapshot.to_dict()), created),
            )
        return snapshot

    def generate_cartesian(
        self, project_id: str, name: str, dims: Sequence[int], physical_dims_m: Sequence[float],
    ) -> GridSnapshot:
        """Build a uniform Cartesian grid from scratch and register it.

        This is the entry point for starting a case without an existing deck:
        the app's model-construction builder only needs a registered
        ``GridSnapshot`` to bind rock/fluid/well/schedule definitions to, and
        does not otherwise care how that grid was produced.
        """
        dims = [int(item) for item in dims]
        physical_dims_m = [float(item) for item in physical_dims_m]
        if len(dims) != 3 or len(physical_dims_m) != 3:
            raise ValueError("dims and physical_dims_m must each have exactly 3 entries")
        if any(item < 1 for item in dims):
            raise ValueError("grid dimensions must be at least 1 cell per axis")
        if any(not (item > 0) for item in physical_dims_m):
            raise ValueError("physical dimensions must be positive")
        cell_count = dims[0] * dims[1] * dims[2]
        if cell_count > self.maximum_generated_cells:
            raise ValueError(
                f"requested grid has {cell_count} cells, exceeding the "
                f"{self.maximum_generated_cells} synchronous generation limit"
            )
        from ...gridprocessing import cart_grid, compute_geometry
        grid = compute_geometry(cart_grid(dims, physical_dims_m))
        return self.register(project_id, grid, name=name, provenance={
            "kind": "cartesian", "dims": dims, "physical_dims_m": physical_dims_m,
        })

    def rebuild(self, snapshot_id: str) -> Any:
        """Deterministically reconstruct the full runnable grid topology.

        A ``GridSnapshot`` only stores bounded summary arrays for UI
        inspection; running a simulation needs the actual face/node
        connectivity, which is rebuilt here from the recorded provenance
        rather than stored a second time.
        """
        snapshot = self.get(snapshot_id)
        kind = str(snapshot.provenance.get("kind", "unknown"))
        if kind == "cartesian":
            from ...gridprocessing import cart_grid, compute_geometry
            grid = compute_geometry(cart_grid(
                snapshot.provenance["dims"], snapshot.provenance["physical_dims_m"],
            ))
            if int(grid.cells.num) != snapshot.cell_count:
                raise RuntimeError("rebuilt grid does not match its recorded snapshot")
            return grid
        raise ValueError(f"grid snapshot {snapshot_id} has no reconstructable provenance ({kind})")

    @staticmethod
    def _from_json(raw: str) -> GridSnapshot:
        data = json.loads(raw)
        data["bounds"] = tuple(tuple(item) for item in data["bounds"])
        data["arrays"] = tuple(GridArraySummary(
            item["name"], tuple(item["shape"]), item["dtype"], item["minimum"],
            item["maximum"], item["mean"], item["percentiles"], item["nonfinite_count"],
        ) for item in data["arrays"])
        data["findings"] = tuple(data["findings"])
        data.setdefault("provenance", {"kind": "unknown"})
        return GridSnapshot(**data)

    def get(self, snapshot_id: str) -> GridSnapshot:
        row = self.database.fetch_one(
            "SELECT summary_json FROM app_grid_snapshots WHERE snapshot_id=?", (snapshot_id,),
        )
        if row is None:
            raise KeyError(f"unknown grid snapshot: {snapshot_id}")
        return self._from_json(row[0])

    def list(self, project_id: str) -> tuple[GridSnapshot, ...]:
        self.projects.get(project_id)
        return tuple(self._from_json(row[0]) for row in self.database.fetch_all(
            "SELECT summary_json FROM app_grid_snapshots WHERE project_id=? ORDER BY created_at",
            (project_id,),
        ))

    def values(
        self, snapshot_id: str, array_name: str, *, offset: int = 0, count: int = 1000,
        stride: int = 1,
    ) -> dict[str, Any]:
        snapshot = self.get(snapshot_id)
        array_name = str(array_name).upper()
        known = {item.name for item in snapshot.arrays}
        if array_name not in known:
            raise KeyError(f"unknown grid array: {array_name}")
        offset, count, stride = int(offset), int(count), int(stride)
        if offset < 0 or not 1 <= count <= self.maximum_slice_values or stride < 1:
            raise ValueError("grid slice is outside configured bounds")
        project = self.projects.get(snapshot.project_id)
        path = Path(project.root_path) / ".pymrst" / "grid_snapshots" / snapshot_id / f"{array_name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        stop = min(array.shape[0], offset + count*stride)
        selected = np.asarray(array[offset:stop:stride])
        return {
            "snapshot_id": snapshot_id, "array": array_name, "offset": offset,
            "count": int(selected.shape[0]), "total": int(array.shape[0]),
            "stride": stride,
            "shape": list(selected.shape), "values": selected.tolist(),
        }

    def array(self, snapshot_id: str, array_name: str) -> np.ndarray:
        """Internal read-only array handle for trusted construction services."""
        snapshot = self.get(snapshot_id); array_name = str(array_name).upper()
        if array_name not in {item.name for item in snapshot.arrays}:
            raise KeyError(f"unknown grid array: {array_name}")
        project = self.projects.get(snapshot.project_id)
        path = Path(project.root_path) / ".pymrst" / "grid_snapshots" / snapshot_id / f"{array_name}.npy"
        return np.load(path, mmap_mode="r", allow_pickle=False)


__all__ = ["GridInspectionService"]
