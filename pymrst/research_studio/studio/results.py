"""Secure lazy exposure of M3 member stores and generated reservoir viewers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from pymrst.research_studio.experiments import MemberResultStore


class ResultAccessError(ValueError):
    pass


def _within(path: Path, roots):
    resolved = path.resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ResultAccessError(f"result path lies outside configured roots: {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class StudioResult:
    result_id: str
    title: str
    store_path: str
    viewer_path: str = ""


class ResultCatalog:
    def __init__(self, allowed_roots: Iterable[str | Path], *,
                 maximum_json_values=200_000):
        if isinstance(allowed_roots, (str, Path)):
            allowed_roots = (allowed_roots,)
        roots = tuple(Path(value).resolve() for value in allowed_roots)
        if not roots:
            raise ValueError("result catalog requires at least one allowed root")
        self.roots = roots
        self.maximum_json_values = int(maximum_json_values)
        self._results = {}

    def register(self, result_id, title, store_path, *, viewer_path=""):
        if not result_id or result_id in self._results:
            raise KeyError(f"result id is empty or already registered: {result_id}")
        store = _within(Path(store_path), self.roots)
        if not (store / "member.json").is_file() or not (store / "arrays.npz").is_file():
            raise ResultAccessError("result directory is not a completed member store")
        viewer = ""
        if viewer_path:
            viewer_file = _within(Path(viewer_path), self.roots)
            if viewer_file.suffix.lower() != ".html" or not viewer_file.is_file():
                raise ResultAccessError("viewer must be an existing HTML file")
            viewer = str(viewer_file)
        value = StudioResult(str(result_id), str(title), str(store), viewer)
        self._results[value.result_id] = value
        return value

    def entries(self):
        return tuple(self._results[key] for key in sorted(self._results))

    def get(self, result_id):
        try:
            return self._results[result_id]
        except KeyError as error:
            raise KeyError(f"unknown result: {result_id}") from error

    def metadata(self, result_id):
        entry = self.get(result_id)
        result = MemberResultStore(entry.store_path).lazy()
        return {"result_id": entry.result_id, "title": entry.title,
                "member": result.record["coordinate"],
                "times": result.times.tolist(),
                "categories": {category: list(result.names(category))
                               for category in ("fields", "regions", "wells",
                                                "convergence", "conservation")},
                "viewer_available": bool(entry.viewer_path)}

    def array(self, result_id, category, name, *, time_index=None,
              start=0, stop=None):
        entry = self.get(result_id)
        result = MemberResultStore(entry.store_path).lazy()
        if category not in {"fields", "regions", "wells", "convergence", "conservation"}:
            raise ResultAccessError("invalid result category")
        if name not in result.names(category):
            raise ResultAccessError(f"unknown result quantity: {category}.{name}")
        if time_index is None:
            values = result.array(category, name)
        else:
            times = result.times
            if time_index < 0 or time_index >= times.size:
                raise ResultAccessError("time index is outside the result history")
            values = result.array(category, name, time_slice=time_index)
        flat = np.asarray(values).reshape(-1)
        stop = flat.size if stop is None else min(int(stop), flat.size)
        start = int(start)
        if start < 0 or stop < start:
            raise ResultAccessError("invalid result slice")
        selected = flat[start:stop]
        if selected.size > self.maximum_json_values:
            raise ResultAccessError(
                f"JSON slice has {selected.size} values; limit is {self.maximum_json_values}")
        return {"result_id": result_id, "category": category, "quantity": name,
                "time_index": time_index, "start": start, "stop": stop,
                "total_values": int(flat.size), "shape": list(np.asarray(values).shape),
                "dtype": str(np.asarray(values).dtype), "values": selected.tolist()}

    def viewer(self, result_id):
        entry = self.get(result_id)
        if not entry.viewer_path:
            raise ResultAccessError("result has no registered reservoir viewer")
        return Path(entry.viewer_path)


__all__ = ["ResultAccessError", "ResultCatalog", "StudioResult"]
