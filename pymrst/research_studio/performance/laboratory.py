"""M4.5 cold/warm, memory and numerical backend evidence."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import time
import tracemalloc
from typing import Callable, Mapping

import numpy as np
import scipy
import scipy.sparse as sp


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    name: str
    version: str
    device: str = "cpu"
    precision: str = "float64"
    compiler: str = "none"
    cache_state: str = "cold"
    reference: bool = False


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    backend: BackendDescriptor
    cold_seconds: float
    warm_seconds_mean: float
    warm_seconds_minimum: float
    peak_bytes: int
    maximum_absolute_error: float
    maximum_relative_error: float
    output_shape: tuple[int, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class BackendQualification:
    name: str
    reference_backend: str
    measurements: Mapping[str, BenchmarkMeasurement]
    hardware: Mapping[str, str]

    @property
    def passed(self):
        return all(item.passed for item in self.measurements.values())


class PerformanceLaboratory:
    def __init__(self, *, repeats=5, warmups=1, absolute_tolerance=0.0,
                 relative_tolerance=1e-12):
        if repeats <= 0 or warmups < 0:
            raise ValueError("benchmark repeats must be positive")
        self.repeats = int(repeats)
        self.warmups = int(warmups)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)

    @staticmethod
    def _call(function, arguments):
        value = function(*arguments)
        if sp.issparse(value):
            return value.toarray()
        return np.asarray(value)

    def qualify(self, name: str, reference: tuple[BackendDescriptor, Callable],
                candidates: Mapping[str, tuple[BackendDescriptor, Callable]],
                arguments: tuple):
        ref_descriptor, ref_function = reference
        reference_value = self._call(ref_function, arguments)
        measurements = {}
        for candidate_name, (descriptor, function) in candidates.items():
            tracemalloc.start()
            started = time.perf_counter()
            value = self._call(function, arguments)
            cold = time.perf_counter()-started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            for _ in range(self.warmups):
                self._call(function, arguments)
            timings = []
            for _ in range(self.repeats):
                started = time.perf_counter()
                value = self._call(function, arguments)
                timings.append(time.perf_counter()-started)
            if value.shape != reference_value.shape:
                raise ValueError(f"backend {candidate_name} output shape differs")
            absolute = np.abs(value-reference_value)
            denominator = np.maximum(np.abs(reference_value), 1.0)
            maximum_absolute = float(np.max(absolute)) if absolute.size else 0.0
            maximum_relative = float(np.max(absolute/denominator)) if absolute.size else 0.0
            passed = bool(np.all(absolute <= self.absolute_tolerance
                                  + self.relative_tolerance*np.abs(reference_value)))
            measurements[candidate_name] = BenchmarkMeasurement(
                descriptor, cold, float(np.mean(timings)), float(np.min(timings)),
                int(peak), maximum_absolute, maximum_relative,
                tuple(value.shape), passed)
        return BackendQualification(
            name, ref_descriptor.name, measurements,
            {"platform": platform.platform(), "numpy": np.__version__,
             "scipy": scipy.__version__, "processor": platform.processor()})


def add_at_reduction(indices, values, size):
    output = np.zeros(int(size), dtype=np.asarray(values).dtype)
    np.add.at(output, np.asarray(indices, dtype=np.int64), values)
    return output


def bincount_reduction(indices, values, size):
    return np.bincount(np.asarray(indices, dtype=np.int64),
                       weights=np.asarray(values), minlength=int(size))


def sorted_segment_reduction(indices, values, size):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values)
    order = np.argsort(indices, kind="stable")
    sorted_indices, sorted_values = indices[order], values[order]
    output = np.zeros(int(size), dtype=values.dtype)
    if sorted_indices.size:
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_indices))+1]
        output[sorted_indices[starts]] = np.add.reduceat(sorted_values, starts)
    return output


def sparse_construction_benchmark(matrix: sp.spmatrix, *, repeats=5):
    canonical = matrix.tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    timings = {"coo_to_csr": [], "canonical_csr_copy_false": [], "check_format": []}
    coo = canonical.tocoo()
    for _ in range(repeats):
        started = time.perf_counter()
        a = sp.coo_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape).tocsr()
        timings["coo_to_csr"].append(time.perf_counter()-started)
        started = time.perf_counter()
        b = sp.csr_matrix((canonical.data, canonical.indices, canonical.indptr),
                          shape=canonical.shape, copy=False)
        timings["canonical_csr_copy_false"].append(time.perf_counter()-started)
        started = time.perf_counter()
        b.check_format(full_check=True)
        timings["check_format"].append(time.perf_counter()-started)
        if (a != canonical).nnz or (b != canonical).nnz:
            raise ValueError("sparse construction candidate changed the matrix")
    return {name: {"mean_seconds": float(np.mean(values)),
                   "minimum_seconds": float(np.min(values))}
            for name, values in timings.items()}


__all__ = ["BackendDescriptor", "BackendQualification",
           "BenchmarkMeasurement", "PerformanceLaboratory", "add_at_reduction",
           "bincount_reduction", "sorted_segment_reduction",
           "sparse_construction_benchmark"]
