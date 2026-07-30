"""Correctness and timing comparison for PyMRST AD backends.

This is the reusable, non-plotting counterpart of MRST's
``benchmarkAutoDiffBackends`` utility.  It deliberately benchmarks a compact
set of representative cell operations and verifies every backend against the
native sparse ADI result.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from pymrst.modules.ad_core.backends import (
    AutoDiffBackend,
    DiagonalAutoDiffBackend,
    SparseAutoDiffBackend,
)
from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    GenericAD,
    init_variables_diagonal,
)
from pymrst.utils.adi import init_variables_adi


@dataclass(frozen=True)
class BackendBenchmarkResult:
    """Measured operations and correctness metadata for one backend."""

    backend: str
    available: bool
    timings: dict[str, float]
    value_errors: dict[str, float]
    jacobian_errors: dict[str, float]
    reason: str = ""

    @property
    def correct(self) -> bool:
        return self.available and all(
            error == 0.0
            for errors in (self.value_errors, self.jacobian_errors)
            for error in errors.values()
        )

    @property
    def total_time(self) -> float:
        return float(sum(self.timings.values()))


def _initialize(backend: AutoDiffBackend, values: list[np.ndarray]):
    if isinstance(backend, DiagonalAutoDiffBackend):
        return init_variables_diagonal(*values, row_major=backend.row_major)
    if isinstance(backend, (SparseAutoDiffBackend, AutoDiffBackend)):
        return init_variables_adi(*values)
    raise TypeError(f"unsupported AD backend type: {type(backend).__name__}")


def _sparse_jacobians(value) -> list:
    if isinstance(value, GenericAD):
        return [block.to_sparse() for block in value.jac]
    return list(value.jac)


def _operations(variables) -> dict[str, Callable[[], object]]:
    x, y, z = variables[:3]
    numeric_y = np.asarray(y.val)
    subset = np.arange(0, len(x), max(1, len(x) // 7), dtype=np.int64)
    return {
        "cell_xy": lambda: x * y,
        "cell_xv": lambda: x * numeric_y,
        "cell_xy_2z": lambda: x * y + 2.0 * z,
        "cell_polynomial": lambda: x**2 + 0.5 * y * z,
        "subset": lambda: (x * y + z)[subset],
    }


def _evaluate(backend, values):
    variables = _initialize(backend, values)
    return {name: function() for name, function in _operations(variables).items()}


def _time_operations(backend, values, iterations):
    variables = _initialize(backend, values)
    timings = {}
    for name, function in _operations(variables).items():
        samples = []
        for _ in range(iterations):
            start = perf_counter()
            function()
            samples.append(perf_counter() - start)
        timings[name] = float(np.median(samples))
    return timings


def _maximum_errors(reference, candidate):
    value_errors = {}
    jacobian_errors = {}
    for name, expected in reference.items():
        actual = candidate[name]
        value_errors[name] = float(
            np.max(np.abs(np.asarray(actual.val) - np.asarray(expected.val)),
                   initial=0.0)
        )
        expected_jac = _sparse_jacobians(expected)
        actual_jac = _sparse_jacobians(actual)
        if len(expected_jac) != len(actual_jac):
            jacobian_errors[name] = float("inf")
            continue
        maximum = 0.0
        for lhs, rhs in zip(actual_jac, expected_jac):
            difference = (lhs - rhs).tocoo()
            if difference.nnz:
                maximum = max(maximum, float(np.max(np.abs(difference.data))))
        jacobian_errors[name] = maximum
    return value_errors, jacobian_errors


def benchmark_auto_diff_backends(
    backends: Iterable[AutoDiffBackend] | None = None,
    *,
    cell_count: int = 1_000,
    block_size: int = 5,
    iterations: int = 5,
    seed: int = 0,
) -> list[BackendBenchmarkResult]:
    """Compare representative AD operations across backend implementations.

    The first result is always checked against an independently evaluated
    sparse reference, so callers may freely reorder or omit backend variants.
    Unavailable optional backends are reported instead of raising.
    """

    if cell_count < 1 or block_size < 3 or iterations < 1:
        raise ValueError(
            "cell_count and iterations must be positive; block_size must be at least 3"
        )
    if backends is None:
        backends = (
            SparseAutoDiffBackend(),
            DiagonalAutoDiffBackend(use_mex=False),
            DiagonalAutoDiffBackend(
                name="Diagonal (RowMajor)", use_mex=False, row_major=True
            ),
        )
    backends = tuple(backends)
    rng = np.random.default_rng(seed)
    values = [rng.random(cell_count) for _ in range(block_size)]
    reference = _evaluate(SparseAutoDiffBackend(), values)
    results = []
    for backend in backends:
        if not isinstance(backend, AutoDiffBackend):
            raise TypeError("backends must contain AutoDiffBackend instances")
        if not backend.available:
            results.append(
                BackendBenchmarkResult(
                    backend=backend.label(),
                    available=False,
                    timings={},
                    value_errors={},
                    jacobian_errors={},
                    reason=backend.reason,
                )
            )
            continue
        candidate = _evaluate(backend, values)
        value_errors, jacobian_errors = _maximum_errors(reference, candidate)
        results.append(
            BackendBenchmarkResult(
                backend=backend.label(),
                available=True,
                timings=_time_operations(backend, values, iterations),
                value_errors=value_errors,
                jacobian_errors=jacobian_errors,
                reason=backend.reason,
            )
        )
    return results


# MATLAB-oriented alias for direct ports and notebooks.
benchmarkAutoDiffBackends = benchmark_auto_diff_backends

__all__ = [
    "BackendBenchmarkResult",
    "benchmark_auto_diff_backends",
    "benchmarkAutoDiffBackends",
]
