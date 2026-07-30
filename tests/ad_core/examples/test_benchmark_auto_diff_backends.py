"""Tests for the MRST ``benchmarkAutoDiffBackends`` counterpart."""

from __future__ import annotations

import pytest

from pymrst.modules.ad_core.backends import (
    AutoDiffBackend,
    DiagonalAutoDiffBackend,
    SparseAutoDiffBackend,
)
from pymrst.modules.ad_core.examples import benchmark_auto_diff_backends


def test_sparse_and_diagonal_backends_are_numerically_identical():
    results = benchmark_auto_diff_backends(
        (
            SparseAutoDiffBackend(),
            DiagonalAutoDiffBackend(use_mex=False),
            DiagonalAutoDiffBackend(
                name="Diagonal row-major", use_mex=False, row_major=True
            ),
        ),
        cell_count=17,
        block_size=4,
        iterations=2,
    )

    assert len(results) == 3
    for result in results:
        assert result.available
        assert result.correct
        assert result.total_time > 0.0
        assert set(result.timings) == {
            "cell_xy",
            "cell_xv",
            "cell_xy_2z",
            "cell_polynomial",
            "subset",
        }


def test_unavailable_backend_is_reported_without_execution():
    backend = AutoDiffBackend(
        name="Optional compiled backend", available=False, reason="not installed"
    )
    result = benchmark_auto_diff_backends(
        [backend], cell_count=3, iterations=1
    )[0]

    assert not result.available
    assert not result.correct
    assert result.reason == "not installed"
    assert result.timings == {}


@pytest.mark.parametrize(
    "options",
    [
        {"cell_count": 0},
        {"block_size": 2},
        {"iterations": 0},
    ],
)
def test_invalid_benchmark_sizes_are_rejected(options):
    with pytest.raises(ValueError):
        benchmark_auto_diff_backends(**options)
