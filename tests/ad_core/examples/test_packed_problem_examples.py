"""End-to-end tests for packed-problem demonstration workflows."""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.ad_core.examples import (
    demo_packed_problems_ad,
    ensemble_packed_problems_example_ad,
)


def test_single_packed_demo_runs_persists_and_collects(tmp_path):
    result = demo_packed_problems_ad(tmp_path)

    assert result.names == ("linear", "quadratic")
    assert result.final_values == (6.0, 12.0)
    assert result.complete_steps == (3, 3)
    assert all(np.array_equal(time, [1.0, 3.0, 6.0])
               for time in result.report_times)

    restarted = demo_packed_problems_ad(tmp_path)
    assert restarted.final_values == result.final_values
    assert restarted.complete_steps == result.complete_steps


def test_ensemble_demo_is_deterministic_and_complete(tmp_path):
    result = ensemble_packed_problems_example_ad(
        tmp_path, realization_count=4, seed=7
    )
    repeated = ensemble_packed_problems_example_ad(
        tmp_path, realization_count=4, seed=7
    )

    assert result.names == tuple(f"realization_{i}" for i in range(1, 5))
    assert result.complete_steps == (3, 3, 3, 3)
    assert np.allclose(result.final_values, repeated.final_values)
    assert all(np.array_equal(time, [1.0, 2.0, 4.0])
               for time in result.report_times)


def test_ensemble_requires_at_least_one_realization(tmp_path):
    with pytest.raises(ValueError):
        ensemble_packed_problems_example_ad(tmp_path, realization_count=0)
