"""Normalized comparison-engine certification."""

import numpy as np
import pytest

from pymrst.research_studio.validation import (
    ComparisonSeries, NamedAxis, NamedFieldSeries, TolerancePolicy,
    active_global_cell_mapping, compare_named_fields, compare_series,
    phase_mapping,
)


def test_comparison_aligns_time_names_units_and_signs():
    reference = ComparisonSeries(
        values=np.array([[1.0, -2.0], [2.0, -4.0], [3.0, -6.0]]),
        times=np.array([0.0, 1.0, 2.0]), entity_names=("INJ", "PROD"),
        unit="m3/s", sign_convention="positive_into_reservoir",
    )
    candidate = ComparisonSeries(
        # Reversed entity order, twice the temporal resolution, cm3/s units,
        # and the opposite simulator sign convention.
        values=-1.0e6*np.array([
            [-2.0, 1.0], [-3.0, 1.5], [-4.0, 2.0],
            [-5.0, 2.5], [-6.0, 3.0],
        ]),
        times=np.linspace(0.0, 2.0, 5), entity_names=("PROD", "INJ"),
        unit="cm3/s", sign_convention="positive_out_of_reservoir",
    )
    result = compare_series(
        reference, candidate, tolerance=TolerancePolicy(absolute=1e-12),
        unit_scale=1e-6, sign_scale=-1.0,
    )
    assert result.passed
    assert result.maximum_absolute_error < 1e-14
    assert result.compared_shape == (3, 2)


def test_comparison_reports_worst_location_and_tolerance_failure():
    reference = ComparisonSeries(
        [[1.0, 2.0], [1.0, 2.0]], [0.0, 1.0], ("A", "B"), unit="bar",
    )
    candidate = ComparisonSeries(
        [[1.0, 2.0], [1.01, 2.2]], [0.0, 1.0], ("A", "B"), unit="bar",
    )
    result = compare_series(
        reference, candidate,
        tolerance=TolerancePolicy(absolute=0.02, relative=0.01, scale=2.0),
    )
    assert not result.passed
    assert result.worst_time == 1.0
    assert result.worst_entity == "B"
    assert result.maximum_absolute_error == pytest.approx(0.2)


def test_comparison_rejects_ambiguous_or_invalid_mappings():
    ref = ComparisonSeries([[1.0]], [1.0], ("P1",), sign_convention="in")
    opposite = ComparisonSeries([[1.0]], [1.0], ("P1",), sign_convention="out")
    with pytest.raises(ValueError, match="sign conventions"):
        compare_series(ref, opposite)
    missing = ComparisonSeries([[1.0]], [1.0], ("P2",), sign_convention="in")
    with pytest.raises(ValueError, match="missing entities"):
        compare_series(ref, missing)
    uncovered = ComparisonSeries([[1.0]], [2.0], ("P1",), sign_convention="in")
    with pytest.raises(ValueError, match="time range"):
        compare_series(ref, uncovered)


def test_named_multidimensional_field_aligns_axis_order_cells_and_phases():
    times = np.array([0.0, 1.0])
    # Reference layout: time, cell, phase.
    reference_values = np.array([
        [[1.0, 10.0], [2.0, 20.0]],
        [[3.0, 30.0], [4.0, 40.0]],
    ])
    reference = NamedFieldSeries(
        reference_values, times,
        (NamedAxis("cell", ("4", "8")), NamedAxis("phase", ("water", "oil"))),
        unit="Pa", sign_convention="reservoir",
    )
    # Candidate layout: time, phase, cell, with an extra active cell and both
    # axes in different orders. Values are kPa.
    candidate_values = np.empty((2, 2, 3))
    candidate_values[:, 0, :] = np.array([[0.020, 0.999, 0.010], [0.040, 0.999, 0.030]])
    candidate_values[:, 1, :] = np.array([[0.002, 0.999, 0.001], [0.004, 0.999, 0.003]])
    candidate = NamedFieldSeries(
        candidate_values, times,
        (NamedAxis("phase", ("O", "W")), NamedAxis("cell", ("8", "99", "4"))),
        unit="kPa", sign_convention="reservoir",
    )
    result = compare_named_fields(
        reference, candidate, tolerance=TolerancePolicy(absolute=0.0),
        unit_scale=1000.0,
        mappings={
            "cell": active_global_cell_mapping([4, 8], [8, 99, 4]),
            "phase": phase_mapping(
                ["water", "oil"], ["O", "W"], aliases={"W": "water", "O": "oil"},
            ),
        },
    )
    assert result.passed
    assert result.compared_shape == (2, 2, 2)
    assert dict(result.worst_coordinates) == {"cell": "4", "phase": "water"}


def test_named_field_reports_all_worst_coordinates_and_rejects_axis_ambiguity():
    axes = (NamedAxis("cell", ("0", "1")), NamedAxis("phase", ("W", "O")))
    reference = NamedFieldSeries(np.zeros((1, 2, 2)), [1.0], axes)
    values = np.zeros((1, 2, 2))
    values[0, 1, 0] = 2.0
    candidate = NamedFieldSeries(values, [1.0], axes)
    result = compare_named_fields(reference, candidate)
    assert not result.passed
    assert result.worst_coordinates == (("cell", "1"), ("phase", "W"))
    wrong = NamedFieldSeries(
        np.zeros((1, 2, 2)), [1.0],
        (NamedAxis("cell", ("0", "1")), NamedAxis("component", ("A", "B"))),
    )
    with pytest.raises(ValueError, match="axis names differ"):
        compare_named_fields(reference, wrong)
    with pytest.raises(ValueError, match="must be unique"):
        NamedAxis("phase", ("W", "W"))
