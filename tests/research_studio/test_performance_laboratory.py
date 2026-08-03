"""M4.5/M4.6 measured accuracy, memory, sparse and solver evidence."""

import numpy as np
import scipy.sparse as sp

from pymrst.research_studio.performance import (
    BackendDescriptor, PerformanceLaboratory, SparsityKey,
    SparsityPatternCache, add_at_reduction, batched_evaluate,
    bincount_reduction, reference_solver_registry, sorted_segment_reduction,
    sparse_construction_benchmark,
)


def test_duplicate_reduction_candidates_match_reference_and_report_costs():
    rng = np.random.default_rng(4)
    indices = rng.integers(0, 100, size=10000)
    values = rng.normal(size=10000)
    laboratory = PerformanceLaboratory(
        repeats=3, absolute_tolerance=1e-12, relative_tolerance=1e-13)
    result = laboratory.qualify(
        "duplicate reduction",
        (BackendDescriptor("np.add.at", np.__version__, reference=True),
         add_at_reduction),
        {"bincount": (BackendDescriptor("bincount", np.__version__),
                      bincount_reduction),
         "sorted": (BackendDescriptor("sorted", np.__version__),
                    sorted_segment_reduction)},
        (indices, values, 100))
    assert result.passed
    assert max(item.maximum_absolute_error
               for item in result.measurements.values()) < 1e-12
    assert all(item.cold_seconds >= 0 and item.peak_bytes > 0
               for item in result.measurements.values())
    assert result.hardware["numpy"] == np.__version__


def test_sparse_construction_separates_conversion_construction_and_validation():
    matrix = sp.random(500, 500, density=.02, format="csr", random_state=2)
    evidence = sparse_construction_benchmark(matrix, repeats=3)
    assert set(evidence) == {"coo_to_csr", "canonical_csr_copy_false", "check_format"}
    assert all(value["mean_seconds"] >= 0 for value in evidence.values())


def test_reference_solvers_pass_and_optional_solvers_report_unavailability():
    diagonal = np.linspace(2.0, 3.0, 40)
    matrix = sp.diags((-np.ones(39), diagonal, -np.ones(39)), (-1, 0, 1),
                      format="csr")
    rhs = np.ones(40)
    evidence = reference_solver_registry().benchmark(matrix, rhs, tolerance=1e-9)
    assert evidence["scipy-direct"].passed
    assert evidence["scipy-cg"].passed
    assert evidence["scipy-gmres"].passed
    for name in ("petsc", "cupy", "amgx"):
        assert not evidence[name].passed
        assert "unavailable" in evidence[name].reason


def test_jacobian_pattern_refill_is_exact_and_buffers_are_isolated():
    matrix = sp.csr_matrix([[2.0, 0, -1], [0, 4, 3], [5, 0, 6]])
    key = SparsityKey("cartesian-3", "tpfa", ("pressure", "sw"), controls="bhp")
    cache = SparsityPatternCache(); pattern = cache.put(key, matrix)
    first = pattern.refill(np.arange(pattern.nonzeros, dtype=float))
    second = pattern.refill(np.arange(pattern.nonzeros, dtype=float)+10)
    first.data[:] = -1
    assert not np.array_equal(first.data, second.data)
    assert np.array_equal(first.indices, second.indices)
    assert cache.get(key).structural_hash == pattern.structural_hash
    cache.invalidate(controls="rate")
    assert len(cache) == 1
    cache.invalidate(topology="cartesian-3")
    assert len(cache) == 0


def test_batched_properties_match_single_vectorized_evaluation():
    pressure = np.linspace(1e5, 3e7, 1003)
    temperature = np.linspace(290, 370, 1003)
    property_function = lambda p, t: np.column_stack(
        (1.0 + 4e-10*(p-1e5), np.exp(-(t-290)/500)))
    expected = property_function(pressure, temperature)
    actual = batched_evaluate(property_function, pressure, temperature,
                              batch_size=97)
    np.testing.assert_array_equal(actual, expected)
